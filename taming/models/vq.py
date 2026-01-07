import torch
import torch.nn.functional as F
import lightning as L
import torch.distributed as dist

from main import instantiate_from_config
from contextlib import contextmanager

from taming.modules.diffusionmodules.light_model import Encoder as LightEncoder, Decoder as LightDecoder
from taming.modules.diffusionmodules.model import Encoder as FullEncoder, Decoder as FullDecoder
from taming.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from taming.modules.util import requires_grad
from collections import OrderedDict
from taming.modules.ema import LitEma
from taming.modules.losses.lsw import lsw_dist

import numpy as np

class VQModel(L.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ### Quantize Related
                 quantconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate=None,
                 ### scheduler config
                 warmup_epochs=1.0, #warmup epochs
                 scheduler_type = "linear-warmup_cosine-decay",
                 accumulate_steps = 1,
                 min_learning_rate = 0,
                 use_ema = False,
                 stage = None,
                 lsw_loss_config = None,
                 ### model type config
                 model_type = "light",  # "light" or "full"
                 ):
        super().__init__()
        self.image_key = image_key
        
        # Select encoder and decoder based on model_type
        if model_type == "light":
            self.encoder = LightEncoder(**ddconfig)
            self.decoder = LightDecoder(**ddconfig)
        elif model_type == "full":
            # For full model, we need to adjust parameters
            # The full model requires additional parameters like attn_resolutions, dropout, resamp_with_conv
            # We'll set reasonable defaults
            full_ddconfig = ddconfig.copy()
            full_ddconfig.setdefault('attn_resolutions', [])
            full_ddconfig.setdefault('dropout', 0.0)
            full_ddconfig.setdefault('resamp_with_conv', True)
            self.encoder = FullEncoder(**full_ddconfig)
            self.decoder = FullDecoder(**full_ddconfig)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Must be 'light' or 'full'")
        
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = instantiate_from_config(quantconfig)
        self.use_ema = use_ema
        self.stage = stage
        self.model_type = model_type
        
        # Initialize LSW loss if config provided
        self.lsw_loss = None
        if lsw_loss_config is not None:
            from taming.modules.losses.lsw import LSWLoss
            self.lsw_loss = LSWLoss(**lsw_loss_config)
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema and stage is None: #no need to construct ema when training transformer
            self.model_ema = LitEma(self)
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.automatic_optimization = False
        self.accumulate_steps = accumulate_steps

        self.strict_loading = False

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        save the state_dict and filter out the 
        '''
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}
        
    def init_from_ckpt(self, path, ignore_keys=list(), stage=None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                ema_keys = set(k for k in sd.keys() if "model_ema" in k)
                
                for k, v in sd.items(): 
                    if "encoder" in k or "decoder" in k or "quant" in k:
                        if "model_ema" in k:
                            k_no_ema = k.replace("model_ema.", "")
                            new_k = ema_mapping[k_no_ema]
                            new_params[new_k] = v   
                        else:
                            s_name = k.replace('.', '')
                            ema_mapping[s_name] = k
                            ema_key = "model_ema." + s_name
                            if ema_key not in ema_keys:
                                new_params[k] = v  # directly load buffers and other parameters without EMA
                
                missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
                    elif "quant" in k:
                        new_params[k] = v
            missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)
        else: ## simple resume
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}:missing_keys={missing_keys}, unexpected_keys={unexpected_keys}")

    def encode(self, x):
        h = self.encoder(x)
        (quant, info), loss_breakdown = self.quantize(h)
        ### using token factorization the info is a tuple (each for embedding)
        return quant, torch.tensor(0.0), info, loss_breakdown

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, diff, indices, loss_break = self.encode(input)
        dec = self.decode(quant)
        for ind in indices.unique():
            self.codebook_count[ind] = 1
        return dec, diff, loss_break

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        h = self.encoder(x)
        (quant, indices), loss_break = self.quantize(h)
        xrec = self.decode(quant)
        for ind in indices.unique():
            self.codebook_count[ind] = 1

        opt_gen, opt_disc = self.optimizers()
        if self.scheduler_type != "None":
            scheduler_gen, scheduler_disc = self.lr_schedulers()

        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        # opt_gen._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        # opt_gen._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################
        
        # optimize generator
        aeloss, log_dict_ae = self.loss(torch.tensor(0.0), loss_break, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")

        # LSW regularization patch fuck trae
        if self.lsw_loss is not None:
            # Get ze_flat (encoded features before quantization)
            ze_flat = h.permute(0, 2, 3, 1).contiguous().view(-1, h.shape[1])
            codebook = self.quantize.embedding.weight

            lsw_loss = self.lsw_loss(ze_flat.detach(), codebook)
            
            aeloss += lsw_loss

        aeloss = aeloss / self.accumulate_steps
        self.manual_backward(aeloss)
        
        if (batch_idx + 1) % self.accumulate_steps == 0:
            opt_gen.step()
            opt_gen.zero_grad()
            if self.scheduler_type != "None":
                scheduler_gen.step()
        
        log_dict_ae["train/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
            
        # optimize discriminator
        discloss, log_dict_disc = self.loss(torch.tensor(0.0), loss_break, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        discloss = discloss / self.accumulate_steps
        self.manual_backward(discloss)
        
        if (batch_idx + 1) % self.accumulate_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()
            if self.scheduler_type != "None":
                scheduler_disc.step()
            
        #if torch.distributed.get_rank() == 0:
        #    print(log_dict_ae, log_dict_disc)

        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # Print reconstruction loss and perceptual loss every 100 steps
        if self.global_step % 100 == 0 and self.trainer.is_global_zero:
            reconstruct_loss = log_dict_ae.get("train/reconstruct_loss", torch.tensor(0.0)).item()
            perceptual_loss = log_dict_ae.get("train/perceptual_loss", torch.tensor(0.0)).item()
            # Print LSW loss if available
            if self.lsw_loss is not None:
                lsw_loss_value = lsw_loss.item() if isinstance(lsw_loss, torch.Tensor) else lsw_loss
                print(f"\nStep {self.global_step}: Reconstruction Loss = {reconstruct_loss:.6f}, Perceptual Loss = {perceptual_loss:.6f}, LSW Loss = {lsw_loss_value:.6f}")
            else:
                print(f"\nStep {self.global_step}: Reconstruction Loss = {reconstruct_loss:.6f}, Perceptual Loss = {perceptual_loss:.6f}")

    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)
            
    def on_train_epoch_start(self):
        self.codebook_count = [0] * self.quantize.n_e
        
    def on_validation_epoch_start(self):
        self.codebook_count = [0] * self.quantize.n_e

    def validation_step(self, batch, batch_idx): 
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        quant, eloss, indices, loss_break = self.encode(x)
        x_rec = self.decode(quant).clamp(-1, 1)
        aeloss, log_dict_ae = self.loss(torch.tensor(0.0), loss_break, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+ suffix)

        discloss, log_dict_disc = self.loss(torch.tensor(0.0), loss_break, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val" + suffix)
        
        for ind in indices.unique():
            self.codebook_count[ind] = 1
        log_dict_ae[f"val{suffix}/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
    
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        
        # Choose optimizer based on model_type
        if self.model_type == "full":
            # Use AdamW for full model with betas=(0.9, 0.95)
            opt_gen = torch.optim.AdamW(list(self.encoder.parameters())+
                                       list(self.decoder.parameters())+
                                       list(self.quantize.parameters()),
                                       lr=lr, betas=(0.9, 0.95))
            opt_disc = torch.optim.AdamW(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.9, 0.95))
        else:
            # Use Adam for light model with betas=(0.5, 0.9)
            opt_gen = torch.optim.Adam(list(self.encoder.parameters())+
                                      list(self.decoder.parameters())+
                                      list(self.quantize.parameters()),
                                      lr=lr, betas=(0.5, 0.9))
            opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                        lr=lr, betas=(0.5, 0.9))
        
        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))
        step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "None":
            return ({"optimizer": opt_gen}, {"optimizer": opt_disc})
    
        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
        else:
            raise NotImplementedError()
        return {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x