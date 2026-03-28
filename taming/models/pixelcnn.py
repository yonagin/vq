import math

import lightning as L
import torch
import torch.nn.functional as F

from main import instantiate_from_config
from taming.modules.pixelcnn import GatedPixelCNN
from taming.modules.util import requires_grad


def disabled_train(self, mode=True):
    """Keep first-stage model in eval mode."""

    return self


class PixelCNNLightningModule(L.LightningModule):
    def __init__(
        self,
        first_stage_config,
        first_stage_key: str = "image",
        cond_stage_key: str = "class_label",
        dim: int = 64,
        n_layers: int = 15,
        n_classes: int = 0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["first_stage_config"])
        self.first_stage_config = first_stage_config
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

        self.first_stage_model = self.instantiate_first_stage(first_stage_config)
        vocab_size = self._infer_vocab_size()
        self.quant_dim = self._infer_quant_dim()
        self.latent_shape: tuple[int, ...] | None = None
        self.net = GatedPixelCNN(
            input_dim=vocab_size, dim=dim, n_layers=n_layers, n_classes=n_classes
        )

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        requires_grad(model, False)
        return model

    def _infer_vocab_size(self) -> int:
        quantizer = getattr(self.first_stage_model, "quantize", None)
        if quantizer is None:
            raise AttributeError("First-stage model must expose a quantizer module")
        for attr in ("n_e", "n_embed", "re_embed"):
            if hasattr(quantizer, attr):
                return int(getattr(quantizer, attr))
        if hasattr(quantizer, "embedding"):
            return int(quantizer.embedding.num_embeddings)
        if hasattr(quantizer, "embed"):
            return int(quantizer.embed.num_embeddings)
        raise AttributeError("Unable to infer vocab size from first-stage quantizer")

    def _infer_quant_dim(self) -> int:
        quantizer = self.first_stage_model.quantize
        if hasattr(quantizer, "e_dim"):
            return int(quantizer.e_dim)
        if hasattr(quantizer, "embedding"):
            return int(quantizer.embedding.embedding_dim)
        if hasattr(quantizer, "embed"):
            return int(quantizer.embed.embedding_dim)
        raise AttributeError("Unable to infer latent dimension from quantizer")

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens_to_grid(tokens)
        return self.net(tokens, labels)

    def get_input(self, batch, key):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).contiguous()
        if x.dtype == torch.double:
            x = x.float()
        return x

    @torch.no_grad()
    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        quant_z, _, indices, _ = self.first_stage_model.encode(x)
        self._set_latent_shape(quant_z)
        if isinstance(indices, (tuple, list)):
            indices = indices[0]
        if indices.dim() == 2:
            indices = indices.view(quant_z.shape[0], quant_z.shape[2], quant_z.shape[3])
        tokens = indices.long()
        return self._tokens_to_grid(tokens)

    def prepare_labels(
        self, batch, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        labels = batch.get(self.cond_stage_key)
        if labels is None:
            return torch.zeros(batch_size, device=device, dtype=torch.long)
        if labels.dim() > 1:
            labels = labels.view(labels.shape[0])
        return labels.to(device=device, dtype=torch.long)

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        with torch.no_grad():
            tokens = self.encode_tokens(x)
        labels = self.prepare_labels(batch, tokens.shape[0], tokens.device)
        grid_tokens = self._tokens_to_grid(tokens)
        logits = self(grid_tokens, labels)
        loss = F.cross_entropy(logits, grid_tokens)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    @torch.no_grad()
    def tokens_to_quant(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._tokens_to_grid(tokens)
        b, h, w = tokens.shape
        shape = (b, h, w, self.quant_dim)
        quant = self.first_stage_model.quantize.get_codebook_entry(
            tokens.view(-1), shape
        )
        return quant

    @torch.no_grad()
    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        quant = self.tokens_to_quant(tokens)
        decoded = self.first_stage_model.decode(quant)
        return decoded.clamp(-1, 1)

    def log_images(self, batch, **kwargs):
        x = self.get_input(batch, self.first_stage_key).to(self.device)
        with torch.no_grad():
            tokens = self.encode_tokens(x)
            recon = self.decode_tokens(tokens)
        return {"inputs": x.clamp(-1, 1), "reconstructions": recon}

    def _set_latent_shape(self, quant_z: torch.Tensor) -> None:
        if quant_z.dim() <= 2:
            self.latent_shape = ()
        else:
            self.latent_shape = tuple(int(d) for d in quant_z.shape[2:])

    def _tokens_to_grid(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.dim() == 3:
            return tokens
        if tokens.dim() != 2:
            raise ValueError(
                f"Expected tokens with 2 or 3 dimensions, got shape {tokens.shape}"
            )

        if self.latent_shape is None:
            raise RuntimeError(
                "Latent shape is unknown. Encode at least one batch or set input_dim explicitly."
            )

        if len(self.latent_shape) == 0:
            height = tokens.shape[1]
            width = 1
        elif len(self.latent_shape) == 1:
            height = self.latent_shape[0]
            width = 1
        else:
            height = self.latent_shape[0]
            width = int(math.prod(self.latent_shape[1:]))

        expected = height * width
        if tokens.shape[1] != expected:
            raise ValueError(
                f"Token sequence length {tokens.shape[1]} does not match latent grid {height}x{width}"
            )
        return tokens.view(tokens.shape[0], height, width)
