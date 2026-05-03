import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import einsum
from einops import rearrange
from collections import namedtuple
import math
from taming.modules.vqvae.qbridge import QBridge_models

LossBreakdown = namedtuple('LossBreakdown', ['per_sample_entropy', 'codebook_entropy', 'vqloss', 'avg_probs'])


class AffineTransform(nn.Module):
    def __init__(
        self,
        feature_size,
        use_running_statistics=False,
        momentum=0.1,
        lr_scale=1,
        num_groups=1,
    ):
        super().__init__()

        self.use_running_statistics = use_running_statistics
        self.num_groups = num_groups

        if use_running_statistics:
            self.momentum = momentum
            self.register_buffer('running_statistics_initialized', torch.zeros(1))
            self.register_buffer('running_ze_mean', torch.zeros(num_groups, feature_size))
            self.register_buffer('running_ze_var', torch.ones(num_groups, feature_size))
            self.register_buffer('running_c_mean', torch.zeros(num_groups, feature_size))
            self.register_buffer('running_c_var', torch.ones(num_groups, feature_size))
        else:
            self.scale = nn.parameter.Parameter(torch.zeros(num_groups, feature_size))
            self.bias = nn.parameter.Parameter(torch.zeros(num_groups, feature_size))
            self.lr_scale = lr_scale

    @torch.no_grad()
    def update_running_statistics(self, z_e, c):
        # Under-estimating z_e statistics slightly is empirically more stable
        # for straight-through estimation in some unnormalized bottlenecks.
        if self.training and self.use_running_statistics:
            unbiased = False

            ze_mean = z_e.mean([0, 1]).unsqueeze(0)
            ze_var = z_e.var([0, 1], unbiased=unbiased).unsqueeze(0)
            c_mean = c.mean([0]).unsqueeze(0)
            c_var = c.var([0], unbiased=unbiased).unsqueeze(0)

            if not self.running_statistics_initialized:
                self.running_ze_mean.data.copy_(ze_mean)
                self.running_ze_var.data.copy_(ze_var)
                self.running_c_mean.data.copy_(c_mean)
                self.running_c_var.data.copy_(c_var)
                self.running_statistics_initialized.fill_(1)
            else:
                self.running_ze_mean = (
                    self.momentum * ze_mean + (1 - self.momentum) * self.running_ze_mean
                )
                self.running_ze_var = (
                    self.momentum * ze_var + (1 - self.momentum) * self.running_ze_var
                )
                self.running_c_mean = (
                    self.momentum * c_mean + (1 - self.momentum) * self.running_c_mean
                )
                self.running_c_var = (
                    self.momentum * c_var + (1 - self.momentum) * self.running_c_var
                )

    def forward(self, codebook):
        scale, bias = self.get_affine_params()
        n, c = codebook.shape
        codebook = codebook.view(self.num_groups, -1, codebook.shape[-1])
        codebook = scale * codebook + bias
        return codebook.reshape(n, c)

    def get_affine_params(self):
        if self.use_running_statistics:
            scale = (self.running_ze_var / (self.running_c_var + 1e-8)).sqrt()
            bias = -scale * self.running_c_mean + self.running_ze_mean
        else:
            scale = 1.0 + self.lr_scale * self.scale
            bias = self.lr_scale * self.bias
        return scale.unsqueeze(1), bias.unsqueeze(1)


class GumbelQuantize(nn.Module):
    """
    credit to @karpathy: https://github.com/karpathy/deep-vector-quantization/blob/main/model.py (thanks!)
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_hiddens, e_dim, n_embed, straight_through=True,
                 kl_weight=5e-4, temp_init=1.0, use_vqinterface=True,
                 remap=None, unknown_index="random"):
        super().__init__()

        self.e_dim = e_dim
        self.n_embed = n_embed

        self.straight_through = straight_through
        self.temperature = temp_init
        self.kl_weight = kl_weight

        self.proj = nn.Conv2d(num_hiddens, n_embed, 1)
        self.embed = nn.Embedding(n_embed, e_dim)

        self.use_vqinterface = use_vqinterface

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_embed} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_embed

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, return_logits=False):
        # force hard = True when we are in eval mode, as we must quantize. actually, always true seems to work
        hard = self.straight_through if self.training else True
        temp = self.temperature if temp is None else temp

        logits = self.proj(z)
        if self.remap is not None:
            # continue only with used logits
            full_zeros = torch.zeros_like(logits)
            logits = logits[:,self.used,...]

        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=hard)
        if self.remap is not None:
            # go back to all entries but unused set to zero
            full_zeros[:,self.used,...] = soft_one_hot
            soft_one_hot = full_zeros
        z_q = einsum('b n h w, n d -> b d h w', soft_one_hot, self.embed.weight)

        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kl_weight * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        if self.remap is not None:
            ind = self.remap_to_used(ind)
        
        # Return consistent interface: ((z_q, indices), loss_breakdown)
        return (z_q, ind), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), diff, torch.tensor(0.0))

    def get_codebook_entry(self, indices, shape):
        b, h, w, c = shape
        assert b*h*w == indices.shape[0]
        indices = rearrange(indices, '(b h w) -> b h w', b=b, h=h, w=w)
        if self.remap is not None:
            indices = self.unmap_to_all(indices)
        one_hot = F.one_hot(indices, num_classes=self.n_embed).permute(0, 3, 1, 2).float()
        z_q = einsum('b n h w, n d -> b d h w', one_hot, self.embed.weight)
        return z_q


class VectorQuantizer(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, l2_normalize=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

        self.l2_normalize = l2_normalize

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_normalize:
            z_flattened_norm = torch.nn.functional.normalize(z.view(-1, self.e_dim))
            embedding_norm = torch.nn.functional.normalize(self.embedding.weight)

            d = torch.sum(z_flattened_norm ** 2, dim=1, keepdim=True) + \
                torch.sum(embedding_norm ** 2, dim=1) - 2 * \
                torch.einsum('b d, n d -> b n', z_flattened_norm, embedding_norm)
        else:
            d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
                torch.sum(self.embedding.weight**2, dim=1) - 2 * \
                torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)

        z_q = self.embedding(min_encoding_indices).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            vq_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            vq_loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])
        
        # Return consistent interface: ((z_q, indices), loss_breakdown)
        return (z_q, min_encoding_indices), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), vq_loss, torch.tensor(0.0))

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q



class VectorQuantizer1D(VectorQuantizer):
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h -> b h c').contiguous()
        assert z.shape[-1] == self.e_dim
        
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, self.embedding.weight).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            vq_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            vq_loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h c -> b c h').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return (z_q, min_encoding_indices), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), vq_loss, torch.tensor(0.0))


def _build_bridge_projector(
    e_dim,
    codebook_size,
    bridge_type="linear",
    bridge_model_name=None,
    bridge_num_layers=5,
    bridge_input_size=None,
    bridge_kwargs=None,
):
    bridge_kwargs = {} if bridge_kwargs is None else dict(bridge_kwargs)

    if bridge_model_name is None:
        bridge_type = bridge_type.lower()
        if bridge_type in ("identity", "none"):
            bridge_model_name = "Qbridge-none"
        elif bridge_type in ("linear", "lin", "single_linear"):
            bridge_model_name = "Qbridge-lin/1"
        elif bridge_type in ("mlp",):
            if bridge_num_layers == 1:
                bridge_model_name = "Qbridge-lin/1"
            elif bridge_num_layers == 5:
                bridge_model_name = "Qbridge-lin/5"
            else:
                raise ValueError(f"Unsupported MLP depth: {bridge_num_layers}. Only 1 and 5 are supported.")
        elif bridge_type in ("dit",):
            bridge_model_name = "QBridge-B/4"
        else:
            raise ValueError(f"Unsupported bridge_type: {bridge_type}")

    if bridge_model_name not in QBridge_models:
        raise ValueError(f"Unknown bridge_model_name: {bridge_model_name}")

    if bridge_input_size is None:
        root = int(math.isqrt(codebook_size))
        if root * root == codebook_size:
            bridge_input_size = root

    if "input_size" not in bridge_kwargs and bridge_input_size is not None:
        bridge_kwargs["input_size"] = bridge_input_size
    if "in_channels" not in bridge_kwargs:
        bridge_kwargs["in_channels"] = e_dim

    return QBridge_models[bridge_model_name](**bridge_kwargs), bridge_model_name


class BridgeVQ(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.25, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=True, bridge_type="linear",
                 bridge_model_name=None, bridge_num_layers=5,
                 bridge_input_size=None, bridge_kwargs=None):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
        for p in self.embedding.parameters():
            p.requires_grad = False

        self.embedding_proj, self.bridge_model_name = _build_bridge_projector(
            e_dim=self.e_dim,
            codebook_size=self.n_e,
            bridge_type=bridge_type,
            bridge_model_name=bridge_model_name,
            bridge_num_layers=bridge_num_layers,
            bridge_input_size=bridge_input_size,
            bridge_kwargs=bridge_kwargs,
        )
        self.bridge_type = bridge_type
    
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def get_quant_codebook(self):
        if self.bridge_model_name in ("Qbridge-none", "Qbridge-lin/1", "Qbridge-lin/5"):
            codebook = self.embedding.weight.unsqueeze(0).transpose(1, 2)
            codebook = self.embedding_proj(codebook)
            return codebook.squeeze(0).transpose(0, 1).contiguous()

        root = int(math.isqrt(self.n_e))
        if root * root != self.n_e:
            raise ValueError(
                f"Bridge model {self.bridge_model_name} requires a square codebook, got n_e={self.n_e}"
            )
        codebook = self.embedding.weight.transpose(0, 1).contiguous().view(1, self.e_dim, root, root)
        codebook = self.embedding_proj(codebook)
        return codebook.view(self.e_dim, self.n_e).transpose(0, 1).contiguous()

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        quant_codebook = self.get_quant_codebook()

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            vq_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            vq_loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])
            
        return (z_q, min_encoding_indices), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), vq_loss, torch.tensor(0.0))

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = F.embedding(indices, self.get_quant_codebook())

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q
    

class BridgeVQ1D(BridgeVQ):
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h -> b h c').contiguous()
        assert z.shape[-1] == self.e_dim
        
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        quant_codebook = self.get_quant_codebook()

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if not self.legacy:
            vq_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            vq_loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h c -> b c h').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return (z_q, min_encoding_indices), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), vq_loss, torch.tensor(0.0))


class SimVQ(BridgeVQ):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("bridge_type", "linear")
        kwargs.setdefault("bridge_model_name", "Qbridge-lin/1")
        super().__init__(*args, **kwargs)


class SimVQ1D(BridgeVQ1D):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("bridge_type", "linear")
        kwargs.setdefault("bridge_model_name", "Qbridge-lin/1")
        super().__init__(*args, **kwargs)


class AffineVQ(VectorQuantizer):
    def __init__(self, n_e, e_dim, beta=0.25, remap=None, unknown_index="random",
                 sane_index_shape=False, legacy=False, l2_normalize=False,
                 use_running_statistics=False, affine_momentum=0.1,
                 affine_lr_scale=1, num_groups=1):
        super().__init__(
            n_e=n_e,
            e_dim=e_dim,
            beta=beta,
            remap=remap,
            unknown_index=unknown_index,
            sane_index_shape=sane_index_shape,
            legacy=legacy,
            l2_normalize=l2_normalize,
        )
        self.affine_transform = AffineTransform(
            feature_size=e_dim,
            use_running_statistics=use_running_statistics,
            momentum=affine_momentum,
            lr_scale=affine_lr_scale,
            num_groups=num_groups,
        )

    def get_quant_codebook(self, z_flattened=None):
        codebook = self.embedding.weight
        if z_flattened is not None:
            self.affine_transform.update_running_statistics(z_flattened.unsqueeze(0), codebook)
        return self.affine_transform(codebook)

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"

        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)

        quant_codebook = self.get_quant_codebook(z_flattened)
        if self.l2_normalize:
            z_for_distance = torch.nn.functional.normalize(z_flattened)
            codebook_for_distance = torch.nn.functional.normalize(quant_codebook)
        else:
            z_for_distance = z_flattened
            codebook_for_distance = quant_codebook

        d = torch.sum(z_for_distance ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook_for_distance**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_for_distance, rearrange(codebook_for_distance, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)

        vq_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1)

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return (z_q, min_encoding_indices), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), vq_loss, torch.tensor(0.0))

    def get_codebook_entry(self, indices, shape):
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1)
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1)

        z_q = F.embedding(indices, self.get_quant_codebook())

        if shape is not None:
            z_q = z_q.view(shape)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q


class AffineVQ1D(AffineVQ):
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"

        z = rearrange(z, 'b c h -> b h c').contiguous()
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)

        quant_codebook = self.get_quant_codebook(z_flattened)
        if self.l2_normalize:
            z_for_distance = torch.nn.functional.normalize(z_flattened)
            codebook_for_distance = torch.nn.functional.normalize(quant_codebook)
        else:
            z_for_distance = z_flattened
            codebook_for_distance = quant_codebook

        d = torch.sum(z_for_distance ** 2, dim=1, keepdim=True) + \
            torch.sum(codebook_for_distance**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_for_distance, rearrange(codebook_for_distance, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)

        vq_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)

        z_q = z + (z_q - z).detach()
        z_q = rearrange(z_q, 'b h c -> b c h').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1)
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1)

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return (z_q, min_encoding_indices), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), vq_loss, torch.tensor(0.0))


class ASVQ(nn.Module):
    def __init__(self, n_e, e_dim, beta, remap=None, unknown_index="random",
                 sane_index_shape=False, fixed_cb=False, use_ema_scale=True, ema_decay=0.99, legacy=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        
        self.fixed_cb = fixed_cb
        self.use_ema_scale = use_ema_scale
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        nn.init.normal_(self.embedding.weight, mean=0, std=1)
        self.embedding.weight.requires_grad = not fixed_cb
        if self.use_ema_scale:
            self.register_buffer('scale', torch.ones(e_dim) * self.e_dim ** -0.5)
            self.register_buffer('ema_decay', torch.tensor(ema_decay))
        else:
            self.scale = nn.Parameter(torch.ones(e_dim) * self.e_dim ** -0.5)

        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.sane_index_shape = sane_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    @torch.no_grad()
    def update_scale(self, z):
        if not self.training:
            return
        batch_std = z.std(dim=0)  # [n]
        self.scale.mul_(self.ema_decay).add_(batch_std, alpha=1 - self.ema_decay)

    def get_norm_cb(self):
        weight = self.embedding.weight
        std = weight.std(dim=0, keepdim=True).detach()
        std = torch.clamp(std, min=1e-8)
        return weight / std

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)
        if self.use_ema_scale:
            self.update_scale(z_flattened)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        quant_codebook = self.scale * self.get_norm_cb() 
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if self.use_ema_scale and self.fixed_cb:
            vq_loss = torch.mean((z_q-z)**2)
        else:
            vq_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])
            
        return (z_q, min_encoding_indices), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), vq_loss, torch.tensor(0.0))

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        quant_codebook = self.scale * self.get_norm_cb()
        z_q = F.embedding(indices, quant_codebook)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class ASVQ1D(ASVQ):
    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h -> b h c').contiguous()
        assert z.shape[-1] == self.e_dim
        
        z_flattened = z.view(-1, self.e_dim)
        if self.use_ema_scale:
            self.update_scale(z_flattened)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        quant_codebook = self.scale * self.get_norm_cb() 
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)
        perplexity = None
        min_encodings = None

        # compute loss for embedding
        if self.use_ema_scale and self.fixed_cb:
            vq_loss = self.beta * torch.mean((z_q-z)**2)
        else:
            vq_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h c -> b c h').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.sane_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])

        return (z_q, min_encoding_indices), LossBreakdown(torch.tensor(0.0), torch.tensor(0.0), vq_loss, torch.tensor(0.0))
