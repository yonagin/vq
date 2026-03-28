"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


### from https://huggingface.co/transformers/v3.2.0/_modules/transformers/generation_utils.html
def top_k_top_p_filtering(
    logits,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering"""
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = filter_value
    return logits


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size, config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[: config.n_unmasked, : config.n_unmasked] = 1
        self.register_buffer(
            "mask", mask.view(1, 1, config.block_size, config.block_size)
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if layer_past is None:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, layer_past=None, return_present=False):
        if return_present:
            assert not self.training
        attn, present = self.attn(self.ln1(x), layer_past=layer_past)

        x = x + attn
        x = x + self.mlp(self.ln2(x))
        if layer_past is not None or return_present:
            return x, present
        return x


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Adapts to unconditional runs.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob=0.0):
        super().__init__()
        self.base_num_classes = max(num_classes, 0)
        table_classes = self.base_num_classes if self.base_num_classes > 0 else 1
        self.embedding_table = nn.Embedding(table_classes, hidden_size)

    def forward(self, labels, train=False, force_drop_ids=None):
        labels = labels.squeeze(-1)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(
        self,
        vocab_size,
        block_size,
        n_layer=12,
        n_head=8,
        n_embd=256,
        cond_dim=256,
        embd_pdrop=0.0,
        resid_dropout_p=0.0,
        attn_dropout_p=0.0,
        ffn_dropout_p=0.1,
        drop_path_rate=0.0,
        n_unmasked=0,
        max_batch_size=32,
        max_seq_len=2048,
        class_num=1000,
        token_drop=0.1,
        cls_token_num=1,
        rope_base=10000,
        norm_eps=1e-5,
        ffn_dim_multiplier=None,
        initalizer_range=0.02,
        multiple_of=256,
        n_kv_head=None,
        shared_aln=False,
        alng=1e-3,
        use_pretrained_codebook=False,
        codebook_ckpt_path=None,
        n_codebook_embd=256,
    ):
        super().__init__()

        # ---- 将版本B的参数映射到版本A的内部参数 ----
        resid_pdrop = resid_dropout_p
        attn_pdrop = attn_dropout_p

        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_unmasked=n_unmasked,
        )

        # --- 保存版本B需要的额外参数到config ---
        self.cls_token_num = cls_token_num
        self.class_num = class_num
        self.use_pretrained_codebook = use_pretrained_codebook

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size + cls_token_num, config.n_embd)
        )
        self.drop = nn.Dropout(config.embd_pdrop)

        # class embedding (对齐版本B接口)
        self.class_emb = LabelEmbedder(class_num, config.n_embd)

        self.token_drop = nn.Dropout(token_drop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        """
        与版本B接口对齐。版本A使用 forward_with_past 管理缓存，
        这里仅做占位，不做实际操作。
        """
        pass

    def forward(self, idx, input_pos=None, mask=None, targets=None):
        """
        - idx: tuple (token_indices, class_labels)
              token_indices: [B, T] token索引
              class_labels:  [B] 或 [B, 1] 类别标签
        - input_pos: 未使用 (版本A使用绝对位置编码)
        - mask: 未使用 (版本A内部使用causal mask)
        - targets: [B, T] 可选的训练目标
        """
        idx, idx_cls = idx[0], idx[1]

        # token embeddings
        token_embeddings = self.tok_emb(idx)

        # class token embeddings (对齐版本B的cls prepend)
        cls_token_embeddings = self.class_emb(idx_cls, train=self.training)[
            :, : self.cls_token_num
        ]

        # 将cls token拼接在token前面
        token_embeddings = torch.cat([cls_token_embeddings, token_embeddings], dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size + self.cls_token_num, (
            "Cannot forward, model block size is exhausted."
        )
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def decode_tokens(self, idx, input_pos=None, targets=None, first_step=False):
        assert not self.training

        if first_step:
            # prefill: idx 是类别标签
            cls_token_embeddings = self.class_emb(idx, train=False)
            token_embeddings = cls_token_embeddings
        else:
            idx_tok, idx_cls = idx[0], idx[1]
            token_embeddings = self.tok_emb(idx_tok)
            # cls embedding 此处不再拼接 (已在prefill中处理)

        t = token_embeddings.shape[1]

        # 使用input_pos来获取位置编码
        if input_pos is not None:
            position_embeddings = self.pos_emb[:, input_pos, :]
        else:
            position_embeddings = self.pos_emb[:, :t, :]

        x = self.drop(token_embeddings + position_embeddings)

        # 无KV cache，直接全量前向 (简化版推理)
        # 注意：这里效率较低，但接口与版本B一致
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def forward_with_past(
        self, idx, embeddings=None, targets=None, past=None, past_length=None
    ):
        assert not self.training
        token_embeddings = self.tok_emb(idx)
        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        if past is not None:
            assert past_length is not None
            past = torch.cat(past, dim=-2)
            past_shape = list(past.shape)
            expected_shape = [
                self.config.n_layer,
                2,
                idx.shape[0],
                self.config.n_head,
                past_length,
                self.config.n_embd // self.config.n_head,
            ]
            assert past_shape == expected_shape, f"{past_shape} =/= {expected_shape}"
            position_embeddings = self.pos_emb[:, past_length, :]
        else:
            position_embeddings = self.pos_emb[:, : token_embeddings.shape[1], :]

        x = self.drop(token_embeddings + position_embeddings)
        presents = []
        for i, block in enumerate(self.blocks):
            x, present = block(
                x,
                layer_past=past[i, ...] if past is not None else None,
                return_present=True,
            )
            presents.append(present)

        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss, torch.stack(presents)


class DummyGPT(nn.Module):
    # for debugging
    def __init__(self, add_value=1):
        super().__init__()
        self.add_value = add_value

    def forward(self, idx):
        return idx + self.add_value, None


class CodeGPT(nn.Module):
    """Takes in semi-embeddings"""

    def __init__(
        self,
        vocab_size,
        block_size,
        in_channels,
        n_layer=12,
        n_head=8,
        n_embd=256,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        n_unmasked=0,
    ):
        super().__init__()
        config = GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            n_unmasked=n_unmasked,
        )
        # input embedding stem
        self.tok_emb = nn.Linear(in_channels, config.n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, idx, embeddings=None, targets=None):
        token_embeddings = self.tok_emb(idx)

        if embeddings is not None:
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


#### sampling utils (对齐版本B接口)


@torch.no_grad()
def sample(
    x,
    model,
    steps,
    temperature=1.0,
    sample_logits=True,
    top_k=None,
    top_p=None,
    cfg_scale=1.0,
    num_samples=16,
    sos_token=0,
    callback=None,
    token_factorization=False,
):
    """
    与版本B的 sample 接口完全对齐。

    - x: 类别标签 [B, 1] 或 None
    - model: GPT模型实例
    - steps: 生成步数
    - cfg_scale: 未使用 (版本A不支持CFG, 忽略)
    - token_factorization: 未使用, 忽略
    """
    if x is None:
        device = next(model.parameters()).device
        x = torch.full((num_samples, 1), sos_token, device=device, dtype=torch.long)

    bs, _ = x.shape
    device = x.device

    # cfg_scale > 1.0 的逻辑在版本A中不支持，退化为无条件
    # 但保持接口一致
    cond_token = x
    sample_seq = x.clone()

    cond_len = x.shape[1]
    max_seq_length = cond_len + steps

    # setup_caches 在版本A中为空操作
    model.setup_caches(
        max_batch_size=bs,
        max_seq_length=max_seq_length,
        dtype=next(model.parameters()).dtype,
    )

    for n in range(steps):
        if callback is not None:
            callback(n)

        if n == 0:  # prefill
            input_pos = torch.arange(0, cond_len, device=device)
        elif n == 1:
            input_pos = torch.tensor([cond_len], device=device)
        else:
            input_pos = input_pos + 1

        logits, _ = model.decode_tokens(x, input_pos=input_pos, first_step=(n == 0))

        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            if top_k > 0 or (top_p is not None and top_p < 1.0):
                logits = top_k_top_p_filtering(
                    logits, top_k=top_k, top_p=top_p if top_p is not None else 1.0
                )

        probs = F.softmax(logits, dim=-1)

        if not sample_logits:
            _, x = torch.topk(probs, k=1, dim=-1)
        else:
            x = torch.multinomial(probs, num_samples=1)

        sample_seq = torch.cat((sample_seq, x), dim=1)

        # 构造下一步输入 (token, cls_label) 元组
        x = (x, cond_token)

    sample_seq = sample_seq[:, cond_len:]  # cut conditioning off
    return sample_seq


#### clustering utils (保持不变)


class KMeans(nn.Module):
    def __init__(self, ncluster=512, nc=3, niter=10):
        super().__init__()
        self.ncluster = ncluster
        self.nc = nc
        self.niter = niter
        self.shape = (3, 32, 32)
        self.register_buffer("C", torch.zeros(self.ncluster, nc))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))

    def is_initialized(self):
        return self.initialized.item() == 1

    @torch.no_grad()
    def initialize(self, x):
        N, D = x.shape
        assert D == self.nc, D
        c = x[torch.randperm(N)[: self.ncluster]]
        for i in range(self.niter):
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            c = torch.stack([x[a == k].mean(0) for k in range(self.ncluster)])
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            print(
                "done step %d/%d, re-initialized %d dead clusters"
                % (i + 1, self.niter, ndead)
            )
            c[nanix] = x[torch.randperm(N)[:ndead]]

        self.C.copy_(c)
        self.initialized.fill_(1)

    def forward(self, x, reverse=False, shape=None):
        if not reverse:
            bs, c, h, w = x.shape
            assert c == self.nc
            x = x.reshape(bs, c, h * w, 1)
            C = self.C.permute(1, 0)
            C = C.reshape(1, c, 1, self.ncluster)
            a = ((x - C) ** 2).sum(1).argmin(-1)
            return a
        else:
            bs, HW = x.shape
            x = self.C[x]
            x = x.permute(0, 2, 1)
            shape = shape if shape is not None else self.shape
            x = x.reshape(bs, *shape)
            return x
