import torch
import torch.nn as nn
import torch.nn.functional as F


def _weights_init(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if "Conv" in classname:
        if hasattr(module, "weight") and module.weight is not None:
            nn.init.xavier_uniform_(module.weight.data)
        if hasattr(module, "bias") and module.bias is not None:
            module.bias.data.zero_()


class GatedActivation(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, dim=1)
        return torch.tanh(x) * torch.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(
        self,
        mask_type: str,
        dim: int,
        kernel: int,
        residual: bool = True,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        if kernel % 2 == 0:
            raise ValueError("Kernel size must be odd for causal masking")
        self.mask_type = mask_type
        self.residual = residual
        self.num_classes = max(int(n_classes), 1)

        self.class_cond_embedding = nn.Embedding(self.num_classes, 2 * dim)

        vert_kernel = (kernel // 2 + 1, kernel)
        vert_padding = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(dim, dim * 2, vert_kernel, padding=vert_padding)
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        horiz_kernel = (1, kernel // 2 + 1)
        horiz_padding = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(dim, dim * 2, horiz_kernel, padding=horiz_padding)
        self.horiz_resid = nn.Conv2d(dim, dim, 1)
        self.gate = GatedActivation()

    def make_causal(self) -> None:
        self.vert_stack.weight.data[:, :, -1].zero_()
        self.horiz_stack.weight.data[:, :, :, -1].zero_()

    def forward(
        self,
        x_v: torch.Tensor,
        x_h: torch.Tensor,
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mask_type == "A":
            self.make_causal()

        labels = labels.view(labels.shape[0]) if labels.dim() > 1 else labels
        cond = self.class_cond_embedding(labels)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, : x_v.size(-1), :]
        out_v = self.gate(h_vert + cond[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, : x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)
        out = self.gate(v2h + h_horiz + cond[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)
        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(
        self,
        input_dim: int = 256,
        dim: int = 64,
        n_layers: int = 15,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(input_dim, dim)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            mask_type = "A" if i == 0 else "B"
            kernel = 7 if i == 0 else 3
            residual = i != 0
            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, input_dim, 1),
        )
        self.apply(_weights_init)

    def forward(self, tokens: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shp = tokens.size() + (-1,)
        x = self.embedding(tokens.view(-1)).view(shp)
        x = x.permute(0, 3, 1, 2)
        x_v, x_h = x, x
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h, labels)
        return self.output_conv(x_h)

    @torch.no_grad()
    def generate(
        self,
        labels: torch.Tensor,
        shape: tuple[int, int],
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        labels = labels.view(labels.shape[0]) if labels.dim() > 1 else labels
        device = next(self.parameters()).device
        samples = torch.zeros(
            (labels.shape[0], shape[0], shape[1]), dtype=torch.long, device=device
        )
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(samples, labels)
                logits = logits[:, :, i, j] / max(temperature, 1e-4)
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    thresh = torch.topk(logits, k)[0][..., -1, None]
                    logits = logits.masked_fill(logits < thresh, -float("inf"))
                if 0.0 < top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(
                        sorted_logits.softmax(dim=-1), dim=-1
                    )
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                        ..., :-1
                    ].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        -1, sorted_indices, sorted_indices_to_remove
                    )
                    logits = logits.masked_fill(indices_to_remove, -float("inf"))
                probs = F.softmax(logits, dim=-1)
                samples[:, i, j] = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return samples
