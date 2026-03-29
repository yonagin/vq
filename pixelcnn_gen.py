"""PixelCNN sampling CLI following llama_gen conventions."""

import argparse
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from taming.models.pixelcnn import PixelCNN


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def chw_to_pillow(x: torch.Tensor) -> Image.Image:
    arr = ((x.clamp(-1, 1) + 1.0) / 2.0).detach().cpu().numpy().transpose(1, 2, 0)
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PixelCNN sampler")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("-o", "--outdir", default="pixelcnn_samples", type=str)
    parser.add_argument("-n", "--num_samples", default=64, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("-k", "--top_k", default=0, type=int)
    parser.add_argument("-p", "--top_p", default=1.0, type=float)
    parser.add_argument("--height", default=16, type=int)
    parser.add_argument("--width", default=16, type=int)
    parser.add_argument("--class_label", default=None, type=int)
    return parser


def load_model(config_path: str, ckpt_path: str) -> PixelCNN:
    config = OmegaConf.load(config_path)
    model = PixelCNN(**config.model.init_args)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"], strict=False)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    model = load_model(args.config, args.ckpt)
    os.makedirs(args.outdir, exist_ok=True)

    total = args.num_samples
    batches = [args.batch_size for _ in range(total // args.batch_size)]
    if total % args.batch_size:
        batches.append(total % args.batch_size)

    print(f"Writing samples to {args.outdir}")
    sample_idx = 0
    for bs in tqdm(batches, desc="Sampling"):
        labels = torch.full(
            (bs,), args.class_label or 0, device=DEVICE, dtype=torch.long
        )
        samples = model.net.generate(
            labels,
            shape=(args.height, args.width),
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        if hasattr(model, "decode_tokens"):
            decoded = model.decode_tokens(samples)
            for i in range(bs):
                pil_img = chw_to_pillow(decoded[i])
                pil_img.save(os.path.join(args.outdir, f"{sample_idx + i:06}.png"))
        else:
            for i in range(bs):
                grid = samples[i].detach().cpu().numpy()
                grid = (grid / grid.max()) if grid.max() > 0 else grid
                grid = (grid * 255).astype(np.uint8)
                pil_img = Image.fromarray(grid)
                pil_img.save(os.path.join(args.outdir, f"{sample_idx + i:06}.png"))
        sample_idx += bs


def main() -> None:
    sys.path.append(os.getcwd())
    parser = get_parser()
    args = parser.parse_args()
    start = time.time()
    run(args)
    print(f"Done in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
