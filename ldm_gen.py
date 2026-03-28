"""Batch image generation script for Latent Diffusion models.

This utility mirrors the CLI ergonomics of ``llama_gen.py`` while reusing the
sampling APIs defined inside ``ldm.models.diffusion.ddpm`` and
``ldm.models.diffusion.ddim``. It loads a pre-trained Latent Diffusion model
from a Lightning checkpoint plus config, optionally applies prompt/class
conditioning, and saves decoded images to disk.
"""

import argparse
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config


try:
    import torch_npu  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch_npu = None


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch, "npu") and torch_npu is not None and torch_npu.npu.is_available():
        return torch.device("npu:0")
    return torch.device("cpu")


DEVICE = pick_device()
RESCALE = lambda x: (x + 1.0) / 2.0


def load_model(config_path: str, ckpt_path: str, device: torch.device) -> Any:
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    if ckpt_path:
        print(f"Loading weights from {ckpt_path}")
        pl_sd = torch.load(ckpt_path, map_location="cpu")
        state_dict = pl_sd.get("state_dict", pl_sd)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {missing}")
        if unexpected:
            print(f"Unexpected keys: {unexpected}")
    model.to(device)
    model.eval()
    if hasattr(model, "cond_stage_model") and model.cond_stage_model is not None:
        model.cond_stage_model.eval()
    return model


def chw_to_pil(x: torch.Tensor) -> Image.Image:
    arr = x.detach().cpu().clamp(-1, 1)
    arr = RESCALE(arr.numpy().transpose(1, 2, 0)).clip(0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def repeat_to_length(values: Sequence[Any], total: int) -> List[Any]:
    if not values:
        raise ValueError("Expected at least one conditioning value")
    if len(values) >= total:
        return list(values[:total])
    reps = math.ceil(total / len(values))
    expanded = list(values) * reps
    return expanded[:total]


def load_lines(path: Optional[str], cast_fn=str) -> List[Any]:
    if not path:
        return []
    lines: List[Any] = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            lines.append(cast_fn(line))
    return lines


def determine_conditioning_kind(model: Any) -> str:
    cond_key = getattr(model, "cond_stage_key", None)
    conditioning_key = getattr(model.model, "conditioning_key", None)
    if conditioning_key is None or cond_key is None:
        return "none"
    cond_key = str(cond_key).lower()
    if cond_key in {"caption", "text", "prompt", "prompts", "txt"}:
        return "text"
    if cond_key == "class_label":
        return "class"
    raise ValueError(
        f"Unsupported conditioning key '{model.cond_stage_key}'. "
        "Only text and class-label conditions are handled by this script."
    )


def plan_condition_sequences(
    kind: str, args: argparse.Namespace, total: int
) -> Dict[str, List[Any]]:
    if kind == "none":
        return {}
    if kind == "text":
        base_prompts: List[str] = []
        if args.prompt:
            base_prompts.append(args.prompt)
        base_prompts.extend(load_lines(args.prompts_file, str))
        if not base_prompts:
            raise ValueError(
                "Text-conditioned models require --prompt or --prompts_file"
            )
        prompts = repeat_to_length(base_prompts, total)
        negative_base: List[str] = []
        if args.negative_prompt:
            negative_base.append(args.negative_prompt)
        negative_base.extend(load_lines(args.negative_prompts_file, str))
        if args.cfg_scale > 1.0 and not negative_base:
            negative_base = [""]
        negatives = repeat_to_length(negative_base, total) if negative_base else []
        return {"main": prompts, "negative": negatives}
    if kind == "class":
        class_values: List[int] = []
        if args.class_label is not None:
            class_values.append(int(args.class_label))
        class_values.extend(load_lines(args.class_labels_file, int))
        if not class_values:
            raise ValueError(
                "Class-conditioned models require --class_label or --class_labels_file"
            )
        labels = repeat_to_length(class_values, total)
        negative_labels: List[int] = []
        if args.uncond_class_label is not None:
            negative_labels.append(int(args.uncond_class_label))
        if args.cfg_scale > 1.0 and not negative_labels:
            raise ValueError(
                "--cfg_scale > 1.0 requires --uncond_class_label for class-conditioned models"
            )
        negatives = repeat_to_length(negative_labels, total) if negative_labels else []
        return {"main": labels, "negative": negatives}
    raise ValueError(f"Unknown conditioning kind '{kind}'")


def build_condition_batch(model: Any, kind: str, entries: List[Any]) -> Optional[Any]:
    if kind == "none":
        return None
    if kind == "text":
        return model.get_learned_conditioning(entries)
    if kind == "class":
        tensor = torch.tensor(
            entries, device=model.device if hasattr(model, "device") else DEVICE
        )
        tensor = tensor.long()
        return model.get_learned_conditioning(tensor)
    raise ValueError(f"Unsupported conditioning kind '{kind}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent Diffusion batch sampler")
    parser.add_argument(
        "--config", required=True, type=str, help="Path to model config (yaml)"
    )
    parser.add_argument(
        "--ckpt", required=True, type=str, help="Path to Lightning checkpoint"
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default="ldm_samples",
        help="Output directory for PNGs",
    )
    parser.add_argument(
        "-n",
        "--num_samples",
        type=int,
        default=16,
        help="Total number of images to draw",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Samples per batch")
    parser.add_argument(
        "--sampler", choices=["ddim", "plms"], default="ddim", help="Sampling algorithm"
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Number of diffusion steps"
    )
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM eta (noise scale)")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature for noise"
    )
    parser.add_argument(
        "--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single text prompt (text-conditioned models)",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default=None,
        help="File with newline-delimited prompts",
    )
    parser.add_argument(
        "--negative_prompt", type=str, default=None, help="Negative prompt when cfg > 1"
    )
    parser.add_argument(
        "--negative_prompts_file",
        type=str,
        default=None,
        help="File with newline-delimited negative prompts",
    )
    parser.add_argument(
        "--class_label",
        type=int,
        default=None,
        help="Class index for class-conditioned models",
    )
    parser.add_argument(
        "--class_labels_file",
        type=str,
        default=None,
        help="File with one class index per line",
    )
    parser.add_argument(
        "--uncond_class_label",
        type=int,
        default=None,
        help="Label to represent null class for CFG",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--log_every_t",
        type=int,
        default=10,
        help="Interval for saving DDIM intermediates",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    sys.path.append(os.getcwd())
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    model = load_model(args.config, args.ckpt, DEVICE)

    sampler = DDIMSampler(model) if args.sampler == "ddim" else PLMSSampler(model)
    shape = (model.channels, model.image_size, model.image_size)

    os.makedirs(args.outdir, exist_ok=True)
    total = args.num_samples
    batches = [args.batch_size for _ in range(total // args.batch_size)]
    if total % args.batch_size:
        batches.append(total % args.batch_size)

    cond_kind = determine_conditioning_kind(model)
    sequences = plan_condition_sequences(cond_kind, args, total)

    sample_shape_str = "x".join(str(x) for x in shape)
    print(
        f"Running {args.sampler.upper()} sampler on device {DEVICE} with shape {sample_shape_str} "
        f"for {args.steps} steps"
    )

    sample_idx = 0
    start_time = time.time()
    with model.ema_scope() if hasattr(model, "ema_scope") else torch.no_grad():
        for batch_id, bs in enumerate(tqdm(batches, desc="Sampling")):
            if bs == 0:
                continue
            start = sample_idx
            end = sample_idx + bs
            cond = build_condition_batch(
                model, cond_kind, sequences.get("main", [])[start:end]
            )
            uncond = None
            if args.cfg_scale > 1.0:
                negative_entries = sequences.get("negative", [])
                if len(negative_entries) < end:
                    raise ValueError(
                        "Not enough negative conditioning entries for requested cfg scale"
                    )
                uncond = build_condition_batch(
                    model, cond_kind, negative_entries[start:end]
                )

            samples, _ = sampler.sample(
                S=args.steps,
                batch_size=bs,
                shape=shape,
                conditioning=cond,
                eta=args.eta,
                quantize_x0=True,
                temperature=args.temperature,
                log_every_t=args.log_every_t,
                unconditional_guidance_scale=args.cfg_scale,
                unconditional_conditioning=uncond,
            )

            decoded = model.decode_first_stage(samples)
            for i in range(decoded.shape[0]):
                pil_img = chw_to_pil(decoded[i])
                filename = os.path.join(args.outdir, f"{sample_idx + i:06}.png")
                pil_img.save(filename)

            sample_idx += bs

    elapsed = time.time() - start_time
    print(f"Saved {sample_idx} samples to {args.outdir} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
