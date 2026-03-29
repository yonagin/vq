"""
gpt_gen.py
GPT-based image generation script with coordinate conditioning.
This script is designed for batch image inference, referencing the code style
of pixelcnn_gen.py and utilizing model/sampling logic from cond_transformer.py
and mingpt.py.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
import importlib
from taming.models.cond_transformer import Net2NetTransformer
from taming.modules.transformer.mingpt import sample


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "class_path" in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))

def chw_to_pillow(x: torch.Tensor) -> Image.Image:
    """Converts a CHW tensor to a Pillow image."""
    arr = x.detach().cpu().numpy().transpose(1, 2, 0)
    arr = (arr + 1.0) / 2.0
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _infer_latent_size(first_stage_config):
    params = first_stage_config.params
    dd = params.ddconfig
    z_channels = dd.z_channels
    resolution = dd.resolution
    downs = len(dd.ch_mult) - 1
    spatial = resolution // (2**downs)
    if spatial * spatial <= 0:
        raise ValueError("Invalid latent spatial size inferred")
    return z_channels, spatial


def _infer_image_resolution(first_stage_config):
    """Infer the full image resolution from the first stage config."""
    return first_stage_config.params.ddconfig.resolution


def make_coord_batch(batch_size, image_resolution):
    """
    Construct a batch of coordinate conditioning inputs, matching
    the dataset convention used during training (e.g. FFHQTrain with coord=True).

    The dataset builds coords at image-space resolution:
        h, w, _ = image.shape
        coord = np.arange(h * w, dtype=np.float32).reshape(h, w, 1) / float(h * w)

    Returns:
        torch.Tensor of shape (batch_size, 1, H, W) on DEVICE, float32.
    """
    h = w = image_resolution
    coord = np.arange(h * w, dtype=np.float32).reshape(h, w, 1) / float(h * w)
    # Replicate for the batch
    coord_batch = np.tile(coord, (batch_size, 1, 1, 1))  # (B, H, W, 1)
    coord_tensor = torch.from_numpy(coord_batch).to(DEVICE)
    # Permute from (B, H, W, C) to (B, C, H, W)
    coord_tensor = coord_tensor.permute(0, 3, 1, 2).contiguous().float()
    return coord_tensor


def get_parser() -> argparse.ArgumentParser:
    """Creates the argument parser for the script."""
    parser = argparse.ArgumentParser(description="GPT-based Conditional Image Sampler")
    parser.add_argument("--config", required=True, type=str, help="Path to the model config file (.yaml)")
    parser.add_argument("--ckpt", required=True, type=str, help="Path to the model checkpoint file (.ckpt)")
    parser.add_argument("-o", "--outdir", default="gpt_samples", type=str, help="Directory to save the output samples")
    parser.add_argument("-n", "--num_samples", default=64, type=int, help="Total number of samples to generate")
    parser.add_argument("--batch_size", default=8, type=int, help="Number of samples to generate in each batch")
    parser.add_argument("--temperature", default=1.0, type=float, help="Sampling temperature. Higher values increase randomness.")
    parser.add_argument("-k", "--top_k", default=100, type=int, help="Top-k filtering for sampling. Set to 0 to disable.")
    parser.add_argument("-p", "--top_p", default=1.0, type=float, help="Nucleus (top-p) filtering for sampling.")
    return parser


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    if sd is not None:
        model.load_state_dict(sd, strict=False)
    if gpu:
        model = model.to(DEVICE)
    if eval_mode:
        model.eval()
    return {"model": model}


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd.get("global_step", None)
        if global_step:
            print(f"loaded model from global step {global_step}.")
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(
        config.model, pl_sd["state_dict"], gpu=gpu, eval_mode=eval_mode
    )["model"]
    return model, global_step

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    parser = get_parser()

    opt, unknown = parser.parse_known_args()
    ckpt = opt.ckpt
    config = OmegaConf.load(opt.config)

    model, global_step = load_model(config, ckpt, gpu=True, eval_mode=True)

    os.makedirs(opt.outdir, exist_ok=True)

    total = opt.num_samples
    batches = [opt.batch_size] * (total // opt.batch_size)
    if total % opt.batch_size > 0:
        batches.append(total % opt.batch_size)

    print(f"Generating {total} samples and saving to {opt.outdir}")
    sample_idx = 0

    z_channels, latent_hw = _infer_latent_size(
        config.model.init_args.first_stage_config
    )
    image_resolution = _infer_image_resolution(
        config.model.init_args.first_stage_config
    )

    h, w = latent_hw, latent_hw
    num_image_tokens = latent_hw ** 2

    for bs in tqdm(batches, desc="Sampling Batches"):

        # Build coordinate conditioning at image resolution, matching the dataset
        c_input = make_coord_batch(bs, image_resolution)

        _, c_indices = model.encode_to_c(c_input)

        # Sample image tokens autoregressively using the transformer
        indices = sample(
            model=model.transformer,
            x=c_indices,
            steps=num_image_tokens,
            temperature=opt.temperature,
            sample_logits=True,
            top_k=opt.top_k if opt.top_k > 0 else None,
            top_p=opt.top_p,
        )

        # Decode the generated token indices back into an image
        z_shape = (bs, z_channels, h, w)
        generated_images = model.decode_to_img(indices, z_shape)

        # Save the generated images
        for i in range(bs):
            pil_img = chw_to_pillow(generated_images[i])
            pil_img.save(os.path.join(opt.outdir, f"{sample_idx + i:06d}.png"))

        sample_idx += bs