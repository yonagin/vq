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
    # Move tensor to CPU and convert to numpy array, handling channel-first format
    arr = x.detach().cpu().numpy().transpose(1, 2, 0)
    # Denormalize from [-1, 1] to [0, 1]
    arr = (arr + 1.0) / 2.0
    # Scale to [0, 255] and convert to uint8
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


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
    parser.add_argument("--height", default=16, type=int, help="Height of the latent token grid")
    parser.add_argument("--width", default=16, type=int, help="Width of the latent token grid")
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
    # load the specified checkpoint
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
    # Add the current working directory to the system path to allow local imports
    sys.path.append(os.getcwd())
    parser = get_parser()

    opt, unknown = parser.parse_known_args()
    ckpt = opt.ckpt
    config = OmegaConf.load(opt.config)  # since only one config

    model, global_step = load_model(config, ckpt, gpu=True, eval_mode=True)

    os.makedirs(opt.outdir, exist_ok=True)

    total = opt.num_samples
    # Create batches based on total samples and batch size
    batches = [opt.batch_size] * (total // opt.batch_size)
    if total % opt.batch_size > 0:
        batches.append(total % opt.batch_size)

    print(f"Generating {total} samples and saving to {opt.outdir}")
    sample_idx = 0

    h, w = opt.height, opt.width
    # Create a normalized coordinate grid for a single sample
    coord_base = np.arange(h * w, dtype=np.float32).reshape(h, w, 1) / float(h * w)
    num_image_tokens = opt.height * opt.width


    for bs in tqdm(batches, desc="Sampling Batches"):

        # Replicate the grid for the entire batch
        coord_batch = np.tile(coord_base, (bs, 1, 1, 1))
        # Convert to a tensor and prepare for the model
        coord_tensor = torch.from_numpy(coord_batch).to(DEVICE)
        # Permute from (B, H, W, C) to (B, C, H, W) as expected by the model
        c_input = coord_tensor.permute(0, 3, 1, 2).contiguous().float()
        _, c_indices = model.encode_to_c(c_input)

        # 4. Sample image tokens autoregressively using the transformer
        indices = sample(
            model=model.transformer,
            x=c_indices,  # The conditioning tokens are the starting sequence
            steps=num_image_tokens,
            temperature=opt.temperature,
            sample_logits=True,
            top_k=opt.top_k if opt.top_k > 0 else None, # Pass None if top_k is 0
            top_p=opt.top_p,
        )

        # 5. Decode the generated token indices back into an image
        # Determine the shape of the latent space tensor: (batch, channels, height, width)
        z_channels = model.first_stage_model.ddconfig.z_channels
        z_shape = (bs, z_channels, opt.height, opt.width)
        print(indices.shape)
        generated_images = model.decode_to_img(indices[:, c_indices.shape[1]:], z_shape)

        # 6. Save the generated images to the output directory
        for i in range(bs):
            pil_img = chw_to_pillow(generated_images[i])
            pil_img.save(os.path.join(opt.outdir, f"{sample_idx + i:06d}.png"))
        
        sample_idx += bs