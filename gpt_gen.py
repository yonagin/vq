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

# We assume the project structure allows these imports.
# Make sure 'taming' is in the Python path.
from taming.models.cond_transformer import Net2NetTransformer
from taming.modules.transformer.mingpt import sample


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_model(config_path: str, ckpt_path: str) -> Net2NetTransformer:
    """
    Loads the Net2NetTransformer model from a configuration file and a checkpoint.
    """
    config = OmegaConf.load(config_path)
    # Instantiate the model using parameters from the config file
    # This assumes the config structure matches the model's __init__ signature
    model = Net2NetTransformer(**config.model.params)
    
    # Load the checkpoint
    state = torch.load(ckpt_path, map_location="cpu")
    
    # PyTorch Lightning saves the model's state in the 'state_dict' key
    if "state_dict" in state:
        state = state["state_dict"]
        
    # Load the state dict, ignoring non-matching keys
    model.load_state_dict(state, strict=False)
    
    # Move model to the target device and set to evaluation mode
    model.to(DEVICE)
    model.eval()
    
    # Freeze model parameters for inference
    for param in model.parameters():
        param.requires_grad = False
        
    print(f"Model loaded from {ckpt_path} and moved to {DEVICE}.")
    return model


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    """Main execution function for generating samples."""
    model = load_model(args.config, args.ckpt)
    os.makedirs(args.outdir, exist_ok=True)

    total = args.num_samples
    # Create batches based on total samples and batch size
    batches = [args.batch_size] * (total // args.batch_size)
    if total % args.batch_size > 0:
        batches.append(total % args.batch_size)

    print(f"Generating {total} samples and saving to {args.outdir}")
    sample_idx = 0

    # 1. Construct the conditional input ('coord') based on faceshq.py
    h, w = args.height, args.width
    # Create a normalized coordinate grid for a single sample
    coord_base = np.arange(h * w, dtype=np.float32).reshape(h, w, 1) / float(h * w)
    # Replicate the grid for the entire batch
    coord_batch = np.tile(coord_base, (bs, 1, 1, 1))
    # Convert to a tensor and prepare for the model
    coord_tensor = torch.from_numpy(coord_batch).to(DEVICE)
    # Permute from (B, H, W, C) to (B, C, H, W) as expected by the model
    c_input = coord_tensor.permute(0, 3, 1, 2).contiguous().float()

    # 2. Encode the coordinate input to get conditioning tokens
    _, c_indices = model.encode_to_c(c_input)

    # 3. Define the number of image tokens to generate
    num_image_tokens = args.height * args.width


    for bs in tqdm(batches, desc="Sampling Batches"):
        # 4. Sample image tokens autoregressively using the transformer
        z_indices = sample(
            model=model.transformer,
            x=c_indices,  # The conditioning tokens are the starting sequence
            steps=num_image_tokens,
            temperature=args.temperature,
            sample_logits=True,
            top_k=args.top_k if args.top_k > 0 else None, # Pass None if top_k is 0
            top_p=args.top_p,
        )

        # 5. Decode the generated token indices back into an image
        # Determine the shape of the latent space tensor: (batch, channels, height, width)
        try:
            z_channels = model.first_stage_model.ddconfig.z_channels
        except AttributeError:
            print("Warning: Could not find 'z_channels' in model config. This may cause issues.")
            # Fallback to a default if necessary, though it might be incorrect
            z_channels = 256 # A common value for VQGANs

        z_shape = (bs, z_channels, args.height, args.width)
        generated_images = model.decode_to_img(z_indices, z_shape)

        # 6. Save the generated images to the output directory
        for i in range(bs):
            pil_img = chw_to_pillow(generated_images[i])
            pil_img.save(os.path.join(args.outdir, f"{sample_idx + i:06d}.png"))
        
        sample_idx += bs


def main() -> None:
    """Script entry point."""
    # Add the current working directory to the system path to allow local imports
    sys.path.append(os.getcwd())
    
    parser = get_parser()
    args = parser.parse_args()
    
    start_time = time.time()
    print(f"Starting generation with arguments: {args}")
    
    run(args)
    
    end_time = time.time()
    print(f"Generation finished successfully in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
