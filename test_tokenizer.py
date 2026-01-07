"""
Tokenizer Test Code for GPT Training
Test the first_stage (VQGAN) reconstruction capability before GPT training
"""

import os
import sys
sys.path.append(os.getcwd())

import torch
from omegaconf import OmegaConf
import importlib
from pathlib import Path
import argparse
import torchvision.utils as vutils
from tqdm import tqdm
import copy
import gc


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "class_path" not in config:
        raise KeyError("Expected key `class_path` to instantiate.")
    return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))


def load_model_from_config(config, sd, gpu=True, eval_mode=True):
    model = instantiate_from_config(config)
    if sd is not None:
        model.load_state_dict(sd, strict=False)
    if gpu:
        model = model.to(DEVICE)
    if eval_mode:
        model.eval()
    return model


def get_args():
    parser = argparse.ArgumentParser(description="Tokenizer test before GPT training")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint file (optional, if not provided GPT will be randomly initialized)")
    parser.add_argument("--num_samples", type=int, default=32, help="number of samples to test")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size for testing")
    parser.add_argument("--save_dir", type=str, default="./tokenizer_test", help="directory to save test results")
    return parser.parse_args()


def main(args):
    # Load config
    print(f"Loading config from: {args.config}")
    config = OmegaConf.load(args.config)

    # Load checkpoint if provided
    pl_sd = None
    if args.ckpt is not None:
        print(f"Loading checkpoint from: {args.ckpt}")
        pl_sd = torch.load(args.ckpt, map_location="cpu")
        global_step = pl_sd.get("global_step", None)
        if global_step:
            print(f"Loaded model from global step {global_step}")
    else:
        print("No checkpoint provided, GPT will be randomly initialized")

    # Load full transformer model on CPU first
    print("Loading full transformer model...")
    model = load_model_from_config(
        config.model, 
        pl_sd["state_dict"] if pl_sd is not None else None, 
        gpu=False,      
        eval_mode=True
    )

    # Extract first_stage only, delete the rest
    print("Extracting first_stage (VQGAN) from model...")
    first_stage = copy.deepcopy(model.first_stage_model)
    del model
    gc.collect()   
    torch.cuda.empty_cache() 

    first_stage = first_stage.to(DEVICE)
    first_stage = first_stage.half()
    first_stage.eval()
    
    # Load validation dataset from config
    print("Loading validation dataset...")
    dataset = instantiate_from_config(config.data)
    dataset.prepare_data()
    dataset.setup()
    dataloader = dataset._val_dataloader()

    # Create save directories
    save_dir = Path(args.save_dir)
    original_dir = save_dir / "original"
    reconstructed_dir = save_dir / "reconstructed"
    original_dir.mkdir(parents=True, exist_ok=True)
    reconstructed_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting tokenizer test on {args.num_samples} samples...")
    print(f"Saving results to: {save_dir}\n")

    num_tested = 0
    # Calculate total batches needed based on num_samples
    total_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    with torch.no_grad(), tqdm(total=total_batches, desc="Testing") as pbar:
        for batch in dataloader:
            if num_tested >= args.num_samples:
                break

            # Get images from batch (handle different data formats)
            if "image" in batch:
                images = batch["image"].permute(0, 3, 1, 2).to(DEVICE).half()
            elif "jpg" in batch:
                images = batch["jpg"].to(DEVICE).half()
            else:
                raise KeyError(f"Unknown image key in batch: {batch.keys()}")

            # Encode and decode using first_stage
            quant, diff, indices, _ = first_stage.encode(images)
            reconstructed = first_stage.decode(quant)

            # Clamp to valid range
            reconstructed = reconstructed.clamp(-1, 1)

            # Save images
            batch_size = min(images.shape[0], args.num_samples - num_tested)
            for i in range(batch_size):
                # Convert from [-1, 1] to [0, 1] for saving
                img_original = (images[i] + 1) / 2
                img_reconstructed = (reconstructed[i] + 1) / 2

                vutils.save_image(
                    img_original,
                    original_dir / f"{num_tested + i:06d}.png",
                    normalize=False
                )
                vutils.save_image(
                    img_reconstructed,
                    reconstructed_dir / f"{num_tested + i:06d}.png",
                    normalize=False
                )

            num_tested += batch_size
            pbar.update(1)    

    print(f"\n{'='*60}")
    print(f"Tokenizer test completed!")
    print(f"\n{'='*60}")



if __name__ == "__main__":
    args = get_args()
    main(args)