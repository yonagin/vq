"""
We provide Tokenizer Evaluation code here.
Refer to 
https://github.com/richzhang/PerceptualSimilarity
https://github.com/mseitzer/pytorch-fid
"""

import os
import sys
sys.path.append(os.getcwd())

import torch
from omegaconf import OmegaConf
import importlib
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
from scipy import linalg
import argparse
import torchvision.utils as vutils
import lpips
import pyiqa
import piq
from cleanfid import fid


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


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


def load_vqgan_new(config, ckpt_path=None, is_gumbel=False):
    model = instantiate_from_config(config.model)
    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        model.load_state_dict(sd, strict=False)
    return model.eval()

def get_args():
    parser = argparse.ArgumentParser(description="inference parameters")
    parser.add_argument("--config_file", required=True, type=str)
    parser.add_argument("--ckpt_path", required=True, type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--psnr_y", action='store_true', help="Use Y channel for PSNR calculation")
    return parser.parse_args()


def main(args):
    config_data = OmegaConf.load(args.config_file)
    config_data.data.init_args.batch_size = args.batch_size

    config_model = load_config(args.config_file, display=False)
    model = load_vqgan_new(config_model, ckpt_path=args.ckpt_path).to(DEVICE)

    codebook_size = model.quantize.n_e
    usage = torch.zeros(codebook_size, dtype=torch.long)

    # LPIPS
    loss_fn_alex = lpips.LPIPS(net="alex").to(DEVICE).eval()
    loss_fn_vgg = lpips.LPIPS(net="vgg").to(DEVICE).eval()
    lpips_alex_sum = 0.0
    lpips_vgg_sum = 0.0

    # PSNR/SSIM running sums 
    psnr_sum = 0.0
    ssim_sum = 0.0
    num_images = 0
    psnr_computer = pyiqa.create_metric('psnr', test_y_channel=args.psnr_y, color_space='rgb', device=DEVICE)

    dataset = instantiate_from_config(config_data.data)
    dataset.prepare_data()
    dataset.setup()
    dataloader = dataset._val_dataloader()

    recons_save_dir = Path(args.config_file).parent / "recons"
    source_save_dir = Path(args.config_file).parent / "source"
    os.makedirs(recons_save_dir, exist_ok=True)
    os.makedirs(source_save_dir, exist_ok=True)

    total = len(dataloader) if hasattr(dataloader, "__len__") else None
    pbar = tqdm(total=total, dynamic_ncols=True)

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].permute(0, 3, 1, 2).to(DEVICE, non_blocking=True)

            if model.use_ema:
                with model.ema_scope():
                    quant, diff, indices, _ = model.encode(images)
                    reconstructed_images = model.decode(quant)
            else:
                quant, diff, indices, _ = model.encode(images)
                reconstructed_images = model.decode(quant)

            reconstructed_images = reconstructed_images.clamp(-1, 1)

            # usage (faster than dict)
            idx_cpu = indices.flatten().detach().to("cpu", non_blocking=True)
            usage += torch.bincount(idx_cpu, minlength=codebook_size)

            # LPIPS (expects [-1,1])
            lpips_alex_sum += loss_fn_alex(images, reconstructed_images).sum().item()
            lpips_vgg_sum += loss_fn_vgg(images, reconstructed_images).sum().item()

            # to [0,1]
            images_01 = (images + 1) / 2
            rec_01 = (reconstructed_images + 1) / 2

            # PSNR & SSIM using pyiqa and piq (per-image)
            B = images_01.shape[0]
            for i in range(B):
                # Convert to format expected by pyiqa and piq
                img_i = images_01[i:i+1]
                rec_i = rec_01[i:i+1]
                
                # PSNR using pyiqa
                psnr_score = psnr_computer(img_i, rec_i)
                psnr_sum += psnr_score.sum().item()
                
                # SSIM using piq
                ssim_score = piq.ssim(img_i, rec_i, data_range=1., reduction='none')
                ssim_sum += ssim_score.sum().item()

            # save images (optional but kept)
            for b in range(B):
                vutils.save_image(
                    images_01[b],
                    os.path.join(source_save_dir, f"{num_images + b}.png"),
                    normalize=False,
                    nrow=1,
                )
                vutils.save_image(
                    rec_01[b],
                    os.path.join(recons_save_dir, f"{num_images + b}.png"),
                    normalize=False,
                    nrow=1,
                )

            num_images += B

            # update progress bar postfix
            psnr_avg = psnr_sum / max(num_images, 1)
            ssim_avg = ssim_sum / max(num_images, 1)
            pbar.set_postfix(
                {
                    "PSNR": f"{psnr_avg:.3f}",
                    "SSIM": f"{ssim_avg:.4f}",
                    "imgs": num_images,
                }
            )
            pbar.update(1)

    pbar.close()
    
    lpips_alex_value = lpips_alex_sum / max(num_images, 1)
    lpips_vgg_value = lpips_vgg_sum / max(num_images, 1)
    ssim_value = ssim_sum / max(num_images, 1)
    psnr_value = psnr_sum / max(num_images, 1)
    fid_value = fid.compute_fid(str(recons_save_dir), str(source_save_dir), mode="clean")
    utilization = (usage > 0).float().mean().item()

    if usage.sum() > 0:
        probs = usage.float() / usage.sum()
        # Only compute log for non-zero probabilities to avoid -inf
        entropy = -torch.sum(probs[probs > 0] * torch.log(probs[probs > 0]))
        ppl_value = torch.exp(entropy).item()
    else:
        ppl_value = 0.0

    def print_and_save(message, file):
        print(message)
        file.write(message + "\n")

    out_path = Path(args.ckpt_path).parent / "result.txt"
    with open(out_path, "w") as f:
        print_and_save(f"FID: {fid_value}", f)
        print_and_save(f"LPIPS_ALEX: {lpips_alex_value}", f)
        print_and_save(f"LPIPS_VGG: {lpips_vgg_value}", f)
        print_and_save(f"SSIM: {ssim_value}", f)
        print_and_save(f"PSNR: {psnr_value}", f)
        print_and_save(f"PPL: {ppl_value}", f)
        print_and_save(f"utilization: {utilization}", f)
        print_and_save(f"num_images: {num_images}", f)

    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    args = get_args()
    main(args)