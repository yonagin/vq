import argparse
import os
import sys
import time
import shutil
import importlib
import numpy as np
import torch

from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from cleanfid import fid

# -----------------------------------------------------------------------------
# Device choose (CUDA/NPU/CPU)
# -----------------------------------------------------------------------------
try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

def pick_device(device_str: str | None):
    """
    device_str:
      - None / "auto": prefer cuda, then npu, then cpu
      - "cuda:0", "cuda", "cpu", "npu:0" ...
    """
    if device_str is None or device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        # NOTE: cleanfid 基于 PyTorch + torchvision 的常规 CUDA 路径；NPU 未必能跑通
        if hasattr(torch, "npu") and torch.npu.is_available():
            return torch.device("npu:0")
        return torch.device("cpu")
    return torch.device(device_str)

# Default device for BOTH generation and FID unless overridden
DEFAULT_DEVICE = pick_device("auto")

rescale = lambda x: (x + 1.) / 2.

# -----------------------------------------------------------------------------
# 1) Model loading helpers
# -----------------------------------------------------------------------------
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

def load_model_from_config(config, sd, device, eval_mode=True):
    model = instantiate_from_config(config)
    if sd is not None:
        model.load_state_dict(sd, strict=False)
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model

def load_model(config, ckpt, device):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        sd = pl_sd.get("state_dict", pl_sd)
    else:
        sd = None
    model = load_model_from_config(config.model, sd, device=device, eval_mode=True)
    return model

def chw_to_pillow(x):
    return Image.fromarray((255 * rescale(x.detach().cpu().numpy().transpose(1, 2, 0)))
                           .clip(0, 255).astype(np.uint8))

# -----------------------------------------------------------------------------
# 2) Sample generation
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_samples(
    model,
    output_dir,
    num_samples,
    batch_size,
    device,
    temperature=1.0,
    top_k=250,
    top_p=1.0,
    cfg_scale=1.0,
    token_factorization=False,
    dim_z=None,
):
    if os.path.exists(output_dir):
        print(f"Warning: Output directory {output_dir} exists. Cleaning it for new samples...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 你的本地依赖
    from taming.modules.transformer.llama import sample

    batches = [batch_size for _ in range(num_samples // batch_size)]
    if num_samples % batch_size > 0:
        batches.append(num_samples % batch_size)

    print(f"Start generating {num_samples} samples to {output_dir}...")
    cnt = 0
    t_start = time.time()

    for _, bs in tqdm(enumerate(batches), desc="Generating", total=len(batches)):
        if bs == 0:
            break

        qzshape = [bs, dim_z, 16, 16]

        index_sample = sample(
            None,
            model=model.transformer,
            steps=256,
            sample_logits=True,
            top_k=top_k,
            callback=None,
            temperature=temperature,
            top_p=top_p,
            token_factorization=token_factorization,
            cfg_scale=cfg_scale,
            num_samples=bs,
        )

        x_sample = model.decode_to_img(index_sample, qzshape)

        for i in range(x_sample.shape[0]):
            img = chw_to_pillow(x_sample[i])
            img.save(os.path.join(output_dir, f"{cnt:06}.png"))
            cnt += 1

    t_end = time.time()
    print(f"Generation finished. Time elapsed: {t_end - t_start:.2f}s")

# -----------------------------------------------------------------------------
# 3) FID evaluation (two paths: official vs custom)
# -----------------------------------------------------------------------------
def ensure_custom_stats(
    custom_name: str,
    real_dir: str,
    mode: str,
    device: torch.device,
    num_workers: int,
    batch_size: int,
    force_recompute: bool,
    real_num: int | None,
    model_name: str,
    verbose: bool,
):
    if not os.path.exists(real_dir):
        raise FileNotFoundError(f"Real image directory not found: {real_dir}")

    exists = fid.test_stats_exists(custom_name, mode=mode, model_name=model_name)
    if exists and not force_recompute:
        print(f"[custom stats] Cache exists: name='{custom_name}', mode='{mode}', model='{model_name}'. Skip recompute.")
        return

    if exists and force_recompute:
        print(f"[custom stats] Cache exists but --force_recompute_real_stats enabled. Removing old stats first...")
        fid.remove_custom_stats(custom_name, mode=mode, model_name=model_name)

    print(f"[custom stats] Computing and caching stats:")
    print(f"  name       = {custom_name}")
    print(f"  real_dir   = {real_dir}")
    print(f"  mode       = {mode}")
    print(f"  model_name = {model_name}")
    print(f"  device     = {device}")
    print(f"  num_workers= {num_workers}")
    print(f"  batch_size = {batch_size}")
    print(f"  real_num   = {real_num if real_num is not None else 'ALL'}")

    fid.make_custom_stats(
        name=custom_name,
        fdir=real_dir,
        num=real_num,
        mode=mode,
        model_name=model_name,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        verbose=verbose,
    )

def compute_fid_official(
    generated_dir: str,
    dataset_name: str,
    dataset_res: int,
    dataset_split: str,
    mode: str,
    device: torch.device,
    num_workers: int,
    batch_size: int,
    model_name: str,
    use_dataparallel: bool,
    verbose: bool,
):
    print("[FID official] Using precomputed official stats:")
    print(f"  dataset_name = {dataset_name}")
    print(f"  dataset_res  = {dataset_res}")
    print(f"  dataset_split= {dataset_split}")
    print(f"  mode         = {mode}")
    print(f"  model_name   = {model_name}")
    print(f"  device       = {device}")
    print(f"  num_workers  = {num_workers}")
    print(f"  batch_size   = {batch_size}")
    return fid.compute_fid(
        fdir1=generated_dir,
        mode=mode,
        model_name=model_name,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        dataset_name=dataset_name,
        dataset_res=dataset_res,
        dataset_split=dataset_split,
        verbose=verbose,
        use_dataparallel=use_dataparallel,
    )

def compute_fid_custom(
    generated_dir: str,
    custom_name: str,
    mode: str,
    device: torch.device,
    num_workers: int,
    batch_size: int,
    model_name: str,
    use_dataparallel: bool,
    verbose: bool,
):
    print("[FID custom] Using cached custom stats:")
    print(f"  custom_name  = {custom_name}")
    print(f"  dataset_split= custom")
    print(f"  mode         = {mode}")
    print(f"  model_name   = {model_name}")
    print(f"  device       = {device}")
    print(f"  num_workers  = {num_workers}")
    print(f"  batch_size   = {batch_size}")
    return fid.compute_fid(
        fdir1=generated_dir,
        mode=mode,
        model_name=model_name,
        num_workers=num_workers,
        batch_size=batch_size,
        device=device,
        dataset_name=custom_name,
        dataset_split="custom",
        verbose=verbose,
        use_dataparallel=use_dataparallel,
    )

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser()

    # ----------------------------
    # Model + generation
    # ----------------------------
    parser.add_argument("--ckpt", type=str, required=False, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=False, help="Path to model config (yaml)")

    parser.add_argument("-n", "--num_samples", type=int, default=5000, help="Number of images to generate")
    parser.add_argument("--gen_batch_size", type=int, default=25, help="Batch size for generation")
    parser.add_argument("-t", "--temperature", type=float, default=1.0)
    parser.add_argument("-k", "--top_k", type=int, default=None)
    parser.add_argument("-p", "--top_p", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=1.0)
    parser.add_argument("--token_factorization", action="store_true")

    parser.add_argument("--gen_input", type=str, default=None,
                        help="Existing generated images folder. If provided, skip model load + generation.")
    parser.add_argument("-o", "--gen_outdir", type=str, default="gen_samples",
                        help="Folder to save generated images")

    # ----------------------------
    # FID options
    # ----------------------------
    parser.add_argument("--stats_source", type=str, choices=["official", "custom"], required=True,
                        help="Use official precomputed stats OR custom stats from --real_folder")

    parser.add_argument("--mode", type=str, default="clean",
                        choices=["clean", "legacy_pytorch", "legacy_tensorflow"],
                        help="cleanfid mode")

    parser.add_argument("--fid_device", type=str, default="auto",
                        help="Device for FID feature extractor: auto/cuda:0/cpu ... (default auto)")

    parser.add_argument("--fid_num_workers", type=int, default=None,
                        help="Dataloader num_workers for FID (default: min(16, cpu_count))")
    parser.add_argument("--fid_batch_size", type=int, default=256,
                        help="Batch size for FID feature extraction")

    parser.add_argument("--model_name", type=str, default="inception_v3",
                        choices=["inception_v3", "clip_vit_b_32"],
                        help="Feature extractor backend in cleanfid")

    parser.add_argument("--use_dataparallel", action="store_true",
                        help="Enable cleanfid DataParallel for feature extractor (multi-GPU)")

    parser.add_argument("--fid_verbose", action="store_true", help="Verbose tqdm/progress in cleanfid")

    # ----------------------------
    # official stats args
    # ----------------------------
    parser.add_argument("--official_dataset_name", type=str, default="ffhq",
                        help="Official dataset name (e.g., ffhq, cifar10, afhq_cat...)")
    parser.add_argument("--official_dataset_res", type=int, default=256,
                        help="Official dataset resolution (e.g., 256)")
    parser.add_argument("--official_dataset_split", type=str, default="trainval70k",
                        help="Official dataset split (e.g., trainval70k)")

    # ----------------------------
    # custom stats args
    # ----------------------------
    parser.add_argument("--real_folder", type=str, default=None,
                        help="Path to real images folder (required for stats_source=custom unless cached exists)")
    parser.add_argument("--custom_stats_name", type=str, default=None,
                        help="Cache name for custom stats (required for stats_source=custom)")
    parser.add_argument("--real_num", type=int, default=None,
                        help="If set, only use first N real images to build stats (debug/fast check)")
    parser.add_argument("--force_recompute_real_stats", action="store_true",
                        help="If set, remove existing cached custom stats first then recompute")

    return parser

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    opt = get_parser().parse_args()

    # FID speed defaults
    fid_device = pick_device(opt.fid_device)
    fid_num_workers = opt.fid_num_workers
    if fid_num_workers is None:
        fid_num_workers = min(16, (os.cpu_count() or 8))

    # 1) Prepare generated images
    if opt.gen_input is not None:
        generated_dir = opt.gen_input
        if not os.path.exists(generated_dir):
            raise FileNotFoundError(f"Generated images directory not found: {generated_dir}")
        print(f"Using existing generated images from: {generated_dir}")
        print("Skipping model loading and generation...")
    else:
        if opt.ckpt is None or opt.config is None:
            raise ValueError("--ckpt and --config are required when --gen_input is not provided")

        print("Loading config and model...")
        config = OmegaConf.load(opt.config)
        model = load_model(config, opt.ckpt, device=DEFAULT_DEVICE)

        # 你原本的 dim_z 获取逻辑
        dim_z = config.model.init_args.first_stage_config.params.quantconfig.params.e_dim

        generate_samples(
            model=model,
            output_dir=opt.gen_outdir,
            num_samples=opt.num_samples,
            batch_size=opt.gen_batch_size,
            device=DEFAULT_DEVICE,
            temperature=opt.temperature,
            top_k=opt.top_k,
            top_p=opt.top_p,
            cfg_scale=opt.cfg_scale,
            token_factorization=opt.token_factorization,
            dim_z=dim_z,
        )
        generated_dir = opt.gen_outdir

    # 2) Compute FID
    try:
        print("Starting FID evaluation...")

        if opt.stats_source == "official":
            fid_score = compute_fid_official(
                generated_dir=generated_dir,
                dataset_name=opt.official_dataset_name,
                dataset_res=opt.official_dataset_res,
                dataset_split=opt.official_dataset_split,
                mode=opt.mode,
                device=fid_device,
                num_workers=fid_num_workers,
                batch_size=opt.fid_batch_size,
                model_name=opt.model_name,
                use_dataparallel=opt.use_dataparallel,
                verbose=opt.fid_verbose,
            )

        elif opt.stats_source == "custom":
            if opt.custom_stats_name is None:
                raise ValueError("--custom_stats_name is required when --stats_source=custom")

            # 如果缓存不存在，必须提供 real_folder 来算；如果缓存存在，可不提供 real_folder
            exists = fid.test_stats_exists(opt.custom_stats_name, mode=opt.mode, model_name=opt.model_name)
            if (not exists) and (opt.real_folder is None):
                raise ValueError(
                    "Custom stats not found in cache, and --real_folder not provided. "
                    "Provide --real_folder for the first run."
                )

            if opt.real_folder is not None:
                ensure_custom_stats(
                    custom_name=opt.custom_stats_name,
                    real_dir=opt.real_folder,
                    mode=opt.mode,
                    device=fid_device,
                    num_workers=fid_num_workers,
                    batch_size=opt.fid_batch_size,
                    force_recompute=opt.force_recompute_real_stats,
                    real_num=opt.real_num,
                    model_name=opt.model_name,
                    verbose=opt.fid_verbose,
                )
            else:
                print("[custom stats] --real_folder not provided; assuming cached stats exist.")

            fid_score = compute_fid_custom(
                generated_dir=generated_dir,
                custom_name=opt.custom_stats_name,
                mode=opt.mode,
                device=fid_device,
                num_workers=fid_num_workers,
                batch_size=opt.fid_batch_size,
                model_name=opt.model_name,
                use_dataparallel=opt.use_dataparallel,
                verbose=opt.fid_verbose,
            )

        else:
            raise ValueError(f"Unknown stats_source: {opt.stats_source}")

        print("-" * 60)
        print(f"FID score: {fid_score:.6f}")
        print("-" * 60)

        # 3) Save result log
        if opt.ckpt:
            model_name = os.path.splitext(os.path.basename(opt.ckpt))[0]
            result_path = os.path.join(os.path.dirname(opt.ckpt), f"{model_name}_fid_results.txt")
            tag = model_name
        else:
            result_path = "fid_results.txt"
            tag = "existing_images"

        with open(result_path, "a", encoding="utf-8") as f:
            f.write(f"Time: {time.asctime()}\n")
            f.write(f"Tag: {tag}\n")
            f.write(f"Args: {vars(opt)}\n")
            f.write(f"FID: {fid_score:.6f}\n\n")

        print(f"Results appended to {result_path}")

    except Exception as e:
        print(f"Error during FID calculation: {e}")
        raise