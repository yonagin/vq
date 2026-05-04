import argparse
import csv
import importlib
import math
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(os.getcwd())


METHOD_ORDER = ["vanilla", "simvq", "affine", "mq", "fvq", "asvq"]
METHOD_LABELS = {
    "vanilla": "VQ",
    "simvq": "SimVQ",
    "affine": "AffineVQ",
    "mq": "MQ",
    "fvq": "FVQ",
    "asvq": "ASVQ",
    "asvq65k": "ASVQ-65K",
}
METHOD_COLORS = {
    "vanilla": "#4C78A8",
    "simvq": "#F58518",
    "affine": "#54A24B",
    "mq": "#B279A2",
    "fvq": "#E45756",
    "asvq": "#111111",
    "asvq65k": "#72B7B2",
}
EPS = 1e-8


def get_obj_from_str(string):
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "class_path" in config:
        return get_obj_from_str(config["class_path"])(**config.get("init_args", dict()))
    if "target" in config:
        return get_obj_from_str(config["target"])(**config.get("params", dict()))
    raise KeyError("Expected `class_path` or `target` in config.")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path):
    return OmegaConf.load(str(path))


def checkpoint_sort_key(path):
    name = Path(path).name
    epoch = re.search(r"epoch[=_-](\d+)", name)
    step = re.search(r"step[=_-](\d+)", name)
    if epoch or step:
        return (
            int(epoch.group(1)) if epoch else 10**9,
            int(step.group(1)) if step else 10**9,
            Path(path).stat().st_mtime,
        )
    return (10**9, 10**9, Path(path).stat().st_mtime)


def parse_ckpt_epoch(path, fallback):
    name = Path(path).name
    match = re.search(r"epoch[=_-](\d+)", name)
    if match:
        return int(match.group(1)) + 1
    try:
        ckpt = torch.load(path, map_location="cpu")
        if "epoch" in ckpt:
            return int(ckpt["epoch"]) + 1
    except Exception:
        pass
    return fallback


def default_ckpt_dir(config):
    callbacks = config.trainer.get("callbacks", [])
    for cb in callbacks:
        if cb.get("class_path", "").endswith("ModelCheckpoint"):
            dirpath = cb.get("init_args", {}).get("dirpath")
            if dirpath:
                return Path(str(dirpath))
    return None


def parse_explicit_ckpts(items):
    explicit = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--ckpt expects method=path1,path2, got: {item}")
        method, paths = item.split("=", 1)
        explicit[method.strip()] = [Path(p.strip()) for p in paths.split(",") if p.strip()]
    return explicit


def discover_checkpoints(config, method, run_root=None, explicit=None):
    if explicit and method in explicit:
        paths = explicit[method]
    else:
        ckpt_dir = default_ckpt_dir(config)
        if run_root is not None:
            ckpt_dir = Path(run_root) / method / "ckpt"
        if ckpt_dir is None:
            return []
        paths = list(Path(ckpt_dir).glob("*.ckpt"))
    paths = [p for p in paths if p.exists()]
    return sorted(paths, key=checkpoint_sort_key)


def list_methods(config_dir, requested):
    available = {p.stem for p in Path(config_dir).glob("*.yaml")}
    if requested:
        ordered = []
        for method in requested:
            if method not in available:
                print(f"[warn] config not found for method={method} in {config_dir}")
                continue
            ordered.append(method)
        return ordered
    ordered = [m for m in METHOD_ORDER if m in available]
    extras = sorted(available.difference(ordered))
    return ordered + extras


def select_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def maybe_ema_scope(model, enabled):
    if enabled and getattr(model, "use_ema", False) and hasattr(model, "ema_scope"):
        return model.ema_scope()
    return nullcontext()


def load_model(config, ckpt_path, device):
    model = instantiate_from_config(config.model)
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] {Path(ckpt_path).name}: missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] {Path(ckpt_path).name}: unexpected keys: {len(unexpected)}")
    model.eval().to(device)
    return model


def codebook_raw(quant):
    if hasattr(quant, "embedding"):
        return quant.embedding.weight.detach()
    if hasattr(quant, "embed"):
        return quant.embed.weight.detach()
    return None


@torch.no_grad()
def effective_codebook(quant):
    if hasattr(quant, "get_quant_codebook"):
        try:
            return quant.get_quant_codebook().detach()
        except TypeError:
            return quant.get_quant_codebook(None).detach()
    if hasattr(quant, "get_norm_cb") and hasattr(quant, "scale"):
        return (quant.scale.detach().view(1, -1) * quant.get_norm_cb().detach()).detach()
    raw = codebook_raw(quant)
    return raw.detach() if raw is not None else None


def analytic_scale_and_geometry(codebook):
    scale = codebook.std(dim=0, unbiased=False).clamp_min(EPS)
    geometry = codebook / scale.unsqueeze(0)
    return scale, geometry


def midpoint_decomposition(prev_codebook, cur_codebook):
    prev_scale, prev_geom = analytic_scale_and_geometry(prev_codebook)
    cur_scale, cur_geom = analytic_scale_and_geometry(cur_codebook)
    mean_scale = 0.5 * (prev_scale + cur_scale)
    mean_geom = 0.5 * (prev_geom + cur_geom)
    scale_term = (cur_scale - prev_scale).unsqueeze(0) * mean_geom.unsqueeze(0)
    geom_term = mean_scale.unsqueeze(0) * (cur_geom - prev_geom)
    delta = cur_codebook - prev_codebook
    scale_energy = float(scale_term.pow(2).sum().item())
    geom_energy = float(geom_term.pow(2).sum().item())
    total_energy = float(delta.pow(2).sum().item())
    return {
        "scale_energy": scale_energy,
        "geom_energy": geom_energy,
        "total_energy": total_energy,
        "residual_energy": float((delta - scale_term - geom_term).pow(2).sum().item()),
        "scale_norm": float(scale_term.norm().item()),
        "geom_norm": float(geom_term.norm().item()),
    }


def maybe_native_scale_stats(quant):
    if hasattr(quant, "scale"):
        scale = quant.scale.detach().float().cpu().view(-1)
        return {
            "native_scale_mean": float(scale.mean().item()),
            "native_scale_cv": float(scale.std(unbiased=False).item() / (scale.mean().item() + EPS)),
        }
    return {
        "native_scale_mean": math.nan,
        "native_scale_cv": math.nan,
    }


def analyze_codebook_trajectory(method, config, checkpoints, device):
    rows = []
    prev_codebook = None
    cumulative_scale = 0.0
    cumulative_geom = 0.0
    cumulative_total = 0.0

    for idx, ckpt_path in enumerate(checkpoints):
        print(f"[traj] {method}: loading {ckpt_path.name}")
        model = load_model(config, ckpt_path, device)
        quant = model.quantize
        codebook = effective_codebook(quant).float().cpu()
        cb_scale, cb_geom = analytic_scale_and_geometry(codebook)

        row = {
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "checkpoint": str(ckpt_path),
            "checkpoint_name": ckpt_path.name,
            "epoch": parse_ckpt_epoch(ckpt_path, idx + 1),
            "codebook_size": int(codebook.shape[0]),
            "embedding_dim": int(codebook.shape[1]),
            "codebook_scale_mean": float(cb_scale.mean().item()),
            "codebook_scale_cv": float(cb_scale.std(unbiased=False).item() / (cb_scale.mean().item() + EPS)),
            "geometry_rms": float(torch.sqrt((cb_geom.pow(2)).mean()).item()),
        }
        row.update(maybe_native_scale_stats(quant))

        if prev_codebook is None:
            row.update(
                {
                    "scale_energy": 0.0,
                    "geom_energy": 0.0,
                    "total_energy": 0.0,
                    "residual_energy": 0.0,
                    "scale_share": math.nan,
                    "geom_share": math.nan,
                }
            )
        else:
            stats = midpoint_decomposition(prev_codebook, codebook)
            cumulative_scale += stats["scale_energy"]
            cumulative_geom += stats["geom_energy"]
            cumulative_total += stats["total_energy"]
            denom = stats["scale_energy"] + stats["geom_energy"] + EPS
            row.update(
                {
                    "scale_energy": stats["scale_energy"],
                    "geom_energy": stats["geom_energy"],
                    "total_energy": stats["total_energy"],
                    "residual_energy": stats["residual_energy"],
                    "scale_share": stats["scale_energy"] / denom,
                    "geom_share": stats["geom_energy"] / denom,
                }
            )

        rows.append(row)
        prev_codebook = codebook
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    denom = cumulative_scale + cumulative_geom + EPS
    summary = {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "num_checkpoints": len(checkpoints),
        "scale_energy_sum": cumulative_scale,
        "geom_energy_sum": cumulative_geom,
        "total_energy_sum": cumulative_total,
        "scale_share": cumulative_scale / denom,
        "geom_share": cumulative_geom / denom,
    }
    return rows, summary


def build_datamodule(config, batch_size=None, num_workers=None):
    data_cfg = OmegaConf.create(OmegaConf.to_container(config.data, resolve=True))
    if batch_size is not None:
        data_cfg["init_args"]["batch_size"] = batch_size
    if num_workers is not None:
        data_cfg["init_args"]["num_workers"] = num_workers
    datamodule = instantiate_from_config(data_cfg)
    datamodule.prepare_data()
    datamodule.setup()
    return datamodule


def get_eval_dataloader(datamodule):
    if hasattr(datamodule, "val_dataloader"):
        return datamodule.val_dataloader()
    if hasattr(datamodule, "_val_dataloader"):
        return datamodule._val_dataloader()
    raise AttributeError("Could not find validation dataloader on the instantiated data module.")


def subsample_pairs(latents, quants, max_points, seed):
    if max_points is None or latents.shape[0] <= max_points:
        return latents, quants
    rng = np.random.default_rng(seed)
    keep = rng.choice(latents.shape[0], size=max_points, replace=False)
    keep.sort()
    return latents[keep], quants[keep]


@torch.no_grad()
def collect_latent_pairs(model, dataloader, device, max_batches, max_points, seed, use_ema_eval):
    latents = []
    quants = []
    total_tokens = 0

    with maybe_ema_scope(model, use_ema_eval):
        iterator = tqdm(dataloader, total=max_batches, dynamic_ncols=True, desc="val")
        for batch_idx, batch in enumerate(iterator):
            if max_batches is not None and batch_idx >= max_batches:
                break
            images = model.get_input(batch, model.image_key).to(device, non_blocking=True)
            hidden = model.encoder(images)
            quantized, _, indices, _ = model.encode(images)

            z_flat = hidden.permute(0, 2, 3, 1).reshape(-1, hidden.shape[1]).detach().cpu()
            q_flat = quantized.permute(0, 2, 3, 1).reshape(-1, quantized.shape[1]).detach().cpu()
            latents.append(z_flat)
            quants.append(q_flat)

            total_tokens += int(indices.reshape(-1).numel())

    latents = torch.cat(latents, dim=0).numpy().astype(np.float64, copy=False)
    quants = torch.cat(quants, dim=0).numpy().astype(np.float64, copy=False)
    latents, quants = subsample_pairs(latents, quants, max_points=max_points, seed=seed)
    return latents, quants, total_tokens


def fit_pca_basis(array, rank):
    centered = array - array.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh[:rank].T
    return array.mean(axis=0, keepdims=True), basis


def js_divergence(p, q):
    p = p.astype(np.float64, copy=False)
    q = q.astype(np.float64, copy=False)
    p = p / (p.sum() + EPS)
    q = q / (q.sum() + EPS)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * (np.log(p + EPS) - np.log(m + EPS)))
    kl_qm = np.sum(q * (np.log(q + EPS) - np.log(m + EPS)))
    return float(0.5 * (kl_pm + kl_qm) / math.log(2.0))


def pca_histogram_js(latents, quants, bins):
    mean, basis = fit_pca_basis(latents, rank=2)
    z2 = (latents - mean) @ basis
    q2 = (quants - mean) @ basis
    lo = np.percentile(z2, 1.0, axis=0)
    hi = np.percentile(z2, 99.0, axis=0)
    pad = 0.05 * np.maximum(hi - lo, EPS)
    hist_range = [
        [float(lo[0] - pad[0]), float(hi[0] + pad[0])],
        [float(lo[1] - pad[1]), float(hi[1] + pad[1])],
    ]
    z_hist, _, _ = np.histogram2d(z2[:, 0], z2[:, 1], bins=bins, range=hist_range)
    q_hist, _, _ = np.histogram2d(q2[:, 0], q2[:, 1], bins=bins, range=hist_range)
    return js_divergence(z_hist.reshape(-1), q_hist.reshape(-1))


def analyze_manifold_match(
    method,
    config,
    checkpoint,
    device,
    dataloader,
    max_batches,
    max_points,
    bins,
    pca_rank,
    seed,
    use_ema_eval,
):
    print(f"[mani] {method}: loading {checkpoint.name}")
    model = load_model(config, checkpoint, device)
    latents, quants, total_tokens = collect_latent_pairs(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_batches,
        max_points=max_points,
        seed=seed,
        use_ema_eval=use_ema_eval,
    )

    metric_row = {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "num_tokens_total": int(total_tokens),
        "num_tokens_sampled": int(latents.shape[0]),
        "occupancy_js_divergence": pca_histogram_js(latents, quants, bins=bins),
    }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metric_row


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def configure_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )
    return plt


def plot_scale_decomposition(summary_rows, output_dir):
    if not summary_rows:
        return
    plt = configure_matplotlib()
    methods = [row["method"] for row in summary_rows]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    scale_share = [row["scale_share"] for row in summary_rows]
    geom_share = [row["geom_share"] for row in summary_rows]
    colors = [METHOD_COLORS.get(m, "#666666") for m in methods]
    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.bar(x, scale_share, color=colors, edgecolor="white", linewidth=0.8, label="Scale term")
    ax.bar(
        x,
        geom_share,
        bottom=scale_share,
        color="white",
        edgecolor=colors,
        linewidth=1.0,
        hatch="///",
        label="Normalized geometry term",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Cumulative update energy share")
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.18))

    for xpos, share in zip(x, scale_share):
        ax.text(xpos, min(share + 0.03, 0.97), f"{share:.0%}", ha="center", va="bottom", fontsize=6)

    for ext in ("pdf", "png"):
        fig.savefig(Path(output_dir) / f"fig_scale_vs_geometry.{ext}", dpi=300)
    plt.close(fig)


def plot_manifold_metrics(metric_rows, output_dir, pca_rank):
    if not metric_rows:
        return
    plt = configure_matplotlib()
    methods = [row["method"] for row in metric_rows]
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [METHOD_COLORS.get(m, "#666666") for m in methods]
    y = np.arange(len(metric_rows))

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    values = [row["occupancy_js_divergence"] for row in metric_rows]
    ax.hlines(y, xmin=np.zeros_like(y, dtype=float), xmax=values, color="#D9D9D9", linewidth=1.2, zorder=1)
    ax.scatter(values, y, s=34, c=colors, edgecolors="black", linewidths=0.35, zorder=2)
    ax.set_xlabel("Occupancy JS divergence ↓")
    ax.set_axisbelow(True)
    ax.grid(axis="x")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    for ext in ("pdf", "png"):
        fig.savefig(Path(output_dir) / f"fig_manifold_match.{ext}", dpi=300)
    plt.close(fig)


def write_summary(path, scale_rows, manifold_rows):
    lines = []
    if scale_rows:
        lines.append("[Scale vs. geometry]")
        ranked = sorted(scale_rows, key=lambda row: row["scale_share"], reverse=True)
        for row in ranked:
            lines.append(
                f"{row['label']}: scale_share={row['scale_share']:.4f}, "
                f"geometry_share={row['geom_share']:.4f}, "
                f"num_checkpoints={row['num_checkpoints']}"
            )
        lines.append("")
    if manifold_rows:
        lines.append("[Manifold matching]")
        for row in manifold_rows:
            lines.append(
                f"{row['label']}: "
                f"js={row['occupancy_js_divergence']:.4f}"
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Empirical analysis for scale/geometry decomposition and manifold matching."
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs/vqvae/imagenet-1k"),
        help="Directory containing method yaml files.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Subset of methods to analyze. Default: all yaml files in config-dir.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Root directory that contains <method>/ckpt/*.ckpt.",
    )
    parser.add_argument(
        "--ckpt",
        action="append",
        default=None,
        help="Explicit checkpoints: method=path1,path2,... . Overrides auto-discovery for that method.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/asvq_imagenet1k"),
        help="Where CSVs and figures are written.",
    )
    parser.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--batch-size", type=int, default=64, help="Validation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Validation dataloader workers.")
    parser.add_argument("--max-batches", type=int, default=32, help="Validation batches for manifold analysis.")
    parser.add_argument(
        "--max-points",
        type=int,
        default=200000,
        help="Maximum number of latent/quantized token pairs kept per method.",
    )
    parser.add_argument("--hist-bins", type=int, default=48, help="Bins per axis for the PCA occupancy histogram.")
    parser.add_argument("--pca-rank", type=int, default=8, help="Rank for the subspace-affinity metric.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for subsampling.")
    parser.add_argument(
        "--no-ema-eval",
        action="store_true",
        help="Disable EMA weights during final manifold evaluation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ensure_dir(args.output_dir)

    methods = list_methods(args.config_dir, args.methods)
    explicit_ckpts = parse_explicit_ckpts(args.ckpt)
    device = select_device(args.device)

    trajectory_rows = []
    trajectory_summary = []
    manifold_rows = []

    method_to_config = {}
    method_to_ckpts = {}
    for method in methods:
        config_path = Path(args.config_dir) / f"{method}.yaml"
        if not config_path.exists():
            continue
        config = load_config(config_path)
        checkpoints = discover_checkpoints(config, method, run_root=args.run_root, explicit=explicit_ckpts)
        if not checkpoints:
            print(f"[warn] no checkpoints found for {method}")
            continue
        method_to_config[method] = config
        method_to_ckpts[method] = checkpoints

    if not method_to_ckpts:
        raise FileNotFoundError(
            "No checkpoints were discovered. Pass --run-root or explicit --ckpt method=path1,path2,..."
        )

    shared_datamodule = build_datamodule(
        next(iter(method_to_config.values())),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    shared_dataloader = get_eval_dataloader(shared_datamodule)

    for method in methods:
        if method not in method_to_ckpts:
            continue
        rows, summary = analyze_codebook_trajectory(
            method=method,
            config=method_to_config[method],
            checkpoints=method_to_ckpts[method],
            device=device,
        )
        trajectory_rows.extend(rows)
        trajectory_summary.append(summary)

    for method in methods:
        if method not in method_to_ckpts:
            continue
        manifold_rows.append(
            analyze_manifold_match(
                method=method,
                config=method_to_config[method],
                checkpoint=method_to_ckpts[method][-1],
                device=device,
                dataloader=shared_dataloader,
                max_batches=args.max_batches,
                max_points=args.max_points,
                bins=args.hist_bins,
                pca_rank=args.pca_rank,
                seed=args.seed,
                use_ema_eval=not args.no_ema_eval,
            )
        )

    trajectory_summary = [row for row in trajectory_summary if row["method"] in methods]
    manifold_rows = [row for row in manifold_rows if row["method"] in methods]

    write_csv(Path(args.output_dir) / "trajectory_checkpoints.csv", trajectory_rows)
    write_csv(Path(args.output_dir) / "trajectory_summary.csv", trajectory_summary)
    write_csv(Path(args.output_dir) / "manifold_metrics.csv", manifold_rows)
    write_summary(Path(args.output_dir) / "summary.txt", trajectory_summary, manifold_rows)

    plot_scale_decomposition(trajectory_summary, args.output_dir)
    plot_manifold_metrics(manifold_rows, args.output_dir, pca_rank=args.pca_rank)

    print(f"Saved trajectory CSV: {Path(args.output_dir) / 'trajectory_summary.csv'}")
    print(f"Saved manifold CSV:   {Path(args.output_dir) / 'manifold_metrics.csv'}")
    print(f"Saved figures to:     {args.output_dir}")


if __name__ == "__main__":
    main()