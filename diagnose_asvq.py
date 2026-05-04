import argparse
import csv
import importlib
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
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
    m = re.search(r"epoch[=_-](\d+)", name)
    if m:
        return int(m.group(1)) + 1
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
        explicit[method.strip()] = paths
    return explicit


def discover_checkpoints(config, method, run_root=None, explicit=None):
    if explicit and method in explicit:
        paths = [Path(p) for p in explicit[method].split(",")]
    else:
        ckpt_dir = default_ckpt_dir(config)
        if run_root is not None:
            ckpt_dir = Path(run_root) / method / "ckpt"
        if ckpt_dir is None:
            return []
        paths = list(Path(ckpt_dir).glob("*.ckpt"))
    paths = [p for p in paths if p.exists()]
    return sorted(paths, key=checkpoint_sort_key)


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


@torch.no_grad()
def decompose_codebook(quant):
    """Return effective codebook, channel scale, and scale-normalized geometry.

    For ASVQ, the decomposition uses its explicit s and normalized codebook.
    For other quantizers, it uses the analytical channel std decomposition of
    the effective codebook, which makes the scale-vs-geometry comparison fair.
    """
    if hasattr(quant, "get_norm_cb") and hasattr(quant, "scale"):
        scale = quant.scale.detach().float().flatten().clamp_min(EPS)
        norm_cb = quant.get_norm_cb().detach().float()
        eff_cb = scale.view(1, -1) * norm_cb
        return eff_cb, scale, norm_cb

    eff_cb = effective_codebook(quant).detach().float()
    scale = eff_cb.std(dim=0, unbiased=False).clamp_min(EPS)
    norm_cb = eff_cb / scale.view(1, -1)
    return eff_cb, scale, norm_cb


def scale_match_error(feature_std, cb_std):
    return float((torch.log(feature_std.clamp_min(EPS)) - torch.log(cb_std.clamp_min(EPS))).abs().mean().item())


def scale_corr(feature_std, cb_std):
    x = feature_std.detach().cpu().numpy()
    y = cb_std.detach().cpu().numpy()
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def js_divergence(p, q):
    p = p / p.sum().clamp_min(EPS)
    q = q / q.sum().clamp_min(EPS)
    m = 0.5 * (p + q)
    return float(0.5 * (p * (p.clamp_min(EPS) / m.clamp_min(EPS)).log()).sum().item() +
                 0.5 * (q * (q.clamp_min(EPS) / m.clamp_min(EPS)).log()).sum().item())


def load_validation_loader(config, batch_size, num_workers):
    data_config = OmegaConf.create(OmegaConf.to_container(config.data, resolve=True))
    data_config.init_args.batch_size = batch_size
    data_config.init_args.num_workers = num_workers
    dataset = instantiate_from_config(data_config)
    dataset.prepare_data()
    dataset.setup()
    return dataset._val_dataloader()


@torch.no_grad()
def collect_features(model, loader, device, max_batches, max_points):
    sum_z = None
    sumsq_z = None
    count = 0
    samples = []
    kept = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = model.get_input(batch, model.image_key).to(device)
        h = model.encoder(x)
        z = rearrange(h, "b c h w -> (b h w) c").float()
        if sum_z is None:
            sum_z = torch.zeros(z.shape[1], device=device)
            sumsq_z = torch.zeros(z.shape[1], device=device)
        sum_z += z.sum(dim=0)
        sumsq_z += (z * z).sum(dim=0)
        count += z.shape[0]

        if kept < max_points:
            take = min(max_points - kept, z.shape[0])
            samples.append(z[:take].detach().cpu())
            kept += take

    mean = sum_z / max(count, 1)
    var = (sumsq_z / max(count, 1) - mean * mean).clamp_min(0)
    std = var.sqrt().detach().cpu()
    sample_tensor = torch.cat(samples, dim=0) if samples else torch.empty(0)
    return {
        "mean": mean.detach().cpu(),
        "std": std,
        "samples": sample_tensor,
        "count": count,
    }


@torch.no_grad()
def argmin_assignments(z, codebook, chunk_size):
    z = z.float()
    codebook = codebook.float()
    cb_t = codebook.t().contiguous()
    cb_norm = (codebook * codebook).sum(dim=1)
    out = []
    for start in range(0, z.shape[0], chunk_size):
        part = z[start:start + chunk_size]
        dist = (part * part).sum(dim=1, keepdim=True) + cb_norm.view(1, -1) - 2 * part @ cb_t
        out.append(dist.argmin(dim=1).cpu())
    return torch.cat(out, dim=0)


def contribution_metrics(prev, cur):
    prev_eff, prev_scale, prev_norm = prev
    cur_eff, cur_scale, cur_norm = cur
    if prev_eff.shape != cur_eff.shape:
        return {}

    denom = prev_eff.norm().clamp_min(EPS)
    scale_only = (cur_scale.view(1, -1) * prev_norm - prev_eff).norm() / denom
    norm_only = (prev_scale.view(1, -1) * cur_norm - prev_eff).norm() / denom
    total = (cur_eff - prev_eff).norm() / denom
    scale_share = scale_only / (scale_only + norm_only).clamp_min(EPS)
    norm_cos_drift = 1.0 - F.cosine_similarity(prev_norm.flatten(), cur_norm.flatten(), dim=0)
    scale_log_drift = (torch.log(cur_scale) - torch.log(prev_scale)).abs().mean()
    return {
        "effective_rel_drift": float(total.item()),
        "scale_only_rel_drift": float(scale_only.item()),
        "norm_only_rel_drift": float(norm_only.item()),
        "scale_share": float(scale_share.item()),
        "norm_share": float((1 - scale_share).item()),
        "norm_cosine_drift": float(norm_cos_drift.item()),
        "scale_log_l1_drift": float(scale_log_drift.item()),
    }


def finite_or_none(x):
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def manifold_metrics(z_samples, feature_std, eff_cb):
    if z_samples.numel() == 0:
        return {}, None
    z = z_samples.float()
    z_mean = z.mean(dim=0, keepdim=True)
    zc = z - z_mean
    _, _, vh = torch.linalg.svd(zc, full_matrices=False)
    basis = vh[: min(16, vh.shape[0])].t().contiguous()
    z_proj = zc @ basis
    cb_proj = (eff_cb.detach().cpu().float() - z_mean) @ basis
    z_var = z_proj.var(dim=0, unbiased=False).clamp_min(EPS)
    cb_var = cb_proj.var(dim=0, unbiased=False).clamp_min(EPS)
    z_dist = z_var / z_var.sum()
    cb_dist = cb_var / cb_var.sum()
    cumulative = (cb_var.cumsum(0) / cb_var.sum()).numpy()
    metrics = {
        "pca_var_js": js_divergence(z_dist, cb_dist),
        "pca_var_l1": float((z_dist - cb_dist).abs().mean().item()),
        "pca_var_cos": float(F.cosine_similarity(z_dist, cb_dist, dim=0).item()),
        "feature_cb_std_ratio_mean": float((eff_cb.detach().cpu().float().std(dim=0, unbiased=False).clamp_min(EPS) /
                                            feature_std.clamp_min(EPS)).mean().item()),
    }
    return metrics, cumulative


def assignment_without_scale_metrics(method, z_samples, eff_cb, scale, norm_cb, chunk_size):
    if method not in ("asvq", "asvq65k") or z_samples.numel() == 0:
        return {}
    z = z_samples.cpu().float()
    full_idx = argmin_assignments(z, eff_cb.cpu(), chunk_size)
    no_scale_idx = argmin_assignments(z, norm_cb.detach().cpu().float(), chunk_size)
    scalar_scale = scale.mean().view(1, 1).cpu() * norm_cb.detach().cpu().float()
    scalar_idx = argmin_assignments(z, scalar_scale, chunk_size)

    full_util = full_idx.unique().numel() / eff_cb.shape[0]
    no_scale_util = no_scale_idx.unique().numel() / eff_cb.shape[0]
    scalar_util = scalar_idx.unique().numel() / eff_cb.shape[0]
    return {
        "remove_s_flip_rate": float((full_idx != no_scale_idx).float().mean().item()),
        "scalar_s_flip_rate": float((full_idx != scalar_idx).float().mean().item()),
        "remove_s_util_delta": float(no_scale_util - full_util),
        "scalar_s_util_delta": float(scalar_util - full_util),
    }


def smooth_xy(x, y, points=160):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return x, y
    order = np.argsort(x)
    x, y = x[order], y[order]
    dense_x = np.linspace(x.min(), x.max(), points)
    dense_y = np.interp(dense_x, x, y)
    if len(x) >= 4:
        radius = max(2, points // 40)
        grid = np.arange(-radius, radius + 1)
        kernel = np.exp(-(grid * grid) / (2 * (radius / 2) ** 2))
        kernel = kernel / kernel.sum()
        pad = np.pad(dense_y, (radius, radius), mode="edge")
        dense_y = np.convolve(pad, kernel, mode="valid")
    return dense_x, dense_y


def filter_valid_series(rows, metric):
    xs, ys = [], []
    for row in rows:
        val = finite_or_none(row.get(metric))
        if val is None:
            continue
        xs.append(row["epoch"])
        ys.append(val)
    return xs, ys


def plot_metric(ax, rows_by_method, metric, ylabel, title, ylim=None):
    for method, rows in rows_by_method.items():
        x, y = filter_valid_series(rows, metric)
        if not x:
            continue
        sx, sy = smooth_xy(x, y)
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, None)
        ax.plot(sx, sy, color=color, lw=2.0, label=label)
        ax.scatter(x, y, color=color, s=16, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, color="#E6E6E6", linewidth=0.8)


def save_question_plots(rows_by_method, pca_curves, out_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "diagnose_asvq.py needs matplotlib for NeurIPS-style figures. "
            "Install it with `pip install matplotlib`, then rerun the script."
        ) from exc

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.4), constrained_layout=True)
    plot_metric(ax, rows_by_method, "scale_share", "Scale-dominant share", "Q1: scale vs. norm change", (0, 1))
    ax.axhline(0.5, color="#999999", lw=1.0, ls="--")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, frameon=False, ncol=2, loc="best")
    for ext in ("pdf", "png"):
        fig.savefig(Path(out_dir) / f"q1_scale_vs_norm.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5), constrained_layout=True)
    plot_metric(axes[0], rows_by_method, "scale_match_log_l1", "Log-scale gap", "Q2a: feature/codebook scale match")
    axes[1].set_title("Q2b: final variance on data PCs")
    axes[1].set_xlabel("PC rank")
    axes[1].set_ylabel("Cumulative variance")
    axes[1].set_ylim(0, 1.02)
    axes[1].grid(True, color="#E6E6E6", linewidth=0.8)
    for method, curve in pca_curves.items():
        if curve is None:
            continue
        x = np.arange(1, len(curve) + 1)
        axes[1].plot(x, curve, lw=2.0, color=METHOD_COLORS.get(method), label=METHOD_LABELS.get(method, method))
    handles, labels = axes[1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.03))
    for ext in ("pdf", "png"):
        fig.savefig(Path(out_dir) / f"q2_manifold_match.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    asvq_rows = rows_by_method.get("asvq", [])
    if asvq_rows:
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.4), constrained_layout=True)
        x1, y1 = filter_valid_series(asvq_rows, "remove_s_flip_rate")
        x2, y2 = filter_valid_series(asvq_rows, "scalar_s_flip_rate")
        if x1:
            sx, sy = smooth_xy(x1, y1)
            ax.plot(sx, sy, color=METHOD_COLORS["asvq"], lw=2.0, label="remove s")
            ax.scatter(x1, y1, color=METHOD_COLORS["asvq"], s=16, zorder=3)
        if x2:
            sx, sy = smooth_xy(x2, y2)
            ax.plot(sx, sy, color="#888888", lw=2.0, ls="--", label="replace by mean(s)")
            ax.scatter(x2, y2, color="#888888", s=16, zorder=3)
        ax.set_title("Q3: assignment change after removing s")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Flip rate")
        ax.set_ylim(0, 1)
        ax.grid(True, color="#E6E6E6", linewidth=0.8)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, frameon=False, loc="best")
        for ext in ("pdf", "png"):
            fig.savefig(Path(out_dir) / f"q3_remove_s_assignment.{ext}", dpi=300, bbox_inches="tight")
        plt.close(fig)


def write_csv(rows, out_path):
    keys = sorted({k for row in rows for k in row.keys()})
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_latex(rows_by_method, out_path):
    final_rows = []
    for method, rows in rows_by_method.items():
        if rows:
            final_rows.append(rows[-1])
    cols = [
        ("method", "Method"),
        ("scale_share", "Scale Share"),
        ("scale_match_log_l1", "Scale Gap"),
        ("pca_var_js", "PCA JS"),
        ("remove_s_flip_rate", "w/o $s$ Flip"),
    ]
    lines = ["\\begin{tabular}{lcccc}", "\\toprule"]
    lines.append(" & ".join(name for _, name in cols) + " \\\\")
    lines.append("\\midrule")
    for row in final_rows:
        vals = [METHOD_LABELS.get(row["method"], row["method"])]
        for key, _ in cols[1:]:
            val = row.get(key, float("nan"))
            vals.append("--" if not np.isfinite(val) else f"{val:.3f}")
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def write_summary(rows_by_method, out_path):
    lines = [
        "# ASVQ empirical summary",
        "",
        "Only three questions are reported here.",
        "Q1 uses `scale_share`: values above 0.5 mean effective-codebook change is more scale-driven than normalized-codebook-driven.",
        "Q2 uses two views: `scale_match_log_l1` for channel-scale matching, and final cumulative variance on data PCs for manifold alignment.",
        "Q3 uses assignment flip rate after removing `s` from ASVQ.",
        "",
        "## Final epoch summary",
        "",
    ]
    for method, rows in rows_by_method.items():
        if not rows:
            continue
        row = rows[-1]
        lines.append(
            f"- {METHOD_LABELS.get(method, method)}: "
            f"scale_share={row.get('scale_share', float('nan')):.3f} | "
            f"scale_gap={row.get('scale_match_log_l1', float('nan')):.3f} | "
            f"pca_js={row.get('pca_var_js', float('nan')):.3f}"
            + (
                f" | remove_s_flip={row.get('remove_s_flip_rate', float('nan')):.3f}"
                if finite_or_none(row.get("remove_s_flip_rate")) is not None
                else ""
            )
        )
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")


def run(args):
    ensure_dir(args.output_dir)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    explicit = parse_explicit_ckpts(args.ckpt)
    methods = args.methods
    all_rows = []
    rows_by_method = {}
    pca_curves = {}

    for method in methods:
        cfg_path = Path(args.config_dir) / f"{method}.yaml"
        if not cfg_path.exists():
            print(f"[skip] missing config: {cfg_path}")
            continue
        config = load_config(cfg_path)
        ckpts = discover_checkpoints(config, method, args.run_root, explicit)
        if args.latest_only and ckpts:
            ckpts = [ckpts[-1]]
        if not ckpts:
            print(f"[skip] no checkpoints for {method}")
            continue

        print(f"[method] {method}: {len(ckpts)} checkpoint(s)")
        rows_by_method[method] = []
        prev_components = None
        loader = None
        final_pca_curve = None

        for i, ckpt_path in enumerate(tqdm(ckpts, desc=method)):
            epoch = parse_ckpt_epoch(ckpt_path, i + 1)
            model = load_model(config, ckpt_path, device)
            eff_cb, scale, norm_cb = decompose_codebook(model.quantize)
            eff_cb_cpu = eff_cb.detach().cpu().float()
            scale_cpu = scale.detach().cpu().float()
            norm_cpu = norm_cb.detach().cpu().float()

            if loader is None:
                loader = load_validation_loader(config, args.batch_size, args.num_workers)
            feats = collect_features(model, loader, device, args.max_batches, args.max_points)
            feature_std = feats["std"].float()
            cb_std = eff_cb_cpu.std(dim=0, unbiased=False).clamp_min(EPS)

            row = {
                "method": method,
                "label": METHOD_LABELS.get(method, method),
                "epoch": epoch,
                "checkpoint": str(ckpt_path),
                "feature_tokens": feats["count"],
                "codebook_size": eff_cb_cpu.shape[0],
                "dim": eff_cb_cpu.shape[1],
                "scale_mean": float(scale_cpu.mean().item()),
                "scale_cv": float((scale_cpu.std(unbiased=False) / scale_cpu.mean().clamp_min(EPS)).item()),
                "cb_std_mean": float(cb_std.mean().item()),
                "feature_std_mean": float(feature_std.mean().item()),
                "scale_match_log_l1": scale_match_error(feature_std, cb_std),
                "scale_match_corr": scale_corr(feature_std, cb_std),
            }

            cur_components = (eff_cb_cpu, scale_cpu, norm_cpu)
            if prev_components is not None:
                row.update(contribution_metrics(prev_components, cur_components))
            else:
                row.update({
                    "effective_rel_drift": float("nan"),
                    "scale_only_rel_drift": float("nan"),
                    "norm_only_rel_drift": float("nan"),
                    "scale_share": float("nan"),
                    "norm_share": float("nan"),
                    "norm_cosine_drift": float("nan"),
                    "scale_log_l1_drift": float("nan"),
                })

            manifold, pca_curve = manifold_metrics(feats["samples"], feature_std, eff_cb_cpu)
            row.update(manifold)
            row.update(assignment_without_scale_metrics(method, feats["samples"], eff_cb_cpu, scale_cpu, norm_cpu, args.assign_chunk_size))
            final_pca_curve = pca_curve

            rows_by_method[method].append(row)
            all_rows.append(row)
            prev_components = cur_components
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        pca_curves[method] = final_pca_curve

    all_rows_for_csv = []
    for row in all_rows:
        cleaned = {}
        for key, value in row.items():
            if isinstance(value, float) and not np.isfinite(value):
                cleaned[key] = ""
            else:
                cleaned[key] = value
        all_rows_for_csv.append(cleaned)

    write_csv(all_rows_for_csv, Path(args.output_dir) / "metrics.csv")
    write_latex(rows_by_method, Path(args.output_dir) / "final_table.tex")
    write_summary(rows_by_method, Path(args.output_dir) / "summary.md")
    save_question_plots(rows_by_method, pca_curves, args.output_dir)
    print(f"[done] wrote results to {args.output_dir}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Empirical diagnostics for ASVQ scale-vs-normalized-codebook claims."
    )
    parser.add_argument("--config-dir", default="configs/vqvae/imagenet-1k")
    parser.add_argument("--methods", nargs="+", default=METHOD_ORDER)
    parser.add_argument("--run-root", default=None, help="Optional root with method/ckpt/*.ckpt layout.")
    parser.add_argument("--ckpt", action="append", default=[], help="Explicit method=path1,path2 override. Can repeat.")
    parser.add_argument("--output-dir", default="analysis/asvq_empirical")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=32)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--assign-chunk-size", type=int, default=2048)
    parser.add_argument("--latest-only", action="store_true", help="Analyze only the last checkpoint of each method.")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
