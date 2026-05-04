from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
RESULT_PATH = ROOT / "vqvae_result.txt"
OUTPUT_PATH = ROOT / "vqvae_result.tex"
CONFIG_DIR = ROOT / "configs" / "vqvae" / "imagenet-1k"

METHOD_CONFIGS = {
    "VanillaVQ": "vanilla.yaml",
    "SimVQ": "simvq.yaml",
    "AffineVQ": "affine.yaml",
    "MQ": "mq.yaml",
    "FVQ": "fvq.yaml",
    "ASVQ": "asvq.yaml",
}


@dataclass
class ResultRow:
    name: str
    transform_type: str
    extra_params: int
    psnr: float
    ssim: float
    ppl: float
    utilization: float


def parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?", value, flags=re.IGNORECASE):
        return float(value)
    return value


def parse_quant_config(path: Path) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_quantconfig = False
    in_params = False
    target = None
    params: dict[str, Any] = {}

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if stripped == "quantconfig:":
            in_quantconfig = True
            in_params = False
            continue

        if in_quantconfig and indent <= 4 and not stripped.startswith(("target:", "params:")):
            break

        if not in_quantconfig:
            continue

        if indent == 6 and stripped.startswith("target:"):
            target = stripped.split(":", 1)[1].strip()
            continue

        if indent == 6 and stripped == "params:":
            in_params = True
            continue

        if in_params:
            if indent <= 6:
                break
            if ":" not in stripped:
                continue
            key, value = stripped.split(":", 1)
            params[key.strip()] = parse_scalar(value)

    if target is None:
        raise ValueError(f"Could not parse quantconfig target from {path}")
    return {"target": target, "params": params}


def linear_count(in_features: int, out_features: int, bias: bool = True) -> int:
    return in_features * out_features + (out_features if bias else 0)


def embedding_count(num_embeddings: int, embedding_dim: int) -> int:
    return num_embeddings * embedding_dim


def conv2d_count(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = True,
) -> int:
    return out_channels * in_channels * kernel_size * kernel_size + (out_channels if bias else 0)


def qbridge_linear_extra(e_dim: int, layers: int) -> int:
    return layers * linear_count(e_dim, e_dim, bias=True)


def dit_extra(e_dim: int, n_e: int, bridge_model_name: str) -> int:
    dit_specs = {
        "QBridge-XS/2": {"depth": 2, "patch_size": 2, "head_hidden_size": 8, "num_heads": 4},
        "QBridge-S/2": {"depth": 2, "patch_size": 2, "head_hidden_size": 16, "num_heads": 4},
        "QBridge-S/4": {"depth": 2, "patch_size": 4, "head_hidden_size": 16, "num_heads": 4},
        "Qbridge-S/4-d4": {"depth": 4, "patch_size": 4, "head_hidden_size": 16, "num_heads": 4},
        "QBridge-S/8": {"depth": 2, "patch_size": 8, "head_hidden_size": 16, "num_heads": 4},
        "QBridge-B/2": {"depth": 2, "patch_size": 2, "head_hidden_size": 32, "num_heads": 4},
        "QBridge-B/8": {"depth": 2, "patch_size": 8, "head_hidden_size": 32, "num_heads": 4},
        "QBridge-B/4": {"depth": 2, "patch_size": 4, "head_hidden_size": 32, "num_heads": 4},
        "QBridge-B/4-d1": {"depth": 1, "patch_size": 4, "head_hidden_size": 32, "num_heads": 4},
        "QBridge-B/4-d4": {"depth": 4, "patch_size": 4, "head_hidden_size": 32, "num_heads": 4},
        "QBridge-L/2-d4": {"depth": 4, "patch_size": 2, "head_hidden_size": 64, "num_heads": 4},
        "QBridge-L/8": {"depth": 2, "patch_size": 8, "head_hidden_size": 64, "num_heads": 4},
        "QBridge-L/4": {"depth": 2, "patch_size": 4, "head_hidden_size": 64, "num_heads": 4},
        "QBridge-L/2": {"depth": 2, "patch_size": 2, "head_hidden_size": 64, "num_heads": 4},
        "QBridge-L/4-d1": {"depth": 1, "patch_size": 4, "head_hidden_size": 64, "num_heads": 4},
        "QBridge-L/4-d4": {"depth": 4, "patch_size": 4, "head_hidden_size": 64, "num_heads": 4},
        "QBridge-XL/4": {"depth": 2, "patch_size": 4, "head_hidden_size": 128, "num_heads": 4},
        "QBridge-XL/4-d4": {"depth": 4, "patch_size": 4, "head_hidden_size": 128, "num_heads": 4},
    }
    spec = dit_specs[bridge_model_name]
    input_size = math.isqrt(n_e)
    if input_size * input_size != n_e:
        raise ValueError(f"{bridge_model_name} requires square codebook size, got n_e={n_e}")

    depth = spec["depth"]
    patch_size = spec["patch_size"]
    hidden_size = spec["head_hidden_size"] * spec["num_heads"]
    mlp_hidden_dim = hidden_size * 2
    num_patches = (input_size // patch_size) ** 2
    num_classes = 1001  # LabelEmbedder(num_classes=1000, dropout_prob=0.1)

    patch_embed = conv2d_count(e_dim, hidden_size, patch_size, bias=True)
    label_embed = embedding_count(num_classes, hidden_size)
    pos_embed = num_patches * hidden_size

    attention = linear_count(hidden_size, 3 * hidden_size, True) + linear_count(hidden_size, hidden_size, True)
    mlp = linear_count(hidden_size, mlp_hidden_dim, True) + linear_count(mlp_hidden_dim, hidden_size, True)
    ada_ln = linear_count(hidden_size, 6 * hidden_size, True)
    dit_block = attention + mlp + ada_ln
    blocks = depth * dit_block

    final_linear = linear_count(hidden_size, patch_size * patch_size * e_dim, True)
    final_ada_ln = linear_count(hidden_size, 2 * hidden_size, True)
    final_layer = final_linear + final_ada_ln

    return patch_embed + label_embed + pos_embed + blocks + final_layer


def compute_extra_params(config_path: Path) -> int:
    quantconfig = parse_quant_config(config_path)
    target = quantconfig["target"]
    params = quantconfig["params"]
    n_e = int(params["n_e"])
    e_dim = int(params["e_dim"])

    if target.endswith("VectorQuantizer"):
        return 0

    if target.endswith("SimVQ"):
        return qbridge_linear_extra(e_dim=e_dim, layers=1)

    if target.endswith("AffineVQ"):
        use_running_statistics = bool(params.get("use_running_statistics", False))
        num_groups = int(params.get("num_groups", 1))
        if use_running_statistics:
            return 4 * num_groups * e_dim + 1
        return 2 * num_groups * e_dim

    if target.endswith("ASVQ"):
        use_ema_scale = bool(params.get("use_ema_scale", True))
        if use_ema_scale:
            return e_dim + 1
        return e_dim

    if target.endswith("BridgeVQ"):
        bridge_type = params.get("bridge_type", "linear")
        bridge_model_name = params.get("bridge_model_name")
        bridge_num_layers = int(params.get("bridge_num_layers", 5))

        if bridge_model_name is None:
            bridge_type = str(bridge_type).lower()
            if bridge_type in ("identity", "none"):
                bridge_model_name = "Qbridge-none"
            elif bridge_type in ("linear", "lin", "single_linear"):
                bridge_model_name = "Qbridge-lin/1"
            elif bridge_type == "mlp":
                if bridge_num_layers == 1:
                    bridge_model_name = "Qbridge-lin/1"
                elif bridge_num_layers == 5:
                    bridge_model_name = "Qbridge-lin/5"
                else:
                    raise ValueError(f"Unsupported MLP depth: {bridge_num_layers}")
            elif bridge_type == "dit":
                bridge_model_name = "QBridge-B/4"
            else:
                raise ValueError(f"Unsupported bridge_type: {bridge_type}")

        if bridge_model_name == "Qbridge-none":
            return 0
        if bridge_model_name == "Qbridge-lin/1":
            return qbridge_linear_extra(e_dim=e_dim, layers=1)
        if bridge_model_name == "Qbridge-lin/5":
            return qbridge_linear_extra(e_dim=e_dim, layers=5)
        return dit_extra(e_dim=e_dim, n_e=n_e, bridge_model_name=bridge_model_name)

    raise ValueError(f"Unsupported quantizer target: {target}")


def parse_results(path: Path) -> dict[str, dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    blocks = [block.strip() for block in text.strip().split("\n\n") if block.strip()]
    parsed: dict[str, dict[str, str]] = {}
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        name = lines[0].rstrip(":")
        values: dict[str, str] = {}
        for line in lines[1:]:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
        parsed[name] = values
    return parsed


def format_param_count(value: int) -> str:
    return f"{value:,}"


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def to_latex(rows: list[ResultRow]) -> str:
    lines = [
        r"\begin{tabular}{l l r r r r r}",
        r"\toprule",
        r"Method & Transform type & Extra params & PSNR & SSIM & PPL & Utilization \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.name} & {row.transform_type} & {format_param_count(row.extra_params)} "
            f"& {format_float(row.psnr)} & {format_float(row.ssim)} "
            f"& {format_float(row.ppl)} & {format_float(row.utilization)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def build_rows() -> list[ResultRow]:
    raw_results = parse_results(RESULT_PATH)
    rows: list[ResultRow] = []
    for method_name, config_name in METHOD_CONFIGS.items():
        entry = raw_results[method_name]
        rows.append(
            ResultRow(
                name=method_name,
                transform_type=entry["Transform type"],
                extra_params=compute_extra_params(CONFIG_DIR / config_name),
                psnr=float(entry["PSNR"]),
                ssim=float(entry["SSIM"]),
                ppl=float(entry["PPL"]),
                utilization=float(entry["utilization"]),
            )
        )
    return rows


def update_result_text(rows: list[ResultRow]) -> str:
    original = RESULT_PATH.read_text(encoding="utf-8")
    updated = original
    for row in rows:
        if row.name == "VanillaVQ":
            continue
        pattern = rf"({re.escape(row.name)}:\s+Transform type:\s+.*?\s+Extra params:)\?"
        updated, count = re.subn(pattern, rf"\g<1>{row.extra_params}", updated, count=1)
        if count != 1:
            raise ValueError(f"Failed to update Extra params for {row.name}")
    return updated


def main() -> None:
    rows = build_rows()
    RESULT_PATH.write_text(update_result_text(rows), encoding="utf-8")
    OUTPUT_PATH.write_text(to_latex(rows), encoding="utf-8")
    print(f"Updated {RESULT_PATH}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
