from __future__ import annotations

import importlib
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


def import_obj(class_path: str) -> Any:
    module_name, obj_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


def instantiate_quantizer(config_path: Path) -> Any:
    quantconfig = parse_quant_config(config_path)
    quantizer_cls = import_obj(quantconfig["target"])
    return quantizer_cls(**quantconfig["params"])


def count_module_state(module: Any) -> int:
    total = 0
    for tensor in module.parameters():
        total += int(tensor.numel())
    for tensor in module.buffers():
        total += int(tensor.numel())
    return total


def compute_extra_params(config_path: Path) -> int:
    quantconfig = parse_quant_config(config_path)
    quantizer = instantiate_quantizer(config_path)
    n_e = int(quantconfig["params"]["n_e"])
    e_dim = int(quantconfig["params"]["e_dim"])
    codebook_params = n_e * e_dim
    total_state = count_module_state(quantizer)
    extra_params = total_state - codebook_params
    if extra_params < 0:
        raise ValueError(
            f"Negative extra params for {config_path}: "
            f"total_state={total_state}, codebook={codebook_params}"
        )
    return extra_params


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


def main() -> None:
    rows = build_rows()
    RESULT_PATH.write_text(update_result_text(rows), encoding="utf-8")
    OUTPUT_PATH.write_text(to_latex(rows), encoding="utf-8")
    print(f"Updated {RESULT_PATH}")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
