from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs" / "vqvae" / "imagenet-1k"

METHOD_CONFIGS = {
    "vanilla": "vanilla.yaml",
    "simvq": "simvq.yaml",
    "affine": "affine.yaml",
    "mq": "mq.yaml",
    "fvq": "fvq.yaml",
    "asvq": "asvq.yaml",
}


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


def instantiate_quantizer(config_path: Path) -> tuple[Any, dict[str, Any]]:
    quantconfig = parse_quant_config(config_path)
    quantizer_cls = import_obj(quantconfig["target"])
    quantizer = quantizer_cls(**quantconfig["params"])
    return quantizer, quantconfig


def count_module_state(module: Any) -> tuple[int, int, int]:
    param_count = sum(int(t.numel()) for t in module.parameters())
    buffer_count = sum(int(t.numel()) for t in module.buffers())
    return param_count, buffer_count, param_count + buffer_count


def compute_extra_params(config_path: Path) -> dict[str, Any]:
    quantizer, quantconfig = instantiate_quantizer(config_path)
    params = quantconfig["params"]
    codebook_params = int(params["n_e"]) * int(params["e_dim"])
    param_count, buffer_count, total_state = count_module_state(quantizer)
    extra_params = total_state - codebook_params
    if extra_params < 0:
        raise ValueError(
            f"Negative extra params for {config_path}: "
            f"params={param_count}, buffers={buffer_count}, codebook={codebook_params}"
        )
    return {
        "target": quantconfig["target"],
        "codebook_params": codebook_params,
        "param_count": param_count,
        "buffer_count": buffer_count,
        "extra_params": extra_params,
    }


def main() -> None:
    for method, config_name in METHOD_CONFIGS.items():
        info = compute_extra_params(CONFIG_DIR / config_name)
        print(f"{method}: {info['extra_params']}")


if __name__ == "__main__":
    main()
