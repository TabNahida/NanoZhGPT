#!/usr/bin/env python
import argparse
import glob
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

try:
    from safetensors.torch import save_file
except Exception:
    save_file = None


def ensure_safetensors() -> None:
    if save_file is None:
        raise ImportError("safetensors is required. Install with: pip install safetensors")


def parse_export_dtype(name: str) -> Optional[torch.dtype]:
    key = name.lower()
    if key == "keep":
        return None
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")


def get_input_paths(input_path: Path, pattern: str, include_last: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pt":
            raise ValueError(f"input file must be .pt, got: {input_path}")
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise ValueError(f"input path is not a file/dir: {input_path}")

    paths = [Path(p) for p in glob.glob(str(input_path / pattern))]
    ckpt_re = re.compile(r"ckpt_(\d+)\.pt$")

    def sort_key(p: Path) -> Tuple[int, str]:
        m = ckpt_re.search(p.name)
        if m:
            return int(m.group(1)), p.name
        return 10**18, p.name

    paths = sorted(paths, key=sort_key)
    if include_last:
        last_path = input_path / "ckpt_last.pt"
        if last_path.exists():
            paths.append(last_path)

    deduped: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def normalize_tensor(tensor: torch.Tensor, target_dtype: Optional[torch.dtype]) -> torch.Tensor:
    out = tensor.detach().to("cpu")
    if target_dtype is not None and out.is_floating_point():
        out = out.to(dtype=target_dtype)
    return out.contiguous()


def materialize_state_dict(
    state_dict: Dict[str, torch.Tensor], target_dtype: Optional[torch.dtype]
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    seen_ptrs: Dict[int, str] = {}

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        norm = normalize_tensor(tensor, target_dtype=target_dtype)
        ptr = norm.untyped_storage().data_ptr()
        if ptr in seen_ptrs:
            norm = norm.clone()
        seen_ptrs[ptr] = name
        out[name] = norm
    return out


def build_metadata(ckpt_path: Path, ckpt_obj: Dict[str, object]) -> Tuple[Dict[str, str], Dict[str, object]]:
    iter_raw = ckpt_obj.get("iter_num", -1)
    try:
        iter_num = int(iter_raw)
    except (TypeError, ValueError):
        iter_num = -1

    train_args = ckpt_obj.get("train_args", {})
    if not isinstance(train_args, dict):
        train_args = {}
    train_dtype = str(train_args.get("dtype", "fp32"))
    model_config = ckpt_obj.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError(f"checkpoint missing model_config: {ckpt_path}")

    metadata = {
        "format": "NanoZhGPT-safetensors-v1",
        "source_checkpoint": str(ckpt_path),
        "iter_num": str(iter_num),
        "train_dtype": train_dtype,
        "model_config": json.dumps(model_config, ensure_ascii=True, separators=(",", ":")),
    }
    sidecar = {
        "source_checkpoint": str(ckpt_path),
        "iter_num": iter_num,
        "train_dtype": train_dtype,
        "model_config": model_config,
    }
    return metadata, sidecar


def convert_one(
    ckpt_path: Path,
    out_dir: Path,
    target_dtype: Optional[torch.dtype],
    dtype_name: str,
    overwrite: bool,
) -> Tuple[bool, Path]:
    out_path = out_dir / f"{ckpt_path.stem}.safetensors"
    if out_path.exists() and not overwrite:
        return False, out_path

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        raise ValueError(f"unsupported checkpoint format: {ckpt_path}")
    model_state = ckpt["model"]
    if not isinstance(model_state, dict):
        raise ValueError(f"checkpoint model is not a state_dict: {ckpt_path}")

    tensor_dict = materialize_state_dict(model_state, target_dtype=target_dtype)
    if not tensor_dict:
        raise ValueError(f"no tensor found in checkpoint model state: {ckpt_path}")

    metadata, sidecar = build_metadata(ckpt_path, ckpt)
    save_file(tensor_dict, str(out_path), metadata=metadata)

    sidecar.update(
        {
            "export_dtype": dtype_name,
            "num_tensors": len(tensor_dict),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    sidecar_path = out_path.with_suffix(".meta.json")
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return True, out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert training .pt checkpoints to .safetensors.")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("checkpoints/gpt100m"),
        help="input .pt file or checkpoint directory",
    )
    p.add_argument("--pattern", type=str, default="ckpt_*.pt", help="used when --input is a directory")
    p.add_argument("--include-last", action="store_true", help="include ckpt_last.pt when input is a directory")
    p.add_argument("--out-dir", type=Path, default=None, help="default: same dir as input")
    p.add_argument("--dtype", type=str, default="keep", choices=["keep", "bf16", "fp16", "fp32"])
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_safetensors()
    target_dtype = parse_export_dtype(args.dtype)
    paths = get_input_paths(args.input, pattern=args.pattern, include_last=args.include_last)
    if not paths:
        raise FileNotFoundError(f"no input checkpoints found in {args.input} with pattern={args.pattern!r}")

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.input.is_file():
        out_dir = args.input.parent
    else:
        out_dir = args.input
    out_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    for ckpt_path in paths:
        ok, out_path = convert_one(
            ckpt_path=ckpt_path,
            out_dir=out_dir,
            target_dtype=target_dtype,
            dtype_name=args.dtype,
            overwrite=args.overwrite,
        )
        if ok:
            converted += 1
            print(f"[ok] {ckpt_path.name} -> {out_path.name}")
        else:
            skipped += 1
            print(f"[skip] exists: {out_path}")

    print(
        f"done. input={len(paths)} converted={converted} skipped={skipped} "
        f"out_dir={out_dir} export_dtype={args.dtype}"
    )


if __name__ == "__main__":
    main()
