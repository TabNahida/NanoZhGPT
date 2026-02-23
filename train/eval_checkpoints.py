#!/usr/bin/env python
import argparse
import csv
import glob
import gzip
import json
import math
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train_xpu import GPT, GPTConfig, resolve_device_type

try:
    from safetensors import safe_open
    from safetensors.torch import load_file as load_safetensors_file
except Exception:
    safe_open = None
    load_safetensors_file = None


def load_env(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def expand_data_files(data_glob: str) -> List[str]:
    files = sorted(glob.glob(data_glob))
    if files:
        return files

    if data_glob.endswith(".json"):
        alt = data_glob[:-5] + ".json.gz"
        files = sorted(glob.glob(alt))
        if files:
            return files

    if not data_glob.endswith(".gz"):
        alt = data_glob + ".gz"
        files = sorted(glob.glob(alt))
        if files:
            return files
    return []


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode="rt", encoding="utf-8", errors="ignore")
    return open(path, mode="rt", encoding="utf-8", errors="ignore")


def iter_texts(path: str, text_field: str) -> Iterable[str]:
    with open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_field)
            if isinstance(text, str) and text:
                yield text


def resolve_device(device_arg: str) -> torch.device:
    device_type = resolve_device_type(device_arg)
    if device_type == "xpu":
        torch.xpu.set_device(0)
        return torch.device("xpu", 0)
    if device_type == "cuda":
        torch.cuda.set_device(0)
        return torch.device("cuda", 0)
    return torch.device("cpu")


def parse_dtype_name(name: str) -> Optional[torch.dtype]:
    key = name.lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16"}:
        return torch.float16
    if key in {"fp32", "float32"}:
        return None
    raise ValueError(f"unsupported dtype: {name}")


def choose_eval_dtype(
    dtype_arg: str,
    train_dtype: Optional[str],
    device: torch.device,
) -> Tuple[Optional[torch.dtype], str]:
    if dtype_arg == "auto":
        raw = (train_dtype or "fp32").lower()
    else:
        raw = dtype_arg.lower()

    dtype = parse_dtype_name(raw)
    if device.type not in {"cuda", "xpu"}:
        return None, "fp32"
    if dtype is None:
        return None, "fp32"
    if dtype == torch.float16:
        return torch.float16, "fp16"
    return torch.bfloat16, "bf16"


def choose_model_dtype(
    dtype_arg: str,
    amp_dtype: Optional[torch.dtype],
    device: torch.device,
) -> Tuple[Optional[torch.dtype], str]:
    if device.type not in {"cuda", "xpu"}:
        return None, "fp32"

    if dtype_arg == "auto":
        if amp_dtype == torch.float16:
            return torch.float16, "fp16"
        if amp_dtype == torch.bfloat16:
            return torch.bfloat16, "bf16"
        return None, "fp32"

    dtype = parse_dtype_name(dtype_arg)
    if dtype is None:
        return None, "fp32"
    if dtype == torch.float16:
        return torch.float16, "fp16"
    return torch.bfloat16, "bf16"


def maybe_autocast(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is not None and device.type in {"cuda", "xpu"}:
        return torch.autocast(device_type=device.type, dtype=amp_dtype)
    return nullcontext()


def is_safetensors_path(path: Path) -> bool:
    return path.suffix.lower() == ".safetensors"


def get_checkpoint_paths(
    checkpoints_dir: Path,
    pattern: str,
    include_last: bool,
    max_checkpoints: int,
) -> List[Path]:
    paths = [Path(p) for p in glob.glob(str(checkpoints_dir / pattern))]
    ckpt_re = re.compile(r"ckpt_(\d+)\.(?:pt|safetensors)$")

    def sort_key(p: Path) -> Tuple[int, str]:
        m = ckpt_re.search(p.name)
        if m:
            return int(m.group(1)), p.name
        return 10**18, p.name

    paths = sorted(paths, key=sort_key)

    if include_last:
        prefers_safe = "safetensors" in pattern
        prefers_pt = pattern.endswith(".pt")
        last_candidates: List[str]
        if prefers_safe:
            last_candidates = ["ckpt_last.safetensors", "ckpt_last.pt"]
        elif prefers_pt:
            last_candidates = ["ckpt_last.pt", "ckpt_last.safetensors"]
        else:
            last_candidates = ["ckpt_last.safetensors", "ckpt_last.pt"]
        for name in last_candidates:
            last_path = checkpoints_dir / name
            if last_path.exists():
                paths.append(last_path)
                break

    deduped: List[Path] = []
    seen = set()
    for p in paths:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    if max_checkpoints > 0:
        return deduped[:max_checkpoints]
    return deduped


def read_safetensors_meta(ckpt_path: Path) -> Dict[str, object]:
    if safe_open is None:
        raise ImportError(
            "safetensors is required to read .safetensors checkpoints. "
            "Install with: pip install safetensors"
        )

    raw_meta: Dict[str, str] = {}
    with safe_open(str(ckpt_path), framework="pt", device="cpu") as f:
        raw_meta = dict(f.metadata() or {})

    sidecar = ckpt_path.with_suffix(".meta.json")
    sidecar_meta: Dict[str, object] = {}
    if sidecar.exists():
        sidecar_meta = json.loads(sidecar.read_text(encoding="utf-8"))

    model_cfg_obj = sidecar_meta.get("model_config")
    model_cfg_raw = raw_meta.get("model_config")
    if model_cfg_raw:
        model_cfg_obj = json.loads(model_cfg_raw)
    if not isinstance(model_cfg_obj, dict):
        raise ValueError(
            f"missing model_config for {ckpt_path}. "
            f"Expected safetensors metadata['model_config'] or {sidecar}."
        )

    iter_raw = raw_meta.get("iter_num", sidecar_meta.get("iter_num", -1))
    try:
        iter_num = int(iter_raw)
    except (TypeError, ValueError):
        iter_num = -1

    train_dtype = str(raw_meta.get("train_dtype", sidecar_meta.get("train_dtype", "fp32")))
    return {
        "iter_num": iter_num,
        "train_dtype": train_dtype,
        "model_config": model_cfg_obj,
    }


def read_checkpoint_meta(ckpt_path: Path) -> Dict[str, object]:
    if is_safetensors_path(ckpt_path):
        return read_safetensors_meta(ckpt_path)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = {
        "iter_num": int(ckpt.get("iter_num", -1)),
        "train_dtype": str(ckpt.get("train_args", {}).get("dtype", "fp32")),
        "model_config": ckpt["model_config"],
    }
    del ckpt
    return meta


def load_checkpoint_for_eval(
    ckpt_path: Path,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    if is_safetensors_path(ckpt_path):
        if load_safetensors_file is None:
            raise ImportError(
                "safetensors is required to load .safetensors checkpoints. "
                "Install with: pip install safetensors"
            )
        meta = read_safetensors_meta(ckpt_path)
        state_dict = load_safetensors_file(str(ckpt_path), device="cpu")
        return state_dict, meta

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"]
    meta = {
        "iter_num": int(ckpt.get("iter_num", -1)),
        "train_dtype": str(ckpt.get("train_args", {}).get("dtype", "fp32")),
        "model_config": ckpt["model_config"],
    }
    del ckpt
    return state_dict, meta


def read_tokens_meta(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "text_field": data.get("text_field"),
        "bos_id": data.get("bos_id"),
        "eos_id": data.get("eos_id"),
    }


def maybe_trim_text(text: str, min_chars: int, max_chars: int) -> Optional[str]:
    if len(text) < min_chars:
        return None
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def build_eval_dataset(
    files: Sequence[str],
    tokenizer: Tokenizer,
    text_field: str,
    seq_len: int,
    max_eval_seqs: int,
    max_val_docs: int,
    encode_batch_size: int,
    min_chars: int,
    max_chars: int,
    bos_id: Optional[int],
    eos_id: Optional[int],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, int]]:
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")

    x_rows: List[List[int]] = []
    y_rows: List[List[int]] = []
    token_buffer: List[int] = []
    offset = 0
    docs_read = 0
    docs_used = 0
    total_tokens = 0
    batch: List[str] = []

    def compact_buffer() -> None:
        nonlocal token_buffer, offset
        if offset > 1_000_000 and offset > len(token_buffer) // 2:
            token_buffer = token_buffer[offset:]
            offset = 0

    def consume_tokens(ids: List[int]) -> bool:
        nonlocal offset, total_tokens
        token_buffer.extend(ids)
        total_tokens += len(ids)
        limit_reached = False

        while len(token_buffer) - offset >= (seq_len + 1):
            if max_eval_seqs > 0 and len(x_rows) >= max_eval_seqs:
                limit_reached = True
                break
            start = offset
            end = start + seq_len + 1
            seq = token_buffer[start:end]
            x_rows.append(seq[:-1])
            y_rows.append(seq[1:])
            offset += seq_len
            compact_buffer()
        return limit_reached

    def flush_batch() -> bool:
        nonlocal docs_used
        if not batch:
            return False
        encoded = tokenizer.encode_batch(batch)
        batch.clear()
        for enc in encoded:
            ids = enc.ids
            if bos_id is not None:
                ids = [bos_id] + ids
            if eos_id is not None:
                ids = ids + [eos_id]
            docs_used += 1
            if consume_tokens(ids):
                return True
        return False

    stop = False
    for path in files:
        if stop:
            break
        for text in iter_texts(path, text_field):
            docs_read += 1
            if max_val_docs > 0 and docs_read > max_val_docs:
                stop = True
                break
            text = maybe_trim_text(text, min_chars=min_chars, max_chars=max_chars)
            if text is None:
                continue
            batch.append(text)
            if len(batch) >= encode_batch_size and flush_batch():
                stop = True
                break
        if stop:
            break

    if not stop:
        flush_batch()

    if not x_rows:
        raise ValueError("validation set produced 0 sequences. Try lower --seq-len or increase docs.")

    x = torch.tensor(x_rows, dtype=torch.long)
    y = torch.tensor(y_rows, dtype=torch.long)
    stats = {
        "num_sequences": int(x.size(0)),
        "seq_len": int(seq_len),
        "docs_read": int(docs_read),
        "docs_used": int(docs_used),
        "stream_tokens": int(total_tokens),
    }
    return x, y, stats


def ckpt_iter_from_name(path: Path, fallback: int) -> int:
    m = re.search(r"ckpt_(\d+)\.(?:pt|safetensors)$", path.name)
    if not m:
        return fallback
    return int(m.group(1))


def is_oom_error(err: BaseException) -> bool:
    msg = str(err).lower()
    keys = ["out of memory", "not enough memory", "oom", "cannot allocate memory", "can't allocate memory"]
    return any(k in msg for k in keys)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "xpu":
        torch.xpu.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def compute_eval_loss(
    model: GPT,
    xb: torch.Tensor,
    yb: torch.Tensor,
    loss_chunk_tokens: int,
) -> torch.Tensor:
    if loss_chunk_tokens <= 0:
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("model returned no loss when targets were provided")
        return loss

    bsz, seqlen = xb.shape
    if seqlen > model.cfg.block_size:
        raise ValueError(f"seqlen {seqlen} > block_size {model.cfg.block_size}")

    pos = torch.arange(0, seqlen, device=xb.device, dtype=torch.long)
    hidden = model.wte(xb) + model.wpe(pos)[None, :, :]
    hidden = model.drop(hidden)
    for block in model.h:
        hidden = block(hidden)
    hidden = model.ln_f(hidden)

    total = torch.zeros((), device=hidden.device, dtype=torch.float32)
    total_tokens = 0
    step = max(1, int(loss_chunk_tokens))
    for start in range(0, seqlen, step):
        h = hidden[:, start : start + step, :]
        t = yb[:, start : start + step]
        logits = model.lm_head(h)
        chunk_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            t.reshape(-1),
            reduction="sum",
        )
        total += chunk_sum.float()
        total_tokens += int(t.numel())
        del logits

    del hidden
    if total_tokens <= 0:
        raise ValueError("empty loss token count")
    return total / float(total_tokens)


def evaluate_one_checkpoint(
    ckpt_path: Path,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    model_dtype: Optional[torch.dtype],
    loss_chunk_tokens: int,
    oom_auto_reduce: bool,
    nan_retry_fp32: bool,
) -> Dict[str, object]:
    t0 = time.time()
    state_dict, meta = load_checkpoint_for_eval(ckpt_path)

    model_cfg = GPTConfig(**meta["model_config"])
    model = GPT(model_cfg)
    if model_dtype is not None:
        model = model.to(device=device, dtype=model_dtype)
    else:
        model = model.to(device)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()

    total_loss = 0.0
    total_rows = 0
    current_batch_size = max(1, int(batch_size))
    oom_retries = 0
    fp32_retries = 0
    eval_amp_dtype = amp_dtype
    with torch.inference_mode():
        i = 0
        while i < x.size(0):
            try:
                xb = x[i : i + current_batch_size].to(device, non_blocking=True)
                yb = y[i : i + current_batch_size].to(device, non_blocking=True)
                with maybe_autocast(device=device, amp_dtype=eval_amp_dtype):
                    loss = compute_eval_loss(
                        model=model,
                        xb=xb,
                        yb=yb,
                        loss_chunk_tokens=loss_chunk_tokens,
                    )
                if not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(
                        f"non-finite eval loss at row={i} ckpt={ckpt_path.name}"
                    )
                rows = int(xb.size(0))
                total_loss += float(loss.item()) * rows
                total_rows += rows
                i += rows
                del xb
                del yb
                del loss
            except FloatingPointError as e:
                if (
                    nan_retry_fp32
                    and fp32_retries == 0
                    and (eval_amp_dtype is not None or model_dtype is not None)
                ):
                    print(
                        f"[warn] {e}; retry {ckpt_path.name} with fp32 forward for stable eval."
                    )
                    model = model.to(dtype=torch.float32)
                    eval_amp_dtype = None
                    fp32_retries += 1
                    total_loss = 0.0
                    total_rows = 0
                    i = 0
                    clear_device_cache(device)
                    continue
                raise
            except RuntimeError as e:
                if not (oom_auto_reduce and is_oom_error(e) and current_batch_size > 1):
                    raise
                old_bs = current_batch_size
                current_batch_size = max(1, current_batch_size // 2)
                oom_retries += 1
                clear_device_cache(device)
                print(
                    f"[warn] OOM during eval on {ckpt_path.name}, "
                    f"reduce batch_size: {old_bs} -> {current_batch_size}"
                )

    val_loss = total_loss / max(1, total_rows)
    ppl = math.exp(min(val_loss, 30.0))

    iter_num = int(meta.get("iter_num", -1))
    iter_num = ckpt_iter_from_name(ckpt_path, fallback=iter_num)

    result = {
        "checkpoint": str(ckpt_path),
        "name": ckpt_path.name,
        "iter_num": int(iter_num),
        "val_loss": float(val_loss),
        "ppl": float(ppl),
        "eval_rows": int(total_rows),
        "eval_batch_size": int(current_batch_size),
        "oom_retries": int(oom_retries),
        "fp32_retries": int(fp32_retries),
        "eval_seconds": float(time.time() - t0),
    }

    del model
    del meta
    clear_device_cache(device)
    return result


def load_model_for_generation(
    ckpt_path: Path,
    device: torch.device,
    model_dtype: Optional[torch.dtype],
) -> Tuple[GPT, GPTConfig]:
    state_dict, meta = load_checkpoint_for_eval(ckpt_path)
    cfg = GPTConfig(**meta["model_config"])
    model = GPT(cfg)
    if model_dtype is not None:
        model = model.to(device=device, dtype=model_dtype)
    else:
        model = model.to(device)
    model.load_state_dict(state_dict, strict=True)
    del state_dict
    model.eval()
    return model, cfg


def generate_sample(
    model: GPT,
    cfg: GPTConfig,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    skip_special_tokens: bool,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> Dict[str, object]:
    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        raise ValueError("prompt produced 0 tokens; please provide a non-empty prompt")

    ids: List[int] = list(prompt_ids)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            idx = ids[-cfg.block_size :]
            x = torch.tensor([idx], dtype=torch.long, device=device)
            with maybe_autocast(device=device, amp_dtype=amp_dtype):
                logits, _ = model(x)
            next_logits = logits[0, -1, :]

            if temperature <= 0:
                next_id = int(torch.argmax(next_logits).item())
            else:
                next_logits = next_logits / max(temperature, 1e-6)
                if top_k > 0:
                    k = min(top_k, int(next_logits.size(-1)))
                    values, indices = torch.topk(next_logits, k=k)
                    probs = torch.softmax(values, dim=-1)
                    pick = int(torch.multinomial(probs, num_samples=1).item())
                    next_id = int(indices[pick].item())
                else:
                    probs = torch.softmax(next_logits, dim=-1)
                    next_id = int(torch.multinomial(probs, num_samples=1).item())
            ids.append(next_id)

    completion_ids = ids[len(prompt_ids) :]
    return {
        "prompt": prompt,
        "prompt_tokens": int(len(prompt_ids)),
        "generated_tokens": int(len(completion_ids)),
        "full_text": tokenizer.decode(ids, skip_special_tokens=skip_special_tokens),
        "completion_text": tokenizer.decode(completion_ids, skip_special_tokens=skip_special_tokens),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = [
        "checkpoint",
        "name",
        "iter_num",
        "val_loss",
        "ppl",
        "eval_rows",
        "eval_batch_size",
        "oom_retries",
        "fp32_retries",
        "eval_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate all checkpoints on VAL_PATH and run sample generation."
    )
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("--val-glob", type=str, default=None, help="Override VAL_PATH from .env")
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    p.add_argument("--tokens-meta", type=Path, default=Path("data/tokens/meta.json"))
    p.add_argument("--text-field", type=str, default=None, help="default: from --tokens-meta or 'text'")
    p.add_argument("--bos-id", type=int, default=None)
    p.add_argument("--eos-id", type=int, default=None)
    p.add_argument("--checkpoints-dir", type=Path, default=Path("checkpoints/gpt100m"))
    p.add_argument("--checkpoint-pattern", type=str, default="ckpt_*.pt")
    p.add_argument("--include-last", action="store_true")
    p.add_argument("--max-checkpoints", type=int, default=0, help="0 means evaluate all")

    p.add_argument("--seq-len", type=int, default=0, help="0 means infer from first checkpoint")
    p.add_argument("--max-eval-seqs", type=int, default=512, help="0 means no limit")
    p.add_argument("--max-val-docs", type=int, default=0, help="0 means no limit")
    p.add_argument("--encode-batch-size", type=int, default=256)
    p.add_argument("--min-chars", type=int, default=1)
    p.add_argument("--max-chars", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "xpu", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    p.add_argument(
        "--model-dtype",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="weights dtype on device for eval; auto follows --dtype on XPU/CUDA",
    )
    p.add_argument(
        "--loss-chunk-tokens",
        type=int,
        default=0,
        help=">0 enables chunked LM loss over sequence dim to reduce peak logits memory",
    )
    p.add_argument(
        "--oom-auto-reduce",
        action="store_true",
        help="auto reduce eval batch size by half on OOM",
    )
    p.add_argument(
        "--nan-retry-fp32",
        action="store_true",
        help="retry a checkpoint in fp32 when non-finite loss is detected",
    )

    p.add_argument("--sample-prompt", type=str, default="今天天气不错，我们来聊聊人工智能。")
    p.add_argument("--sample-max-new-tokens", type=int, default=80)
    p.add_argument("--sample-temperature", type=float, default=0.8)
    p.add_argument("--sample-top-k", type=int, default=50)
    p.add_argument("--sample-each", action="store_true", help="Generate sample for each checkpoint")
    p.add_argument("--keep-special-tokens", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    env = load_env(args.env_file)
    val_glob = args.val_glob or env.get("VAL_PATH")
    if not val_glob:
        raise ValueError("VAL_PATH is missing. Set it in .env or pass --val-glob.")

    checkpoint_paths = get_checkpoint_paths(
        checkpoints_dir=args.checkpoints_dir,
        pattern=args.checkpoint_pattern,
        include_last=args.include_last,
        max_checkpoints=args.max_checkpoints,
    )
    if not checkpoint_paths:
        raise FileNotFoundError(
            f"no checkpoints found in {args.checkpoints_dir} with pattern={args.checkpoint_pattern!r}"
        )

    meta = read_checkpoint_meta(checkpoint_paths[0])
    if args.seq_len > 0:
        seq_len = int(args.seq_len)
    else:
        seq_len = int(meta["model_config"]["block_size"])
    train_dtype = str(meta.get("train_dtype", "fp32"))

    device = resolve_device(args.device)
    amp_dtype, amp_name = choose_eval_dtype(args.dtype, train_dtype, device)
    model_dtype, model_dtype_name = choose_model_dtype(args.model_dtype, amp_dtype, device)

    val_files = expand_data_files(val_glob)
    if not val_files:
        raise FileNotFoundError(f"No validation files matched: {val_glob}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    token_meta = read_tokens_meta(args.tokens_meta)
    text_field = str(args.text_field or token_meta.get("text_field") or "text")
    bos_id = args.bos_id if args.bos_id is not None else token_meta.get("bos_id")
    eos_id = args.eos_id if args.eos_id is not None else token_meta.get("eos_id")

    x, y, ds_stats = build_eval_dataset(
        files=val_files,
        tokenizer=tokenizer,
        text_field=text_field,
        seq_len=seq_len,
        max_eval_seqs=args.max_eval_seqs,
        max_val_docs=args.max_val_docs,
        encode_batch_size=args.encode_batch_size,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        bos_id=bos_id,
        eos_id=eos_id,
    )

    print(
        f"val files={len(val_files)} seqs={ds_stats['num_sequences']} seq_len={ds_stats['seq_len']} "
        f"docs_read={ds_stats['docs_read']} docs_used={ds_stats['docs_used']} stream_tokens={ds_stats['stream_tokens']} "
        f"text_field={text_field!r} bos_id={bos_id} eos_id={eos_id}"
    )
    has_pt = any(p.suffix.lower() == ".pt" for p in checkpoint_paths)
    if has_pt:
        print(
            "[hint] .pt checkpoint contains optimizer states. "
            "For eval-only loading, export to .safetensors and set --checkpoint-pattern ckpt_*.safetensors"
        )
    print(
        f"device={device} eval_dtype={amp_name} model_dtype={model_dtype_name} "
        f"loss_chunk_tokens={args.loss_chunk_tokens} oom_auto_reduce={args.oom_auto_reduce} "
        f"nan_retry_fp32={args.nan_retry_fp32} "
        f"checkpoints={len(checkpoint_paths)}"
    )

    results: List[Dict[str, object]] = []
    samples: List[Dict[str, object]] = []
    t_all = time.time()

    for idx, ckpt_path in enumerate(checkpoint_paths, start=1):
        result = evaluate_one_checkpoint(
            ckpt_path=ckpt_path,
            x=x,
            y=y,
            batch_size=args.batch_size,
            device=device,
            amp_dtype=amp_dtype,
            model_dtype=model_dtype,
            loss_chunk_tokens=args.loss_chunk_tokens,
            oom_auto_reduce=args.oom_auto_reduce,
            nan_retry_fp32=args.nan_retry_fp32,
        )
        results.append(result)
        print(
            f"[{idx:03d}/{len(checkpoint_paths):03d}] {result['name']} "
            f"iter={result['iter_num']} val_loss={result['val_loss']:.4f} ppl={result['ppl']:.4f} "
            f"time={result['eval_seconds']:.1f}s bs={result['eval_batch_size']} "
            f"oom_retries={result['oom_retries']} fp32_retries={result['fp32_retries']}"
        )

        if args.sample_each:
            model, cfg = load_model_for_generation(ckpt_path, device, model_dtype=model_dtype)
            sample = generate_sample(
                model=model,
                cfg=cfg,
                tokenizer=tokenizer,
                prompt=args.sample_prompt,
                max_new_tokens=args.sample_max_new_tokens,
                temperature=args.sample_temperature,
                top_k=args.sample_top_k,
                skip_special_tokens=not args.keep_special_tokens,
                device=device,
                amp_dtype=amp_dtype,
            )
            sample["checkpoint"] = str(ckpt_path)
            samples.append(sample)
            print(f"[sample:{ckpt_path.name}] {sample['completion_text']}")
            del model
            clear_device_cache(device)

    results_sorted = sorted(results, key=lambda r: float(r["val_loss"]))
    best = results_sorted[0]
    best_ckpt = Path(str(best["checkpoint"]))

    if not args.sample_each:
        model, cfg = load_model_for_generation(best_ckpt, device, model_dtype=model_dtype)
        sample = generate_sample(
            model=model,
            cfg=cfg,
            tokenizer=tokenizer,
            prompt=args.sample_prompt,
            max_new_tokens=args.sample_max_new_tokens,
            temperature=args.sample_temperature,
            top_k=args.sample_top_k,
            skip_special_tokens=not args.keep_special_tokens,
            device=device,
            amp_dtype=amp_dtype,
        )
        sample["checkpoint"] = str(best_ckpt)
        samples.append(sample)
        print(f"[sample:best={best_ckpt.name}] {sample['completion_text']}")
        del model
        clear_device_cache(device)

    elapsed = time.time() - t_all
    print(
        f"done. best={best['name']} iter={best['iter_num']} val_loss={best['val_loss']:.4f} "
        f"ppl={best['ppl']:.4f} total_time={elapsed:.1f}s"
    )

    out_json = args.out_json or (args.checkpoints_dir / "val_scores.json")
    out_csv = args.out_csv or (args.checkpoints_dir / "val_scores.csv")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env_file": str(args.env_file),
        "val_glob": val_glob,
        "tokenizer": str(args.tokenizer),
        "tokens_meta": str(args.tokens_meta),
        "text_field": text_field,
        "bos_id": bos_id,
        "eos_id": eos_id,
        "device": str(device),
        "eval_dtype": amp_name,
        "model_dtype": model_dtype_name,
        "loss_chunk_tokens": int(args.loss_chunk_tokens),
        "oom_auto_reduce": bool(args.oom_auto_reduce),
        "nan_retry_fp32": bool(args.nan_retry_fp32),
        "dataset_stats": ds_stats,
        "num_checkpoints": len(results),
        "best_checkpoint": best,
        "scores": results,
        "samples": samples,
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(out_csv, results)
    print(f"saved json: {out_json}")
    print(f"saved csv:  {out_csv}")


if __name__ == "__main__":
    main()
