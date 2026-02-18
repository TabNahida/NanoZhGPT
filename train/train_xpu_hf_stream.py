#!/usr/bin/env python
import argparse
import math
import os
import queue as pyqueue
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None


@dataclass
class GPTConfig:
    vocab_size: int = 50000
    block_size: int = 2048
    n_layer: int = 14
    n_head: int = 10
    n_embd: int = 640
    ffn_mult: int = 4
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout
        self.c_attn = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, n_embd = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(n_embd, dim=2)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, n_embd)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        hidden = cfg.n_embd * cfg.ffn_mult
        self.c_fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gradient_checkpointing = False
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.h = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_gradient_checkpointing(self, enable: bool) -> None:
        self.gradient_checkpointing = bool(enable)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seqlen = idx.shape
        if seqlen > self.cfg.block_size:
            raise ValueError(f"seqlen {seqlen} > block_size {self.cfg.block_size}")
        pos = torch.arange(0, seqlen, device=idx.device, dtype=torch.long)
        x = self.wte(idx) + self.wpe(pos)[None, :, :]
        x = self.drop(x)
        for block in self.h:
            if self.training and self.gradient_checkpointing:
                x = checkpoint_utils.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


class HFStreamTokenSampler:
    def __init__(
        self,
        tokenizer_path: Path,
        dataset_name: str,
        dataset_config: Optional[str],
        split: str,
        text_field: str,
        seq_len: int,
        seed: int,
        rank: int,
        world_size: int,
        encode_batch_size: int = 256,
        shuffle_buffer: int = 10_000,
        min_chars: int = 1,
        max_chars: int = 20_000,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = "</s>",
        random_sample: bool = False,
        stream_buffer_tokens: int = 1_000_000,
    ) -> None:
        if load_dataset is None:
            raise ImportError("datasets is required. Install with: pip install datasets")
        if Tokenizer is None:
            raise ImportError("tokenizers is required. Install with: pip install tokenizers")

        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.vocab_size = int(self.tokenizer.get_vocab_size())

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.split = split
        self.text_field = text_field
        self.seq_len = int(seq_len)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.encode_batch_size = max(1, int(encode_batch_size))
        self.shuffle_buffer = max(0, int(shuffle_buffer))
        self.min_chars = max(0, int(min_chars))
        self.max_chars = int(max_chars)
        self.random_sample = bool(random_sample)
        self.stream_buffer_tokens = max(
            int(stream_buffer_tokens), self.seq_len * self.encode_batch_size + 1
        )
        self.rng = np.random.default_rng(self.seed + self.rank + (0 if split == "train" else 10_000))

        self.bos_id = self.tokenizer.token_to_id(bos_token) if bos_token else None
        self.eos_id = self.tokenizer.token_to_id(eos_token) if eos_token else None
        if bos_token and self.bos_id is None:
            raise ValueError(f"bos token {bos_token!r} not found in tokenizer")
        if eos_token and self.eos_id is None:
            raise ValueError(f"eos token {eos_token!r} not found in tokenizer")

        self.docs_seen = 0
        self.tokens_seen = 0
        self.cached_bytes = 0
        self.total_bytes = 0

        init_cap = max(self.stream_buffer_tokens * 2, self.seq_len + 2)
        self._buffer = np.empty(init_cap, dtype=np.int32)
        self._buf_start = 0
        self._buf_end = 0

        self._epoch = 0
        self._stream = None
        self._iterator = None
        self._reset_stream()

    def _build_stream(self, epoch: int):
        ds = load_dataset(
            path=self.dataset_name,
            name=self.dataset_config,
            split=self.split,
            streaming=True,
        )
        if self.world_size > 1:
            ds = ds.shard(num_shards=self.world_size, index=self.rank)
        if self.split == "train" and self.shuffle_buffer > 0:
            ds = ds.shuffle(seed=self.seed + epoch + self.rank, buffer_size=self.shuffle_buffer)
        return ds

    def _reset_stream(self) -> None:
        self._stream = self._build_stream(self._epoch)
        self._iterator = iter(self._stream)
        self._epoch += 1

    def _next_record(self) -> Dict[str, object]:
        while True:
            try:
                row = next(self._iterator)
                if isinstance(row, dict):
                    return row
            except StopIteration:
                self._reset_stream()

    def _collect_texts(self) -> List[str]:
        texts: List[str] = []
        scanned = 0
        scan_limit = max(10_000, self.encode_batch_size * 200)
        while len(texts) < self.encode_batch_size:
            row = self._next_record()
            scanned += 1
            text = row.get(self.text_field)
            if not isinstance(text, str):
                if scanned >= scan_limit and not texts:
                    raise ValueError(f"no valid text found in field={self.text_field!r}")
                continue
            if self.min_chars > 0 and len(text) < self.min_chars:
                if scanned >= scan_limit and not texts:
                    raise ValueError(f"text in field={self.text_field!r} failed min_chars={self.min_chars}")
                continue
            if self.max_chars > 0 and len(text) > self.max_chars:
                text = text[: self.max_chars]
            if text:
                texts.append(text)
        if not texts:
            raise ValueError("stream returned no texts for tokenization")
        return texts

    def _available_tokens(self) -> int:
        return self._buf_end - self._buf_start

    def _ensure_capacity(self, extra_tokens: int) -> None:
        needed = self._available_tokens() + extra_tokens
        if needed <= self._buffer.size - self._buf_start:
            return

        if self._buf_start > 0:
            remaining = self._available_tokens()
            self._buffer[:remaining] = self._buffer[self._buf_start : self._buf_end]
            self._buf_start = 0
            self._buf_end = remaining
            if needed <= self._buffer.size:
                return

        new_cap = self._buffer.size
        while new_cap < needed:
            new_cap *= 2
        new_buffer = np.empty(new_cap, dtype=self._buffer.dtype)
        remaining = self._available_tokens()
        new_buffer[:remaining] = self._buffer[self._buf_start : self._buf_end]
        self._buffer = new_buffer
        self._buf_start = 0
        self._buf_end = remaining

    def _append_encoded_batch(self, texts: List[str]) -> None:
        encoded = self.tokenizer.encode_batch(texts)
        extra = int(self.bos_id is not None) + int(self.eos_id is not None)
        total_new = sum(len(item.ids) + extra for item in encoded)
        if total_new <= 0:
            return

        self._ensure_capacity(total_new)
        cursor = self._buf_end
        for item in encoded:
            ids = item.ids
            if self.bos_id is not None:
                self._buffer[cursor] = self.bos_id
                cursor += 1
            if ids:
                n = len(ids)
                self._buffer[cursor : cursor + n] = np.asarray(ids, dtype=np.int32)
                cursor += n
            if self.eos_id is not None:
                self._buffer[cursor] = self.eos_id
                cursor += 1
            self.docs_seen += 1

        added = cursor - self._buf_end
        self._buf_end = cursor
        self.tokens_seen += int(added)

    def _refill(self, min_tokens: int) -> None:
        target = max(int(min_tokens), self.stream_buffer_tokens)
        while self._available_tokens() < target:
            texts = self._collect_texts()
            self._append_encoded_batch(texts)

    def _compact_if_needed(self) -> None:
        if self._buf_start == 0:
            return
        remaining = self._available_tokens()
        if self._buf_start < (self._buffer.size // 2) and remaining >= self.seq_len * 4:
            return
        self._buffer[:remaining] = self._buffer[self._buf_start : self._buf_end]
        self._buf_start = 0
        self._buf_end = remaining

    def _sample_random(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        needed = self.seq_len + 1
        self._refill(needed + batch_size * self.seq_len)
        avail = self._available_tokens()
        max_start = avail - self.seq_len - 1
        if max_start <= 0:
            return self._sample_contiguous(batch_size)

        starts = self.rng.integers(0, max_start + 1, size=batch_size, dtype=np.int64)
        starts.sort()
        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)
        base = self._buf_start
        for i, s in enumerate(starts):
            st = base + int(s)
            chunk = self._buffer[st : st + self.seq_len + 1]
            x[i, :] = chunk[:-1]
            y[i, :] = chunk[1:]

        advance = min(batch_size * self.seq_len, max(0, avail - (self.seq_len + 1)))
        self._buf_start += int(advance)
        self._compact_if_needed()
        return x, y

    def _sample_contiguous(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        needed = batch_size * self.seq_len + 1
        self._refill(needed)
        start = self._buf_start
        end = start + needed
        flat = np.asarray(self._buffer[start:end], dtype=np.int64)
        x = flat[:-1].reshape(batch_size, self.seq_len)
        y = flat[1:].reshape(batch_size, self.seq_len)
        self._buf_start += batch_size * self.seq_len
        self._compact_if_needed()
        return x, y

    def sample_batch_arrays(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"invalid batch size: {batch_size}")
        if self.random_sample:
            return self._sample_random(batch_size)
        return self._sample_contiguous(batch_size)

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.sample_batch_arrays(batch_size)
        return torch.from_numpy(x), torch.from_numpy(y)


class BatchPrefetcher:
    def __init__(self, sampler: HFStreamTokenSampler, batch_size: int, prefetch_batches: int) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.prefetch_batches = max(1, int(prefetch_batches))
        self.queue: pyqueue.Queue = pyqueue.Queue(maxsize=self.prefetch_batches)
        self.stop_event = threading.Event()
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                batch = self.sampler.sample_batch_arrays(self.batch_size)
                while not self.stop_event.is_set():
                    try:
                        self.queue.put(batch, timeout=0.1)
                        break
                    except pyqueue.Full:
                        continue
        except Exception as e:
            while not self.stop_event.is_set():
                try:
                    self.queue.put(e, timeout=0.1)
                    break
                except pyqueue.Full:
                    continue

    def next_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        item = self.queue.get()
        if isinstance(item, Exception):
            raise RuntimeError("prefetch worker failed") from item
        return item

    def close(self) -> None:
        self.stop_event.set()
        self.worker.join(timeout=5.0)


def resolve_device_type(device_arg: str) -> str:
    if device_arg == "auto":
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device_arg


def setup_distributed(device_type: str):
    ddp = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not ddp:
        return False, 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    backend = "xccl" if device_type == "xpu" else "nccl" if device_type == "cuda" else "gloo"
    dist.init_process_group(backend=backend)
    return True, rank, world_size, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def get_device(device_arg: str, local_rank: int) -> torch.device:
    device_arg = resolve_device_type(device_arg)
    if device_arg == "xpu":
        torch.xpu.set_device(local_rank)
        return torch.device("xpu", local_rank)
    if device_arg == "cuda":
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_lr(iter_num: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float) -> float:
    if iter_num < warmup_iters:
        return lr * float(iter_num) / float(max(1, warmup_iters))
    if iter_num >= max_iters:
        return min_lr
    decay_ratio = (iter_num - warmup_iters) / float(max(1, max_iters - warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


def unwrap_model(model: nn.Module) -> nn.Module:
    m = model
    if isinstance(m, DDP):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


def estimate_loss(
    model: nn.Module,
    sampler: HFStreamTokenSampler,
    eval_iters: int,
    batch_size: int,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
) -> float:
    model.eval()
    losses = []
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type in {"cuda", "xpu"}
        else nullcontext()
    )
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = sampler.sample_batch(batch_size)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast_ctx:
                _, loss = model(x, y)
            losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def try_enable_compile(
    model: nn.Module,
    enable_compile: bool,
    device: torch.device,
    seq_len: int,
    amp_dtype: Optional[torch.dtype],
    is_master: bool,
) -> Tuple[nn.Module, bool]:
    if not enable_compile:
        return model, False
    eager_model = model
    try:
        compiled = torch.compile(model)
        warmup_len = min(seq_len, 8)
        dummy_x = torch.zeros((1, warmup_len), dtype=torch.long, device=device)
        dummy_y = torch.zeros((1, warmup_len), dtype=torch.long, device=device)
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=amp_dtype)
            if amp_dtype is not None and device.type in {"cuda", "xpu"}
            else nullcontext()
        )
        with torch.no_grad():
            with autocast_ctx:
                compiled(dummy_x, dummy_y)
        if is_master:
            print("torch.compile enabled")
        return compiled, True
    except Exception as e:
        if is_master:
            msg = str(e).strip().splitlines()[0] if str(e).strip() else "unknown error"
            print(f"[warn] torch.compile failed ({type(e).__name__}: {msg})")
            print("[warn] fallback to eager mode; rerun without --compile for the same behavior.")
        return eager_model, False


def is_oom_error(err: BaseException) -> bool:
    msg = str(err).lower()
    keys = ["out of memory", "not enough memory", "oom", "cannot allocate memory", "can't allocate memory"]
    return any(k in msg for k in keys)


def get_memory_stats(device: torch.device) -> Optional[Dict[str, int]]:
    if device.type == "xpu":
        alloc = int(torch.xpu.memory_allocated(device))
        reserved = int(torch.xpu.memory_reserved(device))
        peak_alloc = int(torch.xpu.max_memory_allocated(device))
        peak_reserved = int(torch.xpu.max_memory_reserved(device))
        free = -1
        total = -1
        if hasattr(torch.xpu, "mem_get_info"):
            try:
                free, total = torch.xpu.mem_get_info(device)
                free = int(free)
                total = int(total)
            except Exception:
                pass
        return {
            "alloc": alloc,
            "reserved": reserved,
            "peak_alloc": peak_alloc,
            "peak_reserved": peak_reserved,
            "free": free,
            "total": total,
        }
    if device.type == "cuda":
        alloc = int(torch.cuda.memory_allocated(device))
        reserved = int(torch.cuda.memory_reserved(device))
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        free = -1
        total = -1
        if hasattr(torch.cuda, "mem_get_info"):
            try:
                free, total = torch.cuda.mem_get_info(device)
                free = int(free)
                total = int(total)
            except Exception:
                pass
        return {
            "alloc": alloc,
            "reserved": reserved,
            "peak_alloc": peak_alloc,
            "peak_reserved": peak_reserved,
            "free": free,
            "total": total,
        }
    return None


def fmt_gb(num_bytes: int) -> str:
    if num_bytes is None or num_bytes < 0:
        return "n/a"
    return f"{num_bytes / (1024**3):.2f}G"


def fmt_eta(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "n/a"
    s = int(seconds + 0.5)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def to_basic_types(obj):
    if isinstance(obj, dict):
        return {k: to_basic_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_basic_types(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_basic_types(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a ~100M GPT on XPU/CUDA/CPU using HF streaming data.")
    p.add_argument("--data-dir", type=Path, default=Path("data/tokens"), help=argparse.SUPPRESS)
    p.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    p.add_argument("--hf-dataset", type=str, default="allenai/c4")
    p.add_argument("--hf-config", type=str, default="zh")
    p.add_argument("--hf-train-split", type=str, default="train")
    p.add_argument("--hf-val-split", type=str, default="validation")
    p.add_argument("--hf-text-field", type=str, default="text")
    p.add_argument("--stream-shuffle-buffer", type=int, default=50_000)
    p.add_argument("--encode-batch-size", type=int, default=256)
    p.add_argument("--stream-buffer-tokens", type=int, default=1_000_000)
    p.add_argument("--min-chars", type=int, default=1)
    p.add_argument("--max-chars", type=int, default=20_000)
    p.add_argument("--bos-token", type=str, default=None)
    p.add_argument("--eos-token", type=str, default="</s>")

    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/gpt100m"))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "xpu", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--val-ratio", type=float, default=0.02, help="<=0 disables val evaluation")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--oom-auto-reduce", action="store_true")
    p.add_argument("--min-batch-size", type=int, default=1)
    p.add_argument("--prefetch-batches", type=int, default=16)
    p.add_argument("--shards-per-batch", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--ram-cache-gb", type=float, default=0.0, help=argparse.SUPPRESS)
    p.add_argument("--random-sample", action="store_true", help="Sample random windows from stream buffer")

    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--n-layer", type=int, default=14)
    p.add_argument("--n-head", type=int, default=10)
    p.add_argument("--n-embd", type=int, default=640)
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--no-tie-embeddings", action="store_true")

    p.add_argument("--batch-size", type=int, default=2, help="Micro batch size per device")
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--max-iters", type=int, default=20000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr", type=float, default=3e-5)
    p.add_argument("--warmup-iters", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)

    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--log-interval", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    runtime_device_type = resolve_device_type(args.device)
    ddp, rank, world_size, local_rank = setup_distributed(runtime_device_type)
    is_master = rank == 0
    device = get_device(args.device, local_rank)
    device_type = device.type

    seed = args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    amp_dtype: Optional[torch.dtype] = None
    if args.dtype == "bf16":
        amp_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        amp_dtype = torch.float16
    hf_config = args.hf_config.strip() if args.hf_config is not None else None
    if hf_config == "":
        hf_config = None

    train_sampler = HFStreamTokenSampler(
        tokenizer_path=args.tokenizer,
        dataset_name=args.hf_dataset,
        dataset_config=hf_config,
        split=args.hf_train_split,
        text_field=args.hf_text_field,
        seq_len=args.seq_len,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        encode_batch_size=args.encode_batch_size,
        shuffle_buffer=args.stream_shuffle_buffer,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        bos_token=args.bos_token,
        eos_token=args.eos_token,
        random_sample=args.random_sample,
        stream_buffer_tokens=args.stream_buffer_tokens,
    )
    cfg = GPTConfig(
        vocab_size=train_sampler.vocab_size,
        block_size=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        bias=args.bias,
        tie_embeddings=not args.no_tie_embeddings,
    )

    val_sampler = None
    if args.val_ratio > 0.0:
        try:
            val_sampler = HFStreamTokenSampler(
                tokenizer_path=args.tokenizer,
                dataset_name=args.hf_dataset,
                dataset_config=hf_config,
                split=args.hf_val_split,
                text_field=args.hf_text_field,
                seq_len=args.seq_len,
                seed=args.seed + 1,
                rank=rank,
                world_size=world_size,
                encode_batch_size=args.encode_batch_size,
                shuffle_buffer=0,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                bos_token=args.bos_token,
                eos_token=args.eos_token,
                random_sample=False,
                stream_buffer_tokens=max(args.seq_len * 8, args.stream_buffer_tokens // 4),
            )
        except Exception as e:
            val_sampler = None
            if is_master:
                msg = str(e).strip().splitlines()[0] if str(e).strip() else "unknown error"
                print(f"[warn] val stream init failed ({type(e).__name__}: {msg}); validation disabled.")

    model = GPT(cfg).to(device)
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    model_params = count_params(model)
    if is_master:
        print(f"device={device} ddp={ddp} world_size={world_size}")
        print(f"model params: {model_params:,}")
        print(f"grad_ckpt={args.gradient_checkpointing} oom_auto_reduce={args.oom_auto_reduce}")
        print(
            f"stream={args.hf_dataset}/{hf_config}:{args.hf_train_split} "
            f"text_field={args.hf_text_field} prefetch={args.prefetch_batches}"
        )

    model, compile_enabled = try_enable_compile(
        model=model,
        enable_compile=args.compile,
        device=device,
        seq_len=args.seq_len,
        amp_dtype=amp_dtype,
        is_master=is_master,
    )
    if is_master and args.compile and not compile_enabled:
        cxx = os.environ.get("CXX")
        if cxx:
            print(f"[hint] CXX={cxx}")
        else:
            print("[hint] set CXX to your compiler path if you want to use --compile.")

    if ddp:
        ddp_device_ids = [local_rank] if device_type in {"cuda", "xpu"} else None
        model = DDP(model, device_ids=ddp_device_ids)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    scaler_enabled = args.dtype == "fp16"
    try:
        scaler = torch.amp.GradScaler(device_type, enabled=scaler_enabled)
    except TypeError:
        scaler = torch.amp.GradScaler(enabled=scaler_enabled)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    iter_num = 0
    best_val = float("inf")

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrap_model(model).load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        iter_num = int(ckpt.get("iter_num", 0))
        best_val = float(ckpt.get("best_val", best_val))
        if is_master:
            print(f"resumed from {args.resume} at iter={iter_num}")

    oom_auto_reduce = bool(args.oom_auto_reduce and not ddp)
    if args.oom_auto_reduce and ddp and is_master:
        print("[warn] --oom-auto-reduce is disabled in DDP mode.")

    current_batch_size = max(args.min_batch_size, args.batch_size)
    tokens_per_iter = current_batch_size * args.seq_len * args.grad_accum * world_size
    if is_master:
        print(f"tokens/iter={tokens_per_iter:,} (micro_batch={current_batch_size})")

    if device_type == "xpu" and hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats(device)
    if device_type == "cuda" and hasattr(torch.cuda, "reset_peak_memory_stats"):
        torch.cuda.reset_peak_memory_stats(device)

    train_prefetcher = None
    if args.prefetch_batches > 0:
        train_prefetcher = BatchPrefetcher(
            sampler=train_sampler,
            batch_size=current_batch_size,
            prefetch_batches=args.prefetch_batches,
        )

    model.train()
    t0 = time.time()
    train_start = time.time()
    start_iter = iter_num
    tokens_seen = int(iter_num * tokens_per_iter)
    log_data_wait = 0.0
    log_h2d = 0.0
    log_compute = 0.0
    try:
        while iter_num < args.max_iters:
            lr = get_lr(iter_num, args.warmup_iters, args.max_iters, args.lr, args.min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            running_loss = 0.0
            step_aborted = False

            for micro in range(args.grad_accum):
                try:
                    t_data = time.time()
                    if train_prefetcher is None:
                        x_np, y_np = train_sampler.sample_batch_arrays(current_batch_size)
                    else:
                        x_np, y_np = train_prefetcher.next_batch()
                    log_data_wait += time.time() - t_data

                    t_h2d = time.time()
                    x = torch.from_numpy(x_np).to(device, non_blocking=True)
                    y = torch.from_numpy(y_np).to(device, non_blocking=True)
                    log_h2d += time.time() - t_h2d

                    ddp_sync = not ddp or (micro == args.grad_accum - 1)
                    sync_ctx = nullcontext() if ddp_sync else model.no_sync()
                    autocast_ctx = (
                        torch.autocast(device_type=device_type, dtype=amp_dtype)
                        if amp_dtype is not None and device_type in {"cuda", "xpu"}
                        else nullcontext()
                    )
                    t_compute = time.time()
                    with sync_ctx:
                        with autocast_ctx:
                            _, loss = model(x, y)
                            loss = loss / args.grad_accum
                        running_loss += float(loss.item())
                        if scaler_enabled:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                    log_compute += time.time() - t_compute
                except RuntimeError as e:
                    if not (oom_auto_reduce and is_oom_error(e)):
                        raise
                    old_bs = current_batch_size
                    if old_bs <= args.min_batch_size:
                        if is_master:
                            print(
                                f"[error] OOM at min batch_size={old_bs}. "
                                "Try --gradient-checkpointing or shorter --seq-len."
                            )
                        raise
                    new_bs = max(args.min_batch_size, old_bs // 2)
                    optimizer.zero_grad(set_to_none=True)
                    if device_type == "xpu":
                        torch.xpu.empty_cache()
                    elif device_type == "cuda":
                        torch.cuda.empty_cache()
                    current_batch_size = new_bs
                    tokens_per_iter = current_batch_size * args.seq_len * args.grad_accum * world_size
                    if train_prefetcher is not None:
                        train_prefetcher.close()
                        train_prefetcher = BatchPrefetcher(
                            sampler=train_sampler,
                            batch_size=current_batch_size,
                            prefetch_batches=args.prefetch_batches,
                        )
                    step_aborted = True
                    if is_master:
                        print(
                            f"[warn] OOM detected, reduce micro_batch: {old_bs} -> {new_bs}. "
                            f"new tokens/iter={tokens_per_iter:,}"
                        )
                    break

            if step_aborted:
                continue

            if scaler_enabled:
                scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if scaler_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            iter_num += 1
            tokens_seen += int(tokens_per_iter)

            if iter_num % args.log_interval == 0:
                dt = max(time.time() - t0, 1e-6)
                iter_per_s = args.log_interval / dt
                tok_per_s = iter_per_s * tokens_per_iter
                loss_report = running_loss
                if ddp:
                    t = torch.tensor([loss_report], device=device)
                    dist.all_reduce(t, op=dist.ReduceOp.SUM)
                    loss_report = float(t.item() / world_size)
                io_total = log_data_wait + log_h2d
                step_total = io_total + log_compute
                io_pct = 100.0 * io_total / max(step_total, 1e-9)
                progress = 100.0 * iter_num / max(1, args.max_iters)
                elapsed = time.time() - train_start
                done_iters = max(1, iter_num - start_iter)
                avg_iter_s = done_iters / max(elapsed, 1e-6)
                eta_sec = (args.max_iters - iter_num) / max(avg_iter_s, 1e-9)
                mem_stats = get_memory_stats(device)
                mem_msg = ""
                if mem_stats is not None:
                    mem_msg = (
                        f" mem={fmt_gb(mem_stats['alloc'])}/{fmt_gb(mem_stats['reserved'])}"
                        f" peak={fmt_gb(mem_stats['peak_alloc'])}/{fmt_gb(mem_stats['peak_reserved'])}"
                    )
                    if mem_stats["total"] > 0:
                        mem_msg += f" total={fmt_gb(mem_stats['total'])}"
                if is_master:
                    print(
                        f"iter={iter_num:6d}/{args.max_iters} ({progress:6.2f}%) eta={fmt_eta(eta_sec)} "
                        f"loss={loss_report:.4f} lr={lr:.3e} bs={current_batch_size} "
                        f"iter/s={iter_per_s:.2f} tok/s={tok_per_s:,.0f} tokens={tokens_seen:,} "
                        f"io%={io_pct:.1f} data={log_data_wait:.2f}s h2d={log_h2d:.2f}s compute={log_compute:.2f}s"
                        f"{mem_msg}"
                    )
                t0 = time.time()
                log_data_wait = 0.0
                log_h2d = 0.0
                log_compute = 0.0

            if iter_num % args.eval_interval == 0:
                eval_bs = current_batch_size
                train_eval = estimate_loss(
                    model, train_sampler, args.eval_iters, eval_bs, device=device, amp_dtype=amp_dtype
                )
                val_eval = float("nan")
                if val_sampler is not None:
                    val_eval = estimate_loss(
                        model, val_sampler, args.eval_iters, eval_bs, device=device, amp_dtype=amp_dtype
                    )
                if ddp:
                    vals = torch.tensor([train_eval, val_eval], device=device)
                    if math.isnan(val_eval):
                        vals[1] = 0.0
                    dist.all_reduce(vals, op=dist.ReduceOp.SUM)
                    train_eval = float(vals[0].item() / world_size)
                    val_eval = float("nan") if val_sampler is None else float(vals[1].item() / world_size)
                if is_master:
                    print(f"[eval] iter={iter_num} train_loss={train_eval:.4f} val_loss={val_eval:.4f}")
                if val_sampler is not None and not math.isnan(val_eval):
                    best_val = min(best_val, val_eval)
                model.train()

            if is_master and iter_num % args.save_interval == 0:
                ckpt = {
                    "model": unwrap_model(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "iter_num": iter_num,
                    "best_val": best_val,
                    "model_config": asdict(cfg),
                    "train_args": to_basic_types(vars(args)),
                }
                ckpt_path = args.out_dir / f"ckpt_{iter_num:06d}.pt"
                torch.save(ckpt, ckpt_path)
                print(f"saved {ckpt_path}")

        if is_master:
            last = args.out_dir / "ckpt_last.pt"
            ckpt = {
                "model": unwrap_model(model).state_dict(),
                "optimizer": optimizer.state_dict(),
                "iter_num": iter_num,
                "best_val": best_val,
                "model_config": asdict(cfg),
                "train_args": to_basic_types(vars(args)),
            }
            torch.save(ckpt, last)
            print(f"training done. final checkpoint: {last}")
    finally:
        if train_prefetcher is not None:
            train_prefetcher.close()
        cleanup_distributed()


if __name__ == "__main__":
    main()
