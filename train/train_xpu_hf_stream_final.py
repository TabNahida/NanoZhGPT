#!/usr/bin/env python
import argparse
import json
import math
import os
import itertools
import queue as pyqueue
import threading
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch.nn.parallel import DistributedDataParallel as DDP


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


class TokenShardSampler:
    def __init__(
        self,
        data_dir: Path,
        split: str,
        seq_len: int,
        val_ratio: float,
        shards_per_batch: int = 1,
        contiguous_batch: bool = True,
        ram_cache_gb: float = 0.0,
    ) -> None:
        meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
        self.dtype = np.dtype(meta["dtype"])
        all_shards = list(meta["shards"])
        if not all_shards:
            raise ValueError("no shards in meta.json")
        split_shards = self._split_shards(all_shards, split, val_ratio)
        self.arrays: List[np.ndarray] = []
        self.lengths: List[int] = []
        self.max_starts: List[int] = []
        self.shards_per_batch = max(1, int(shards_per_batch))
        self.contiguous_batch = bool(contiguous_batch)
        self.ram_budget_bytes = int(max(0.0, ram_cache_gb) * (1024**3))
        self.cached_bytes = 0
        self.total_bytes = 0
        for item in split_shards:
            p = data_dir / item["file"]
            arr = np.memmap(p, dtype=self.dtype, mode="r")
            max_start = int(arr.shape[0]) - seq_len - 1
            if max_start > 0:
                shard_bytes = int(arr.nbytes)
                self.total_bytes += shard_bytes
                if self.cached_bytes + shard_bytes <= self.ram_budget_bytes:
                    arr = np.asarray(arr, dtype=self.dtype).copy()
                    self.cached_bytes += shard_bytes
                self.arrays.append(arr)
                self.lengths.append(int(arr.shape[0]))
                self.max_starts.append(max_start)
        if not self.arrays:
            raise ValueError(f"split={split} has no shard with enough tokens for seq_len={seq_len}")
        weights = np.asarray(self.max_starts, dtype=np.float64)
        self.probs = weights / weights.sum()
        self.seq_len = seq_len
        self.rng = np.random.default_rng()

    @staticmethod
    def _split_shards(
        shards: Sequence[Dict[str, int]], split: str, val_ratio: float
    ) -> Sequence[Dict[str, int]]:
        if split not in {"train", "val"}:
            raise ValueError(f"invalid split: {split}")
        if val_ratio <= 0.0 or len(shards) < 2:
            return shards if split == "train" else []
        n_val = int(len(shards) * val_ratio)
        n_val = max(1, n_val)
        n_val = min(n_val, len(shards) - 1)
        if split == "train":
            return shards[:-n_val]
        return shards[-n_val:]

    def _sample_from_shard(self, shard_idx: int, count: int) -> Tuple[np.ndarray, np.ndarray]:
        arr = self.arrays[shard_idx]
        arr_len = self.lengths[shard_idx]
        out_x = np.empty((count, self.seq_len), dtype=np.int64)
        out_y = np.empty((count, self.seq_len), dtype=np.int64)

        # Prefer one contiguous window per shard to reduce random small reads.
        needed = count * (self.seq_len + 1)
        if self.contiguous_batch and count > 1 and needed <= arr_len:
            base_max = arr_len - needed
            base = int(self.rng.integers(0, base_max + 1))
            flat = np.asarray(arr[base : base + needed], dtype=np.int64)
            block = flat.reshape(count, self.seq_len + 1)
            out_x[:, :] = block[:, :-1]
            out_y[:, :] = block[:, 1:]
            return out_x, out_y

        starts = self.rng.integers(0, self.max_starts[shard_idx] + 1, size=count, dtype=np.int64)
        starts.sort()
        for i, start in enumerate(starts):
            chunk = np.asarray(arr[int(start) : int(start) + self.seq_len + 1], dtype=np.int64)
            out_x[i, :] = chunk[:-1]
            out_y[i, :] = chunk[1:]
        return out_x, out_y

    def sample_batch_arrays(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = int(batch_size)
        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)

        shards_per_batch = min(self.shards_per_batch, batch_size, len(self.arrays))
        selected = self.rng.choice(
            len(self.arrays),
            size=shards_per_batch,
            replace=False if shards_per_batch <= len(self.arrays) else True,
            p=self.probs,
        )
        counts = np.full(shards_per_batch, batch_size // shards_per_batch, dtype=np.int64)
        counts[: batch_size % shards_per_batch] += 1

        cursor = 0
        for i, shard_idx in enumerate(selected):
            c = int(counts[i])
            if c <= 0:
                continue
            sx, sy = self._sample_from_shard(int(shard_idx), c)
            x[cursor : cursor + c, :] = sx
            y[cursor : cursor + c, :] = sy
            cursor += c

        if shards_per_batch > 1:
            perm = self.rng.permutation(batch_size)
            x = x[perm]
            y = y[perm]
        return x, y

    def sample_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x, y = self.sample_batch_arrays(batch_size)
        return torch.from_numpy(x), torch.from_numpy(y)



def _require_pkg(pkg: str, extra: str = ""):
    try:
        __import__(pkg)
    except Exception as e:
        hint = f"Please install '{pkg}' (pip install {pkg})"
        if extra:
            hint += f" and {extra}"
        raise RuntimeError(hint) from e


class HFStreamingTokenSampler:
    """Stream tokens from a Hugging Face dataset (datasets.load_dataset(..., streaming=True)).

    This sampler packs a continuous token stream into fixed-length sequences of (seq_len + 1),
    then returns (x, y) pairs where y is the next-token shift of x.

    Expected dataset format:
      - Prefer a pre-tokenized field (default: input_ids) containing a list[int].
      - Alternatively, provide --hf-text-field + --hf-tokenizer to tokenize on-the-fly (requires transformers).
    """

    def __init__(
        self,
        dataset: str,
        name: Optional[str],
        split: str,
        seq_len: int,
        token_field: str = "input_ids",
        text_field: str = "text",
        tokenizer_name_or_path: Optional[str] = None,
        shuffle_buffer: int = 0,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
        cache_dir: Optional[Path] = None,
        trust_remote_code: bool = False,
    ) -> None:
        _require_pkg("datasets")
        self.dataset = dataset
        self.name = name
        self.split = split
        self.seq_len = int(seq_len)
        self.token_field = str(token_field)
        self.text_field = str(text_field)
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.shuffle_buffer = int(max(0, shuffle_buffer))
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(max(1, world_size))
        self.cache_dir = cache_dir
        self.trust_remote_code = bool(trust_remote_code)

        self._tokenizer = None
        self._tok_buf: List[int] = []
        self._buf_pos = 0
        self._trim_threshold = 1_000_000  # tokens

        self._manual_mod_shard = False
        self._build_stream()
        self._reset_iter()

    def _build_stream(self) -> None:
        from datasets import load_dataset

        kwargs = dict(split=self.split, streaming=True)
        if self.name:
            kwargs["name"] = self.name
        if self.cache_dir is not None:
            kwargs["cache_dir"] = str(self.cache_dir)
        # datasets.load_dataset supports trust_remote_code in newer versions; safe to pass only if present.
        try:
            ds = load_dataset(self.dataset, **kwargs, trust_remote_code=self.trust_remote_code)
        except TypeError:
            ds = load_dataset(self.dataset, **kwargs)

        if self.shuffle_buffer > 0 and hasattr(ds, "shuffle"):
            ds = ds.shuffle(buffer_size=self.shuffle_buffer, seed=self.seed)

        # DDP sharding: prefer native node split/shard when available (fast, no wasted reads).
        if self.world_size > 1:
            try:
                from datasets.distributed import split_dataset_by_node

                ds = split_dataset_by_node(ds, world_size=self.world_size, rank=self.rank)
            except Exception:
                if hasattr(ds, "shard"):
                    try:
                        ds = ds.shard(num_shards=self.world_size, index=self.rank, contiguous=True)
                    except TypeError:
                        ds = ds.shard(num_shards=self.world_size, index=self.rank)
                else:
                    # Last resort fallback: manual modulo shard in the iterator (may waste reads).
                    self._manual_mod_shard = True

        self._stream = ds

    def _iter_examples(self):
        if not self._manual_mod_shard or self.world_size <= 1:
            yield from self._stream
            return
        # Manual modulo sharding fallback
        for i, ex in enumerate(self._stream):
            if (i % self.world_size) == self.rank:
                yield ex

    def _reset_iter(self) -> None:
        self._it = iter(self._iter_examples())

    def _get_tokenizer(self):
        if self._tokenizer is not None:
            return self._tokenizer
        if self.tokenizer_name_or_path is None:
            raise RuntimeError(
                "Dataset example has no token field and no tokenizer was provided. "
                "Provide --hf-token-field for pre-tokenized datasets, or --hf-tokenizer to tokenize text."
            )
        _require_pkg("transformers", extra="'transformers' (pip install transformers)")
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True)
        return self._tokenizer

    def _extract_tokens(self, ex: Dict) -> List[int]:
        if self.token_field in ex and ex[self.token_field] is not None:
            toks = ex[self.token_field]
            if isinstance(toks, (list, tuple, np.ndarray)):
                # Some datasets store batched tokens (list[list[int]]); flatten conservatively.
                if len(toks) > 0 and isinstance(toks[0], (list, tuple, np.ndarray)):
                    flat: List[int] = []
                    for row in toks:
                        flat.extend(int(t) for t in row)
                    return flat
                return [int(t) for t in toks]
            # Single scalar token id
            return [int(toks)]

        if self.text_field in ex and ex[self.text_field] is not None:
            tok = self._get_tokenizer()
            ids = tok(str(ex[self.text_field]), add_special_tokens=False)["input_ids"]
            return [int(t) for t in ids]

        keys = ", ".join(sorted(ex.keys()))
        raise KeyError(
            f"Example has neither token_field='{self.token_field}' nor text_field='{self.text_field}'. "
            f"Available keys: {keys}"
        )

    def _append_tokens(self, toks: List[int]) -> None:
        self._tok_buf.extend(toks)
        # Trim occasionally to avoid unbounded growth when using pointer slicing.
        if self._buf_pos > self._trim_threshold:
            self._tok_buf = self._tok_buf[self._buf_pos :]
            self._buf_pos = 0

    def _ensure(self, n: int) -> None:
        while (len(self._tok_buf) - self._buf_pos) < n:
            try:
                ex = next(self._it)
            except StopIteration:
                self._reset_iter()
                ex = next(self._it)
            toks = self._extract_tokens(ex)
            if toks:
                self._append_tokens(toks)

    def _take(self, n: int) -> List[int]:
        self._ensure(n)
        s = self._buf_pos
        e = s + n
        chunk = self._tok_buf[s:e]
        self._buf_pos = e
        return chunk

    def sample_batch_arrays(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = int(batch_size)
        x = np.empty((batch_size, self.seq_len), dtype=np.int64)
        y = np.empty((batch_size, self.seq_len), dtype=np.int64)
        need = self.seq_len + 1
        for i in range(batch_size):
            chunk = self._take(need)
            arr = np.asarray(chunk, dtype=np.int64)
            x[i, :] = arr[:-1]
            y[i, :] = arr[1:]
        return x, y

class BatchPrefetcher:
    def __init__(self, sampler: TokenShardSampler, batch_size: int, prefetch_batches: int) -> None:
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
    sampler,
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
            x_np, y_np = sampler.sample_batch_arrays(batch_size)
            x = torch.from_numpy(x_np).to(device, non_blocking=True)
            y = torch.from_numpy(y_np).to(device, non_blocking=True)
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
    p = argparse.ArgumentParser(description="Train a ~100M GPT on XPU/CUDA/CPU using token shards.")
    p.add_argument("--data-dir", type=Path, default=Path("data/tokens"))
    p.add_argument("--hf-dataset", type=str, default=None,
                   help="Hugging Face dataset name/path. If set, use HF streaming instead of --data-dir shards.")
    p.add_argument("--hf-name", type=str, default=None, help="HF dataset config/subset name (optional).")
    p.add_argument("--hf-train-split", type=str, default="train", help="HF split for training (streaming).")
    p.add_argument("--hf-val-split", type=str, default="validation",
                   help="HF split for validation (streaming). Use 'none' to disable.")
    p.add_argument("--hf-token-field", type=str, default="input_ids", help="Field name for pre-tokenized ids.")
    p.add_argument("--hf-text-field", type=str, default="text", help="Field name for raw text (if tokenizing).")
    p.add_argument("--hf-tokenizer", type=str, default=None,
                   help="Tokenizer name/path for on-the-fly tokenization (requires transformers).")
    p.add_argument("--hf-cache-dir", type=Path, default=None, help="HF datasets cache dir (optional).")
    p.add_argument("--hf-shuffle-buffer", type=int, default=10000,
                   help="Streaming shuffle buffer (0 disables). Larger improves randomness but uses more RAM.")
    p.add_argument("--hf-trust-remote-code", action="store_true",
                   help="Pass trust_remote_code=True to datasets.load_dataset when supported.")
    p.add_argument("--vocab-size", type=int, default=None,
                   help="Vocab size override (required for --hf-dataset unless meta.json exists).")
    p.add_argument("--out-dir", type=Path, default=Path("checkpoints/gpt100m"))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "xpu", "cuda", "cpu"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--compile", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--val-ratio", type=float, default=0.02)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--oom-auto-reduce", action="store_true")
    p.add_argument("--min-batch-size", type=int, default=1)
    p.add_argument("--prefetch-batches", type=int, default=16)
    p.add_argument("--shards-per-batch", type=int, default=1)
    p.add_argument("--ram-cache-gb", type=float, default=0.0)
    p.add_argument("--random-sample", action="store_true", help="Disable contiguous-window batch sampling")

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

    use_hf = args.hf_dataset is not None

    vocab_size = None
    data_meta = None
    meta_path = args.data_dir / "meta.json"
    if meta_path.exists():
        try:
            data_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            vocab_size = int(data_meta.get("vocab_size", 0)) or None
        except Exception:
            data_meta = None

    if use_hf:
        if args.vocab_size is not None:
            vocab_size = int(args.vocab_size)
        if vocab_size is None:
            raise ValueError(
                "When using --hf-dataset, please provide --vocab-size "
                "(or keep a meta.json under --data-dir with vocab_size)."
            )
    else:
        if data_meta is None:
            raise FileNotFoundError(
                f"meta.json not found under {args.data_dir}. "
                "Either provide token shards with meta.json, or use --hf-dataset for streaming."
            )
        vocab_size = int(data_meta["vocab_size"])

    cfg = GPTConfig(
        vocab_size=int(vocab_size),
        block_size=args.seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        bias=args.bias,
        tie_embeddings=not args.no_tie_embeddings,
    )

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

    if use_hf:
        train_sampler = HFStreamingTokenSampler(
            dataset=args.hf_dataset,
            name=args.hf_name,
            split=args.hf_train_split,
            seq_len=args.seq_len,
            token_field=args.hf_token_field,
            text_field=args.hf_text_field,
            tokenizer_name_or_path=args.hf_tokenizer,
            shuffle_buffer=args.hf_shuffle_buffer,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            cache_dir=args.hf_cache_dir,
            trust_remote_code=args.hf_trust_remote_code,
        )
        val_sampler = None
        if args.hf_val_split and str(args.hf_val_split).lower() not in {"none", "null", ""}:
            try:
                val_sampler = HFStreamingTokenSampler(
                    dataset=args.hf_dataset,
                    name=args.hf_name,
                    split=args.hf_val_split,
                    seq_len=args.seq_len,
                    token_field=args.hf_token_field,
                    text_field=args.hf_text_field,
                    tokenizer_name_or_path=args.hf_tokenizer,
                    shuffle_buffer=max(0, args.hf_shuffle_buffer // 4),
                    seed=args.seed + 1,
                    rank=rank,
                    world_size=world_size,
                    cache_dir=args.hf_cache_dir,
                    trust_remote_code=args.hf_trust_remote_code,
                )
            except Exception as e:
                if is_master:
                    print(f"[warn] could not create val sampler from split='{args.hf_val_split}': {e}")
                val_sampler = None
    else:
        train_sampler = TokenShardSampler(
            args.data_dir,
            split="train",
            seq_len=args.seq_len,
            val_ratio=args.val_ratio,
            shards_per_batch=args.shards_per_batch,
            contiguous_batch=not args.random_sample,
            ram_cache_gb=args.ram_cache_gb,
        )
        val_sampler = None
        if args.val_ratio > 0.0:
            try:
                val_sampler = TokenShardSampler(
                    args.data_dir,
                    split="val",
                    seq_len=args.seq_len,
                    val_ratio=args.val_ratio,
                    shards_per_batch=max(1, args.shards_per_batch),
                    contiguous_batch=not args.random_sample,
                    ram_cache_gb=max(0.0, args.ram_cache_gb * 0.25),
                )
            except ValueError:
                val_sampler = None

    model = GPT(cfg).to(device)
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    model_params = count_params(model)
    if is_master:
        print(f"device={device} ddp={ddp} world_size={world_size}")
        print(f"model params: {model_params:,}")
        print(f"grad_ckpt={args.gradient_checkpointing} oom_auto_reduce={args.oom_auto_reduce}")
        if hasattr(train_sampler, "total_bytes") and getattr(train_sampler, "total_bytes", 0) > 0:
            cached_gb = float(getattr(train_sampler, "cached_bytes", 0)) / (1024**3)
            total_gb = float(getattr(train_sampler, "total_bytes", 0)) / (1024**3)
            print(
                f"data cache: {cached_gb:.2f}/{total_gb:.2f} GB in RAM "
                f"(prefetch={args.prefetch_batches}, shards_per_batch={args.shards_per_batch})"
            )
        elif use_hf:
            print(
                f"data: hf://{args.hf_dataset} "
                f"name={args.hf_name} train_split={args.hf_train_split} val_split={args.hf_val_split} "
                f"token_field={args.hf_token_field} shuffle_buffer={args.hf_shuffle_buffer} "
                f"(prefetch={args.prefetch_batches})"
            )
            total_gb = train_sampler.total_bytes / (1024**3)
            print(
                f"data cache: {cached_gb:.2f}/{total_gb:.2f} GB in RAM "
                f"(prefetch={args.prefetch_batches}, shards_per_batch={args.shards_per_batch})"
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
