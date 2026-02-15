#!/usr/bin/env python
import argparse
import gzip
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from tokenizers import Tokenizer


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
    import glob

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
    return files


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


class ShardWriter:
    def __init__(
        self,
        out_dir: Path,
        dtype: np.dtype,
        max_tokens_per_shard: int,
        prefix: str = "shard",
    ) -> None:
        self.out_dir = out_dir
        self.dtype = dtype
        self.max_tokens_per_shard = max_tokens_per_shard
        self.prefix = prefix
        self.shard_index = 0
        self.current_tokens = 0
        self.total_tokens = 0
        self.shards = []
        self._fp = None
        self._path = None
        self._open_new_shard()

    def _open_new_shard(self) -> None:
        self._path = self.out_dir / f"{self.prefix}_{self.shard_index:06d}.bin"
        self._fp = open(self._path, "wb")
        self.current_tokens = 0

    def _close_current_shard(self) -> None:
        if self._fp is None:
            return
        self._fp.close()
        self.shards.append({"file": self._path.name, "num_tokens": self.current_tokens})
        self._fp = None

    def add_tokens(self, token_ids: List[int]) -> None:
        if not token_ids:
            return
        arr = np.asarray(token_ids, dtype=self.dtype)
        idx = 0
        while idx < arr.size:
            remaining = self.max_tokens_per_shard - self.current_tokens
            if remaining <= 0:
                self._close_current_shard()
                self.shard_index += 1
                self._open_new_shard()
                remaining = self.max_tokens_per_shard
            take = min(remaining, arr.size - idx)
            chunk = arr[idx : idx + take]
            chunk.tofile(self._fp)
            self.current_tokens += int(take)
            self.total_tokens += int(take)
            idx += int(take)

    def close(self) -> None:
        if self._fp is None:
            return
        self._close_current_shard()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize json/json.gz into binary token shards.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--data-glob", type=str, default=None, help="Override DATA_PATH from .env")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--tokenizer", type=Path, default=Path("tokenizer.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/tokens"))
    parser.add_argument("--max-tokens-per-shard", type=int, default=50_000_000)
    parser.add_argument("--encode-batch-size", type=int, default=256)
    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=20_000)
    parser.add_argument("--max-docs", type=int, default=0, help="0 means no limit")
    parser.add_argument("--eos-token", type=str, default="</s>")
    parser.add_argument("--bos-token", type=str, default=None)
    parser.add_argument("--progress-every", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = load_env(args.env_file)
    data_glob = args.data_glob or env.get("DATA_PATH")
    if not data_glob:
        raise ValueError("DATA_PATH is missing. Set it in .env or pass --data-glob.")

    files = expand_data_files(data_glob)
    if not files:
        raise FileNotFoundError(f"No files matched: {data_glob}")

    tok = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tok.get_vocab_size()
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32

    eos_id = tok.token_to_id(args.eos_token) if args.eos_token else None
    if args.eos_token and eos_id is None:
        raise ValueError(f"eos token {args.eos_token!r} not found in tokenizer")

    bos_id = tok.token_to_id(args.bos_token) if args.bos_token else None
    if args.bos_token and bos_id is None:
        raise ValueError(f"bos token {args.bos_token!r} not found in tokenizer")

    writer = ShardWriter(
        out_dir=args.out_dir,
        dtype=dtype,
        max_tokens_per_shard=args.max_tokens_per_shard,
        prefix="train",
    )

    start_t = time.time()
    num_docs = 0
    num_skipped = 0
    batch: List[str] = []

    def flush_batch() -> None:
        nonlocal num_docs
        if not batch:
            return
        encoded = tok.encode_batch(batch)
        for enc in encoded:
            if args.max_docs > 0 and num_docs >= args.max_docs:
                break
            ids = enc.ids
            if bos_id is not None:
                ids = [bos_id] + ids
            if eos_id is not None:
                ids = ids + [eos_id]
            writer.add_tokens(ids)
            num_docs += 1
        batch.clear()

    for path in files:
        for text in iter_texts(path, args.text_field):
            if len(text) < args.min_chars:
                num_skipped += 1
                continue
            if args.max_chars > 0 and len(text) > args.max_chars:
                text = text[: args.max_chars]
            batch.append(text)
            if len(batch) >= args.encode_batch_size:
                flush_batch()
            if args.max_docs > 0 and num_docs + len(batch) >= args.max_docs:
                flush_batch()
            if args.max_docs > 0 and num_docs >= args.max_docs:
                break
            if args.progress_every > 0 and num_docs > 0 and num_docs % args.progress_every == 0:
                elapsed = max(time.time() - start_t, 1e-6)
                rate_docs = num_docs / elapsed
                rate_tokens = writer.total_tokens / elapsed
                print(
                    f"[progress] docs={num_docs:,} tokens={writer.total_tokens:,} "
                    f"docs/s={rate_docs:.1f} tok/s={rate_tokens:.0f}"
                )
        if args.max_docs > 0 and num_docs >= args.max_docs:
            break

    flush_batch()
    writer.close()

    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tokenizer_path": str(args.tokenizer),
        "text_field": args.text_field,
        "data_glob": data_glob,
        "num_input_files": len(files),
        "input_files": files,
        "vocab_size": vocab_size,
        "dtype": np.dtype(dtype).name,
        "eos_token": args.eos_token,
        "eos_id": eos_id,
        "bos_token": args.bos_token,
        "bos_id": bos_id,
        "max_tokens_per_shard": args.max_tokens_per_shard,
        "num_docs": num_docs,
        "num_skipped": num_skipped,
        "total_tokens": writer.total_tokens,
        "shards": writer.shards,
    }
    meta_path = args.out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = max(time.time() - start_t, 1e-6)
    print(f"done. docs={num_docs:,} skipped={num_skipped:,} total_tokens={writer.total_tokens:,}")
    print(f"shards={len(writer.shards)} dtype={meta['dtype']} out={args.out_dir}")
    print(f"throughput docs/s={num_docs / elapsed:.1f} tok/s={writer.total_tokens / elapsed:.0f}")


if __name__ == "__main__":
    main()
