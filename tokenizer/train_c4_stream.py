#!/usr/bin/env python
import argparse
import gc
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    from datasets import get_dataset_config_names, load_dataset
except ImportError:
    get_dataset_config_names = None
    load_dataset = None

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def format_bytes(num: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, num))
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f}{unit}"
        value /= 1024.0
    return f"{value:.2f}TB"


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            for entry in current.iterdir():
                try:
                    if entry.is_symlink():
                        continue
                    if entry.is_dir():
                        stack.append(entry)
                    elif entry.is_file():
                        total += entry.stat().st_size
                except OSError:
                    continue
        except OSError:
            continue
    return total


@dataclass
class BudgetSnapshot:
    rss_bytes: int
    peak_rss_bytes: int
    disk_bytes: int


class BudgetGuard:
    def __init__(
        self,
        memory_budget_gb: float,
        disk_budget_gb: float,
        tracked_dirs: Iterable[Path],
    ) -> None:
        self.memory_budget_bytes = int(memory_budget_gb * (1024**3)) if memory_budget_gb > 0 else 0
        self.disk_budget_bytes = int(disk_budget_gb * (1024**3)) if disk_budget_gb > 0 else 0
        resolved: List[Path] = []
        for p in tracked_dirs:
            if p is None:
                continue
            try:
                resolved.append(p.resolve())
            except OSError:
                resolved.append(p)
        resolved = sorted({str(p): p for p in resolved}.values(), key=lambda x: len(str(x)))

        # Keep only non-overlapping roots to avoid double-counting nested dirs.
        roots: List[Path] = []
        for cand in resolved:
            inside_existing = False
            for root in roots:
                try:
                    if cand.is_relative_to(root):
                        inside_existing = True
                        break
                except AttributeError:
                    cand_s = str(cand)
                    root_s = str(root).rstrip("\\/") + os.sep
                    if cand_s == str(root) or cand_s.startswith(root_s):
                        inside_existing = True
                        break
            if not inside_existing:
                roots.append(cand)
        self.tracked_dirs = roots
        self.peak_rss_bytes = 0
        self.last_rss_bytes = 0
        self.last_disk_bytes = 0

        if self.memory_budget_bytes > 0:
            if psutil is None:
                raise ImportError("psutil is required for memory budget checks. Install with: pip install psutil")
            self._proc = psutil.Process(os.getpid())
        else:
            self._proc = None

    def check(self, tag: str) -> BudgetSnapshot:
        rss = 0
        if self._proc is not None:
            rss = int(self._proc.memory_info().rss)
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
            if self.memory_budget_bytes > 0 and rss > self.memory_budget_bytes:
                raise RuntimeError(
                    f"memory budget exceeded at {tag}: rss={format_bytes(rss)} "
                    f"> budget={format_bytes(self.memory_budget_bytes)}"
                )

        disk = sum(dir_size_bytes(p) for p in self.tracked_dirs)
        if self.disk_budget_bytes > 0 and disk > self.disk_budget_bytes:
            raise RuntimeError(
                f"disk budget exceeded at {tag}: usage={format_bytes(disk)} "
                f"> budget={format_bytes(self.disk_budget_bytes)}"
            )

        self.last_rss_bytes = rss
        self.last_disk_bytes = disk
        return BudgetSnapshot(rss_bytes=rss, peak_rss_bytes=self.peak_rss_bytes, disk_bytes=disk)


@dataclass
class StreamStats:
    language: str
    docs_scanned: int = 0
    docs_emitted: int = 0
    skipped_missing: int = 0
    skipped_short: int = 0
    truncated_docs: int = 0
    emitted_chars: int = 0
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def elapsed_sec(self) -> float:
        if self.finished_at <= self.started_at:
            return 0.0
        return self.finished_at - self.started_at


def resolve_languages(
    dataset_name: str, languages_arg: str, excludes_arg: str
) -> List[str]:
    if languages_arg.strip().lower() != "all":
        langs = parse_csv(languages_arg)
        if not langs:
            raise ValueError("no languages configured; pass --languages zh,en,... or --languages all")
        return langs

    if get_dataset_config_names is None:
        raise ImportError("datasets is required. Install with: pip install datasets")

    configs = list(get_dataset_config_names(dataset_name))
    excludes = set(parse_csv(excludes_arg))
    langs = [cfg for cfg in configs if cfg not in excludes]
    if not langs:
        raise ValueError("no dataset configs left after applying excludes")
    return sorted(langs)


def build_stream(
    dataset_name: str,
    language: str,
    split: str,
    cache_dir: Path,
    shuffle_buffer: int,
    seed: int,
):
    if load_dataset is None:
        raise ImportError("datasets is required. Install with: pip install datasets")
    ds = load_dataset(
        path=dataset_name,
        name=language,
        split=split,
        streaming=True,
        cache_dir=str(cache_dir),
    )
    if shuffle_buffer > 0:
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    return ds


def language_text_iterator(
    *,
    dataset_name: str,
    language: str,
    split: str,
    text_field: str,
    cache_dir: Path,
    shuffle_buffer: int,
    seed: int,
    min_chars: int,
    max_chars: int,
    max_docs: int,
    check_every: int,
    progress_every: int,
    phase: str,
    guard: BudgetGuard,
    stats: StreamStats,
) -> Iterator[str]:
    stats.started_at = time.time()
    stream = build_stream(dataset_name, language, split, cache_dir, shuffle_buffer, seed)

    for row in stream:
        stats.docs_scanned += 1
        text = row.get(text_field) if isinstance(row, dict) else None
        if not isinstance(text, str):
            stats.skipped_missing += 1
            continue
        if min_chars > 0 and len(text) < min_chars:
            stats.skipped_short += 1
            continue
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars]
            stats.truncated_docs += 1
        if not text:
            stats.skipped_short += 1
            continue

        stats.docs_emitted += 1
        stats.emitted_chars += len(text)
        if check_every > 0 and (stats.docs_emitted % check_every == 0):
            snap = guard.check(f"{phase}:{language}:{stats.docs_emitted}")
            if progress_every > 0 and (stats.docs_emitted % progress_every == 0):
                print(
                    f"[{phase}/{language}] docs={stats.docs_emitted:,} scanned={stats.docs_scanned:,} "
                    f"rss={format_bytes(snap.rss_bytes)} disk={format_bytes(snap.disk_bytes)}"
                )
        yield text
        if max_docs > 0 and stats.docs_emitted >= max_docs:
            break

    stats.finished_at = time.time()


def train_tokenizer(
    *,
    output_path: Path,
    vocab_size: int,
    min_frequency: int,
    unk_token: str,
    special_tokens: List[str],
    texts: Iterator[str],
) -> Tokenizer:
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()
    trainer_kwargs = {
        "vocab_size": int(vocab_size),
        "min_frequency": int(min_frequency),
        "special_tokens": special_tokens,
        "show_progress": True,
    }
    try:
        trainer_kwargs["initial_alphabet"] = ByteLevelPreTokenizer.alphabet()
    except Exception:
        pass
    trainer = BpeTrainer(**trainer_kwargs)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output_path))
    return tokenizer


def tokenizer_summary(tokenizer: Tokenizer) -> Dict[str, object]:
    return {
        "vocab_size": int(tokenizer.get_vocab_size()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream-train byte-level BPE tokenizers from Hugging Face datasets "
            "(per-language + one global tokenizer)."
        )
    )
    parser.add_argument("--dataset", type=str, default="allenai/c4")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--languages", type=str, required=True, help="comma-separated configs, or 'all'")
    parser.add_argument("--exclude-configs", type=str, default="", help="comma-separated config names to skip")
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument("--out-dir", type=Path, default=Path("tokenizer/c4_stream"))
    parser.add_argument("--cache-dir", type=Path, default=None, help="default: <out-dir>/_hf_cache")
    parser.add_argument("--keep-cache", action="store_true", help="keep HF cache after training")

    parser.add_argument("--vocab-size", type=int, default=50000, help="per-language tokenizer vocab size")
    parser.add_argument("--global-vocab-size", type=int, default=50000, help="global tokenizer vocab size")
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--unk-token", type=str, default="<unk>")
    parser.add_argument("--special-tokens", type=str, default="<s>,</s>,<pad>,<unk>,<mask>")

    parser.add_argument("--min-chars", type=int, default=1)
    parser.add_argument("--max-chars", type=int, default=20000)
    parser.add_argument("--max-docs-per-lang", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--max-docs-per-lang-global",
        type=int,
        default=-1,
        help="default: same as --max-docs-per-lang; 0 means no limit",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--memory-budget-gb", type=float, default=50.0)
    parser.add_argument("--disk-budget-gb", type=float, default=20.0)
    parser.add_argument("--budget-check-every", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=50000)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if load_dataset is None:
        raise ImportError("datasets is required. Install with: pip install datasets")
    if psutil is None and args.memory_budget_gb > 0:
        raise ImportError("psutil is required for memory budget checks. Install with: pip install psutil")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir is not None else (out_dir / "_hf_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_dir / "hub"))

    languages = resolve_languages(args.dataset, args.languages, args.exclude_configs)
    specials = parse_csv(args.special_tokens)
    if args.unk_token not in specials:
        specials.append(args.unk_token)

    guard = BudgetGuard(
        memory_budget_gb=args.memory_budget_gb,
        disk_budget_gb=args.disk_budget_gb,
        tracked_dirs=[out_dir, cache_dir],
    )
    guard.check("init")

    started = time.time()
    per_language_runs: List[Dict[str, object]] = []

    for idx, language in enumerate(languages):
        print(f"[train/lang] {language} ({idx + 1}/{len(languages)})")
        stats = StreamStats(language=language)
        text_iter = language_text_iterator(
            dataset_name=args.dataset,
            language=language,
            split=args.split,
            text_field=args.text_field,
            cache_dir=cache_dir,
            shuffle_buffer=args.shuffle_buffer,
            seed=args.seed + idx,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
            max_docs=args.max_docs_per_lang,
            check_every=args.budget_check_every,
            progress_every=args.progress_every,
            phase="lang",
            guard=guard,
            stats=stats,
        )
        output_path = out_dir / f"tokenizer_{language}.json"
        model = train_tokenizer(
            output_path=output_path,
            vocab_size=args.vocab_size,
            min_frequency=args.min_frequency,
            unk_token=args.unk_token,
            special_tokens=specials,
            texts=text_iter,
        )
        guard.check(f"after-save:{language}")
        if stats.finished_at <= stats.started_at:
            stats.finished_at = time.time()
        run_info = {
            "phase": "lang",
            "language": language,
            "output": str(output_path),
            "stats": asdict(stats),
            "tokenizer": tokenizer_summary(model),
        }
        per_language_runs.append(run_info)
        print(
            f"[done/lang] {language} docs={stats.docs_emitted:,} scanned={stats.docs_scanned:,} "
            f"vocab={run_info['tokenizer']['vocab_size']:,}"
        )
        del model
        gc.collect()

    global_limit = args.max_docs_per_lang if args.max_docs_per_lang_global < 0 else args.max_docs_per_lang_global
    global_stats_by_lang: List[StreamStats] = []

    def global_texts() -> Iterator[str]:
        for idx, language in enumerate(languages):
            stats = StreamStats(language=language)
            global_stats_by_lang.append(stats)
            yield from language_text_iterator(
                dataset_name=args.dataset,
                language=language,
                split=args.split,
                text_field=args.text_field,
                cache_dir=cache_dir,
                shuffle_buffer=args.shuffle_buffer,
                seed=args.seed + 10000 + idx,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                max_docs=global_limit,
                check_every=args.budget_check_every,
                progress_every=args.progress_every,
                phase="global",
                guard=guard,
                stats=stats,
            )

    print("[train/global] all languages")
    global_path = out_dir / "tokenizer_global.json"
    global_model = train_tokenizer(
        output_path=global_path,
        vocab_size=args.global_vocab_size,
        min_frequency=args.min_frequency,
        unk_token=args.unk_token,
        special_tokens=specials,
        texts=global_texts(),
    )
    guard.check("after-save:global")
    print(f"[done/global] vocab={global_model.get_vocab_size():,}")

    ended = time.time()
    final_snap = guard.check("final")
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "split": args.split,
        "text_field": args.text_field,
        "languages": languages,
        "special_tokens": specials,
        "unk_token": args.unk_token,
        "vocab_size_per_language": args.vocab_size,
        "vocab_size_global": args.global_vocab_size,
        "min_frequency": args.min_frequency,
        "max_chars": args.max_chars,
        "max_docs_per_lang": args.max_docs_per_lang,
        "max_docs_per_lang_global": global_limit,
        "shuffle_buffer": args.shuffle_buffer,
        "seed": args.seed,
        "budgets": {
            "memory_budget_gb": args.memory_budget_gb,
            "disk_budget_gb": args.disk_budget_gb,
            "peak_rss_bytes": final_snap.peak_rss_bytes,
            "final_disk_bytes": final_snap.disk_bytes,
        },
        "output_dir": str(out_dir),
        "cache_dir": str(cache_dir),
        "runs": per_language_runs,
        "global": {
            "output": str(global_path),
            "tokenizer": tokenizer_summary(global_model),
            "stats_by_language": [asdict(s) for s in global_stats_by_lang],
        },
        "elapsed_sec": ended - started,
    }
    meta_path = out_dir / "stream_train_meta.json"
    meta_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[meta] {meta_path}")
    print(
        f"[budget] peak_rss={format_bytes(final_snap.peak_rss_bytes)} "
        f"disk={format_bytes(final_snap.disk_bytes)}"
    )

    if not args.keep_cache and cache_dir.exists():
        print(f"[cleanup] removing cache: {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
