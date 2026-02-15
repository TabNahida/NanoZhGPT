# NanoZhGPT
A tiny Chinese GPT implementation

## C++ Byte-Level BPE Trainer
- Reads DATA_PATH from .env (json/json.gz glob).
- Build (requires zlib):
  - xmake f -m release
  - xmake
- Run:
  - xmake run byte_bpe_train --output tokenizer.json
- Chunk resume:
  - Outputs per-chunk files to `chunks/` for resume.
  - Use `--chunk-files N` to group files per chunk, `--no-resume` to force rebuild.

## LLM Pretrain (XPU)

### 1) Prepare token shards (json/json.gz -> binary)
```powershell
python train/prepare_shards.py `
  --env-file .env `
  --tokenizer tokenizer.json `
  --out-dir data/tokens `
  --text-field text `
  --max-tokens-per-shard 50000000 `
  --encode-batch-size 256
```

Notes:
- Reads `DATA_PATH` from `.env` by default.
- Supports `*.json` and `*.json.gz` globs.
- Writes `data/tokens/meta.json` and `data/tokens/train_*.bin`.

### 2) Train ~100M GPT on XPU
```powershell
python train/train_xpu.py `
  --data-dir data/tokens `
  --out-dir checkpoints/gpt100m `
  --device xpu `
  --dtype bf16 `
  --gradient-checkpointing `
  --oom-auto-reduce `
  --min-batch-size 1 `
  --seq-len 2048 `
  --n-layer 14 `
  --n-head 10 `
  --n-embd 640 `
  --batch-size 2 `
  --grad-accum 16 `
  --prefetch-batches 32 `
  --shards-per-batch 1 `
  --ram-cache-gb 8 `
  --max-iters 20000
```

This default model is around 100M params (with tied embeddings).
On Windows, `--compile` may require extra compiler/runtime setup. The script will auto-fallback to eager if compile fails.
For IO-heavy runs, keep `--shards-per-batch 1` (better locality), increase `--prefetch-batches`, and set `--ram-cache-gb` according to free memory.
Training logs include progress (`iter/max_iters`, `%`, `eta`, `tokens`) and memory (`alloc/reserved/peak/total` on XPU/CUDA).
For tighter memory, use `--gradient-checkpointing`, reduce `--batch-size`, or lower `--seq-len`. With `--oom-auto-reduce`, micro-batch will be halved automatically after OOM (single-device mode).

### 3) Multi-XPU (DDP)
```powershell
torchrun --standalone --nproc_per_node=2 train/train_xpu.py `
  --data-dir data/tokens `
  --out-dir checkpoints/gpt100m_ddp `
  --device xpu `
  --dtype bf16
```
