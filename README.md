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
