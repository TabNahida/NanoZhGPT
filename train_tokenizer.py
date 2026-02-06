import os
import glob
import gzip
import orjson
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC

# ---------- 配置 ----------
DATA_GLOB = r"D:\Data\AI\LLM\Text\allenai_c4\multilingual\c4-zh.tfrecord-*-of-01024.json.gz"   # 改成你的路径
TEXT_FIELD = "text"               # 改成你的字段名
VOCAB_SIZE = 50000                # 多语+大语料常用 50k/80k/100k
MIN_FREQ = 2
BATCH_SIZE = 2000                 # 2000~10000 视内存/CPU 调
MAX_CHARS_PER_DOC = 20000         # 可选：裁剪超长文本，避免少量巨文拖慢
OUT_JSON = "bytelevel_bpe_tokenizer.json"

SPECIAL_TOKENS = ["<s>", "</s>", "<pad>", "<unk>", "<mask>"]

# Rust 线程数（可选）：不设也行；想控线程就设
os.environ.setdefault("RAYON_NUM_THREADS", str(os.cpu_count() or 8))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

def iter_text_batches(paths):
    batch = []
    for p in paths:
        print("Processing File:", p)
        with gzip.open(p, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = orjson.loads(line)
                except orjson.JSONDecodeError:
                    continue

                txt = obj.get(TEXT_FIELD, "")
                if not isinstance(txt, str):
                    continue
                txt = txt.strip()
                if not txt:
                    continue

                if MAX_CHARS_PER_DOC and len(txt) > MAX_CHARS_PER_DOC:
                    txt = txt[:MAX_CHARS_PER_DOC]

                batch.append(txt)
                if len(batch) >= BATCH_SIZE:
                    yield batch
                    batch = []

    if batch:
        yield batch

def main():
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        raise FileNotFoundError(f"No files matched: {DATA_GLOB}")
    print(f"Found {len(paths)} files.")
    paths = paths[:50]  # 测试时可只用前50个文件，正式训练时删掉这行
    print(f"Using {len(paths)} files.")

    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQ,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),  # Byte-level 关键：覆盖所有字节
        show_progress=True,
    )

    tokenizer.train_from_iterator(iter_text_batches(paths), trainer=trainer)
    tokenizer.save(OUT_JSON)
    print("Saved:", OUT_JSON)

if __name__ == "__main__":
    main()
