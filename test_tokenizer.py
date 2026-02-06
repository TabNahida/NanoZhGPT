# pip install tokenizers
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel

tok = Tokenizer.from_file("tokenizer.json")
text = "你好 world! 123"

enc = tok.encode(text)
tok.decoder = ByteLevel()
print(enc.tokens)
print(enc.ids)
print(tok.decode(enc.ids))

print("vocab_size:", tok.get_vocab_size())
print("unk_id:", tok.token_to_id("<unk>"))
