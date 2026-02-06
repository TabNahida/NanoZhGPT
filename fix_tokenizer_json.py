from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

tok = Tokenizer(BPE.from_file("vocab.json", "merges.txt", unk_token="<unk>"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
tok.save("tokenizer.json")
