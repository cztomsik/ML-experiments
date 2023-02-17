# TODO: check if this actually works, this is just what I recall from before
def bpe(text, special_tokens=["[UNK]"], unk_token="[UNK]"):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(BPE(unk_token=unk_token))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        [text], trainer=BpeTrainer(special_tokens=special_tokens))
    return tokenizer


def tiktoken(encoding="gpt2"):
    import tiktoken
    return tiktoken.get_encoding(encoding)


def unique_chars(text):
    class UniqueCharTokenizer:
        def __init__(self, text):
            vocab = sorted(set(text))
            self.vocab = vocab
            self.vocab_size = len(vocab)
            self.stoi = {ch: i for i, ch in enumerate(vocab)}
            self.itos = {i: ch for i, ch in enumerate(vocab)}

        def encode(self, text):
            return [self.stoi[ch] for ch in text]

        def decode(self, tokens):
            return "".join([self.itos[i] for i in tokens])

    return UniqueCharTokenizer(text)
