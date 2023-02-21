# TODO: check if this actually works, this is just what I recall from before
def bpe(text, mask_token=None, vocab_size=None):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Whitespace
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train_from_iterator(
        [text], trainer=BpeTrainer(special_tokens=[mask_token] if mask_token else [], vocab_size=vocab_size))
    setattr(tokenizer, "mask_token", mask_token)
    setattr(tokenizer, "mask_token_id", tokenizer.token_to_id(mask_token))
    return tokenizer


# def tiktoken(encoding="gpt2"):
#     import tiktoken
#     return tiktoken.get_encoding(encoding)


def unique_chars(text, mask_token=None):
    class Result:
        def __init__(self, ids):
            self.ids = ids

    class UniqueCharTokenizer:
        def __init__(self, text):
            vocab = sorted(set(text))

            if mask_token is not None:
                vocab = vocab + [mask_token]
                self.mask_token = mask_token
                self.mask_token_id = len(vocab) - 1

            self.vocab = vocab
            self.stoi = {ch: i for i, ch in enumerate(vocab)}
            self.itos = {i: ch for i, ch in enumerate(vocab)}

        def get_vocab_size(self):
            return len(self.vocab)

        def encode(self, text):
            ids = [self.stoi[ch] for ch in text]
            return Result(ids)

        def decode(self, tokens):
            return "".join([self.itos[i] for i in tokens])

    return UniqueCharTokenizer(text)
