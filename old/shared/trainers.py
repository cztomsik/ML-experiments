import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
import lightning as pl


class BaseLMTrainer(pl.Trainer):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(max_epochs=0, accelerator="gpu" if device ==
                         "cuda" else device, **kwargs)

    def train(self, input, test_size=1_500, batch_size=1, epochs=1, epoch_size=6_000):
        tokens = self.wrapper.tokenizer.encode(input).ids
        train = self.wrapper.create_dataset(tokens[:-test_size])
        val = self.wrapper.create_dataset(tokens[-test_size:])
        train_loader = DataLoader(train, batch_size=batch_size, num_workers=0,
                                  sampler=RandomSampler(train, False, epoch_size))
        val_loader = DataLoader(val, batch_size=batch_size, num_workers=0)

        # https://github.com/Lightning-AI/lightning/issues/11425
        self.fit_loop.max_epochs += epochs

        self.fit(self.wrapper, train_loader, val_loader)


class CausalTrainer(BaseLMTrainer):
    def __init__(self, model, tokenizer, lr=0.007, val_prompt="And now", **kwargs):
        super().__init__(**kwargs)
        self.wrapper = CausalWrapper(model, tokenizer, lr, val_prompt)


class MLMTrainer(BaseLMTrainer):
    def __init__(self, model, tokenizer, lr=0.001, **kwargs):
        super().__init__(**kwargs)
        self.wrapper = MLMWrapper(model, tokenizer, lr)


class BaseLMWrapper(pl.LightningModule):
    def __init__(self, model, tokenizer, lr):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.lr = lr

    def forward(self, x, y=None):
        logits = self.model(x)
        return logits if y is None else F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    def create_dataset(self, tokens):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self(*batch)

    def validation_step(self, batch, batch_idx):
        loss = self(*batch)
        self.log("test_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(
            optim, gamma=0.95, last_epoch=-1)
        return [optim], [sched]


class CausalWrapper(BaseLMWrapper):
    def __init__(self, model, tokenizer, lr, val_prompt, **kwargs):
        super().__init__(model, tokenizer, lr, **kwargs)
        self.val_prompt = val_prompt

    def create_dataset(self, tokens):
        return CausalDataset(tokens, self.model.block_size)

    def validation_epoch_end(self, outs):
        print(self.generate(self.val_prompt, 64))

    # inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    @torch.no_grad()
    def generate(self, str, max_new_tokens, top_k=10):
        ids = torch.tensor(self.tokenizer.encode(str).ids,
                           dtype=torch.long, device=self.device)[None, :]
        for _ in range(max_new_tokens):
            out = self(ids[:, -self.model.block_size:])
            logits = out[:, -1, :]
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
            step_res = torch.multinomial(
                F.softmax(logits, dim=-1), num_samples=1)
            # auto-regression
            ids = torch.cat((ids, step_res), dim=1)
        return self.tokenizer.decode(ids[0].tolist())


class MLMWrapper(BaseLMWrapper):
    def __init__(self, model, tokenizer, lr, **kwargs):
        super().__init__(model, tokenizer, lr, **kwargs)

    def create_dataset(self, tokens):
        return MLMDataset(tokens, self.model.block_size, self.tokenizer.mask_token_id)

    def validation_epoch_end(self, outs):
        s = f"Hello, my n{self.tokenizer.mask_token}me is"
        print(s)
        print(self.fill(s))

    @torch.no_grad()
    def fill(self, str, top_k=3):
        ids = torch.tensor(self.tokenizer.encode(str).ids,
                           dtype=torch.long, device=self.device)
        mask = ids == self.tokenizer.mask_token_id
        logits = self(ids[None, :])[0, :ids.size(0)][mask]
        probs, ids = torch.topk(logits.softmax(-1), top_k)
        # return tokens with their probabilities
        return [list(zip(self.tokenizer.decode(ids[i].tolist()), probs[i].tolist())) for i in range(len(probs))]


class CausalDataset(Dataset):
    def __init__(self, data, block_size):
        super().__init__()
        self.data = torch.tensor(data)
        self.block_size = block_size

    def __len__(self):
        return self.data.size(0) - self.block_size - 1

    def __getitem__(self, i):
        end = i + self.block_size
        return self.data[i:end], self.data[i + 1:end + 1]


class MLMDataset(Dataset):
    def __init__(self, data, block_size, mask_token_id):
        super().__init__()
        self.data = torch.tensor(data)
        self.block_size = block_size
        self.mask_token_id = mask_token_id

    def __len__(self):
        return self.data.size(0) - self.block_size

    def __getitem__(self, i):
        end = i + self.block_size
        x = self.data[i:end].clone()
        y = x.clone()
        # randomly replace 15% of tokens with mask token in x and with -1 in y
        mask = torch.rand(x.size()) < 0.15
        x[mask] = self.mask_token_id
        y[~mask] = -1
        return x, y
