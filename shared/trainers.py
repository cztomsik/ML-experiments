import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler
import lightning as pl


class CausalTrainer(pl.Trainer):
    def __init__(self, model, tokenizer, device="cpu", **kwargs):
        super().__init__(max_epochs=0, accelerator="gpu" if device ==
                         "cuda" else device, **kwargs)
        self.wrapper = CausalWrapper(model, tokenizer)

    def train(self, input, test_size=1_500, batch_size=1, epochs=1, epoch_size=6_000):
        block_size = self.wrapper.model.block_size
        tokens = self.wrapper.tokenizer.encode(input).ids
        train = CausalDataset(tokens[:-test_size], block_size)
        val = CausalDataset(tokens[-test_size:], block_size)
        train_loader = DataLoader(train, batch_size=batch_size, num_workers=0,
                                  sampler=RandomSampler(train, False, epoch_size))
        val_loader = DataLoader(val, batch_size=batch_size, num_workers=0)

        # https://github.com/Lightning-AI/lightning/issues/11425
        self.fit_loop.max_epochs += epochs

        self.fit(self.wrapper, train_loader, val_loader)


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


class CausalWrapper(pl.LightningModule):
    def __init__(self, model, tokenizer, lr=0.007):
        super().__init__()
        self.lr = lr
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, x, y=None):
        logits = self.model(x)
        return logits if y is None else F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)

    def training_step(self, batch, batch_idx):
        return self(*batch)

    def validation_step(self, batch, batch_idx):
        loss = self(*batch)
        self.log("test_loss", loss, prog_bar=True,
                 on_step=False, on_epoch=True)
        return loss

    def validation_epoch_end(self, outs):
        print(self.generate("And now", 64))

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ExponentialLR(
            optim, gamma=0.95, last_epoch=-1)
        return [optim], [sched]

    # inspired by https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
    @torch.no_grad()
    def generate(self, str, max_new_tokens, top_k=10):
        ids = torch.tensor(self.tokenizer.encode(str).ids,
                           dtype=torch.long).unsqueeze(0).to(self.device)
        for _ in range(max_new_tokens):
            #out = self(ids[:, -block_size:])
            out = self(ids)
            logits = out[:, -1, :]
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("Inf")
            step_res = torch.multinomial(
                F.softmax(logits, dim=-1), num_samples=1)
            # auto-regression
            ids = torch.cat((ids, step_res), dim=1)
        return self.tokenizer.decode(ids[0].tolist())
