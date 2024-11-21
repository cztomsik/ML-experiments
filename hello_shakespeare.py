# /// script
# dependencies = [
#   "numpy==1.26.4",
#   "mlx==0.20.0",
# ]
# ///
import math
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# tokenize input
with open("data/shakespeare.txt", "r") as f:
    data = f.read()
    vocab = {c: i for i, c in enumerate(sorted(list(set(data))))}
    tokens = mx.array([vocab[c] for c in data], dtype=mx.int32)

class Llama(nn.Module):
    def __init__(self, vocab_size, n_layer, n_head, n_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_dim)
        self.layers = [Layer(n_head, n_dim) for _ in range(n_layer)]
        self.norm = nn.RMSNorm(n_dim)
        self.lm_head = nn.Linear(n_dim, vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1], self.embedding.weight.dtype)
        x = self.embedding(x)
        for l in self.layers:
            x = l(x, mask)
        x = self.norm(x)
        return self.lm_head(x)

class Layer(nn.Module):
    def __init__(self, n_head, n_dim):
        super().__init__()
        self.attention = Attention(n_head, n_dim)
        self.norm1 = nn.RMSNorm(n_dim)
        self.norm2 = nn.RMSNorm(n_dim)
        self.lin1 = nn.Linear(n_dim, 4 * n_dim, bias=False)
        self.lin2 = nn.Linear(n_dim, 4 * n_dim, bias=False)
        self.lin3 = nn.Linear(4 * n_dim, n_dim, bias=False)

    def __call__(self, x, mask):
        x = x + self.attention(self.norm1(x), mask)
        y = self.norm2(x)
        a = self.lin1(y)
        b = self.lin2(y)
        y = a * mx.sigmoid(a) * b
        return x + self.lin3(y)

class Attention(nn.Module):
    def __init__(self, n_head, n_dim):
        super().__init__()
        self.n_head = n_head

        self.rope = nn.RoPE(n_dim // n_head, traditional=True)
        self.q = nn.Linear(n_dim, n_dim, bias=False)
        self.k = nn.Linear(n_dim, n_dim, bias=False)
        self.v = nn.Linear(n_dim, n_dim, bias=False)
        self.proj = nn.Linear(n_dim, n_dim, bias=False)

    def __call__(self, x, mask):
        q, k, v = self.q(x), self.k(x), self.v(x)
        B, L, _ = q.shape

        q = q.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)

        q = self.rope(q)
        k = self.rope(k)

        scale = math.sqrt(1 / q.shape[-1])
        w = (q * scale) @ k.transpose(0, 1, 3, 2)
        w = mx.softmax(w + mask, axis=-1)
        return self.proj((w @ v).transpose(0, 2, 1, 3).reshape(B, L, -1))

model = Llama(vocab_size=len(vocab), n_layer=4, n_head=4, n_dim=128)
mx.eval(model.parameters())

def loss(model, x, y, reduce=True):
    losses = nn.losses.cross_entropy(model(x), y)
    return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

def strides(n_ctx, tokens):
    return np.lib.stride_tricks.as_strided(
        tokens,
        shape=(tokens.size - (n_ctx + 2), n_ctx + 1),
        strides=(tokens.itemsize, tokens.itemsize),
    )

def generate(model, str, n):
    x = mx.array([vocab[c] for c in str], dtype=mx.int32)[None, :]
    for _ in range(n):
        y = model(x)
        x = mx.concatenate([x, mx.random.categorical(y[:, -1])[None, :]], axis=1)
    return "".join([list(vocab.keys())[i] for i in x[0].tolist()])

optimizer = optim.AdamW(learning_rate=0.0001)
step = nn.value_and_grad(model, loss)

state = [model.state, optimizer.state]
samples = strides(256, tokens)

for epoch in range(25):
    print(generate(model, "And now", 100))
    
    for i in range(1_000):
        ids = np.random.randint(0, samples.shape[0], 32)
        buf = mx.array(samples[ids], dtype=mx.int32)
        x = buf[:, :-1]
        y = buf[:, 1:]

        loss, grads = step(model, x, y)
        optimizer.update(model, grads)
        mx.eval(state)

        if i % 100 == 0:
            print(f"epoch: {epoch}, iter: {i}, loss: {loss}")

print(generate(model, "O God, O God!", 512))