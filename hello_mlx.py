# /// script
# dependencies = [
#   "mlx==0.20.0",
# ]
# ///
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

inputs = mx.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=mx.float16)
targets = mx.array([[0], [1], [1], [0]], dtype=mx.float16)

model = nn.Sequential(
    nn.Linear(inputs.shape[1], 2),
    nn.Tanh(),
    nn.Linear(2, 1),
    nn.Tanh()
)

def loss(model, x, y):
  return nn.losses.mse_loss(model(x), y)

optimizer = optim.SGD(learning_rate=0.2)
step = nn.value_and_grad(model, loss)

for i in range(10_000):
  loss, grads = step(model, inputs, targets)

  if i % 1_000 == 0:
   print(f"iter: {i}, loss: {loss:.4f}")
  
  optimizer.update(model, grads)

# should be 0, 1, 1, 0
print(model(inputs))
