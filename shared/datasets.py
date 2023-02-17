import  torch
from torch.utils.data import Dataset, Subset


def causal(tokens, block_size=256, test_size=1500):
    # TODO: split could/should be ratio
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

    train = CausalDataset(tokens[:-test_size], block_size)
    test = CausalDataset(tokens[-test_size:], block_size)
    return train, test


# TODO:
# - fill in the middle
# - denoise, fix casing, fix punctuation, etc.
