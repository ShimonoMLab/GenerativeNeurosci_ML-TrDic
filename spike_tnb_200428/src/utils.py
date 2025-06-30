import numpy as np
import torch


def sample(data):
    data = data.to('cpu').numpy()

    values = np.random.random(data.shape)
    samples = (data >= values).astype(np.float32)
    return torch.from_numpy(samples)
