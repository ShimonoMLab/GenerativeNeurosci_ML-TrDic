from torch.utils.data import Dataset

class SplitDataset(Dataset):

    def __init__(self, data, *, split_size):
        super().__init__()
        self.data = data.split(split_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
