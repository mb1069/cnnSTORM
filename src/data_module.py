from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import pytorch_lightning as pl
import numpy as np
import os

from src.data.data_processing import process_STORM_datadir

dtype = np.float32

data_dir = os.path.join(os.path.dirname(__file__), 'raw_data')


class CustomDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]

        self.x = x.astype(dtype)
        self.y = y.astype(dtype)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __len__(self):
        return self.x.shape[0]


def load_datasets(test_size):
    X, y = process_STORM_datadir(data_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=1 - test_size)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)
    return train_dataset, val_dataset


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, test_size=0.2):
        super().__init__()
        self.batch_size = batch_size
        self.test_size = test_size

    def setup(self, stage):
        # called on every GPU
        self.train, self.val = load_datasets(self.test_size)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=64)
