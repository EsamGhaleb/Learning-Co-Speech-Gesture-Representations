import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader


class ProbingDataModule(pl.LightningDataModule):
    def __init__(self, concat_features, labels, split_proportion=[0.6, 0.1, 0.3], batch_size=32, seed=42):
        super().__init__()
        self.concat_features = concat_features
        self.labels = labels
        self.batch_size = batch_size
        self.seed = seed
        self.split_proportion = split_proportion

    def setup(self, stage: str):
        self.dataset = ProbingDataset(self.concat_features, self.labels)
        self.train, self.val, self.test = torch.utils.data.random_split(
            self.dataset,
            self.split_proportion,
            generator=torch.Generator().manual_seed(self.seed)
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
    

class ProbingDataset(Dataset):
    def __init__(self, concat_features, labels):
        super().__init__()
        self.concat_features = concat_features
        self.labels = labels

    def __len__(self):
        return len(self.concat_features)
    
    def __getitem__(self, idx):
        return self.concat_features[idx], self.labels[idx]