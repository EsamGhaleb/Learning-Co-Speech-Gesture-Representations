import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl


class BinaryProbingModule(pl.LightningModule):
    def __init__(self, probing_model, lr=0.001, pos_weight=1):
        super().__init__()
        self.probing_model = probing_model
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
        self.lr = lr

        self.metrics = torch.nn.ModuleDict({
            "accuracy": torchmetrics.classification.Accuracy(task="binary"),
            "recall": torchmetrics.classification.Recall(task="binary"),
            "precision": torchmetrics.classification.Precision(task="binary"),
            "f1": torchmetrics.classification.F1Score(task="binary"),
            "roc-auc": torchmetrics.AUROC(task="binary")
        })

        self.save_hyperparameters()

    def forward(self, x):
        return self.probing_model(x)
    
    def prepare_batch(self, batch):
        features, label = batch
        return features.float(), label
    
    def training_step(self, batch, batch_idx):
        features, label = self.prepare_batch(batch)
        output = self(features).squeeze()
        loss = self.criterion(output, label.float())
        self.log("train/loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, batch_idx, prefix="val")

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx, prefix="test")

    def _eval_step(self, batch, batch_idx, prefix="test"):
        features, label = self.prepare_batch(batch)
        output = self(features).squeeze()
        loss = self.criterion(output, label.float())
        pred = (torch.nn.functional.sigmoid(output) > 0.5).int()
        self.log(f"{prefix}/loss", loss)
        for metric in self.metrics:
            self.log(
                f"{prefix}/{metric}", 
                self.metrics[metric](pred if metric != "roc-auc" else output, label)
            )
        return {
            "pred": list(pred.cpu().detach().numpy()),
            "label": list(label.cpu().detach().numpy())
        }
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class LinearProbe(nn.Module):
    def __init__(self, concat_feature_size):
        super().__init__()
        self.linear = nn.Linear(concat_feature_size, 1)

    def forward(self, x):
        return self.linear(x)
    

class NonLinearProbe(nn.Module):
    def __init__(self, concat_feature_size):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(concat_feature_size, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64)
        )
        self.block2 = nn.Sequential(
            nn.Linear(64, 8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8)
        )
        self.out = nn.Linear(8, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.out(x)
