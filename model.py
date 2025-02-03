from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from nn import ConvLSTM, Conv3D


class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


class STMLightning(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, learning_rate: Optional[float] = 5.25E-05):
        super(STMLightning, self).__init__()

        self.model = Conv3D(in_channels, out_channels, kernel_size)
        self.learning_rate = learning_rate

        self.loss1 = nn.MSELoss()
        self.loss2 = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[..., 5:5+90, 5:5+144]
        
        loss1 = self.loss1(y_hat, y)
        loss2 = self.loss2(y_hat, y)
        total_loss = (loss1 + loss2) / 2

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[..., 5:5+90, 5:5+144]

        loss1 = self.loss1(y_hat, y)
        loss2 = self.loss2(y_hat, y)

        total_loss = (loss1 + loss2) / 2

        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)

        loss1 = self.loss1(y_hat, y)
        loss2 = self.loss2(y_hat, y)

        total_loss = loss1 + loss2

        self.log("test_loss", total_loss, prog_bar=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        
        # Define the scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        # Return both optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to monitor
                "interval": "epoch",    # Frequency of the scheduler
                "frequency": 1,         # How often to check the monitor
                "strict": True,         # Enforce the availability of the monitored metric
            },
        }