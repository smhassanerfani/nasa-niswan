from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src import ConvLSTM, SimVP_Model, Conv3D, SAConvLSTM 
from utils import RMSELoss

class STMLightning(pl.LightningModule):
    def __init__(self, model_args, data_args):
        super(STMLightning, self).__init__()
        
        self.save_hyperparameters()
        self.model_args = model_args
        self.data_args = data_args
        image_size=(self.data_args['size'][0] + 2*self.data_args['padding'][0], 
                    self.data_args['size'][1] + 2*self.data_args['padding'][1]
            )   

        if 'ConvLSTM' == self.model_args['model_name']:

            self.model = ConvLSTM(
                image_size=image_size,
                in_channels=self.model_args['in_channels'], 
                out_channels=self.model_args['out_channels'], 
                hidden_channels=self.model_args['encoder_channels'],
                kernel_size=self.model_args['kernel_size']
                )

        elif 'SimVP' == self.model_args['model_name']:
            self.model = SimVP_Model(
                in_shape=(48, 5, 160, 160),
                hid_S=self.model_args['hid_S'], 
                hid_T=self.model_args['hid_T'], 
                N_S=self.model_args['N_S'], 
                N_T=self.model_args['N_T'], 
                model_type='gSTA',
                mlp_ratio=8.0,
                drop=0.0, 
                drop_path=0.2,
                spatio_kernel_enc=self.model_args['spatio_kernel_enc'], 
                spatio_kernel_dec=self.model_args['spatio_kernel_dec'], 
                act_inplace=True
                )
        
        elif 'Conv3D' == self.model_args['model_name']:

            self.model = Conv3D(
                in_channels=self.model_args['in_channels'], 
                out_channels=self.model_args['out_channels'], 
                kernel_size=self.model_args['kernel_size']
                )
        
        elif 'SAConvLSTM' == self.model_args['model_name']:
            self.model = SAConvLSTM(
                image_size=self.data_args['size'][0] + 2*self.data_args['padding'][0], 
                input_channels=self.model_args['in_channels'], 
                output_channels=self.model_args['out_channels'],
                hidden_channels=self.model_args['encoder_channels'], 
                kernel_size=self.model_args['kernel_size']
            )


        # Initialize loss functions based on config
        loss_dict = {
            'MSE': nn.MSELoss(),
            'MAE': nn.L1Loss(),
            'RMSE': RMSELoss(dataset=data_args['data_name'])
        }
        self.losses = [loss_dict[loss] for loss in self.model_args['loss']]

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[..., 
                      self.data_args['padding'][0]:self.data_args['padding'][0]+self.data_args['size'][0], 
                      self.data_args['padding'][1]:self.data_args['padding'][1]+self.data_args['size'][1]]
        
        total_loss = sum(loss(y_hat, y) for loss in self.losses) / len(self.losses)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[..., 
                      self.data_args['padding'][0]:self.data_args['padding'][0]+self.data_args['size'][0], 
                      self.data_args['padding'][1]:self.data_args['padding'][1]+self.data_args['size'][1]]

        total_loss = sum(loss(y_hat, y) for loss in self.losses) / len(self.losses)

        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def configure_optimizers(self):

        optimizer_params = {
            'params': self.model.parameters(),
            'lr': self.model_args['learning_rate'],
            'weight_decay': self.model_args['weight_decay']
        }
        
        if self.model_args['optimizer'] == 'Adam':
            optimizer = torch.optim.Adam(**optimizer_params)
        
        elif self.model_args['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(**optimizer_params)

        # Define the scheduler
        if self.model_args['scheduler'] == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 
                **self.model_args['scheduler_params']
            )
            interval = 'epoch'
        elif self.model_args['scheduler'] == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                **self.model_args['scheduler_params']
            )
            interval = 'epoch'
        elif self.model_args['scheduler'] == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                **self.model_args['scheduler_params']
            )
            interval = 'epoch'
        elif self.model_args['scheduler'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                **self.model_args['scheduler_params']
            )
            interval = 'step'
        else:
            scheduler = None
            interval = 'epoch'

        # Return both optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to monitor
                "interval": interval,   # Frequency of the scheduler
                "frequency": 1,         # How often to check the monitor
                "strict": True,         # Enforce the availability of the monitored metric
            },
        }
