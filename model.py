from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src import ConvLSTM, SimVP_Model, Conv3D, SAConvLSTM 

class STMLightning(pl.LightningModule):
    def __init__(self, model_args, data_args):
        super(STMLightning, self).__init__()
        
        self.save_hyperparameters()
        self.model_args = model_args
        self.data_args = data_args

        if 'ConvLSTM' == self.model_args['model_name']:

            self.model = ConvLSTM(
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


        self.loss1 = nn.MSELoss()
        self.loss2 = nn.L1Loss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[..., 
                      self.data_args['padding'][0]:self.data_args['padding'][0]+self.data_args['size'][0], 
                      self.data_args['padding'][1]:self.data_args['padding'][1]+self.data_args['size'][1]]
        
        loss1 = self.loss1(y_hat, y)
        loss2 = self.loss2(y_hat, y)
        total_loss = (loss1 + loss2) / 2

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = y_hat[..., 
                      self.data_args['padding'][0]:self.data_args['padding'][0]+self.data_args['size'][0], 
                      self.data_args['padding'][1]:self.data_args['padding'][1]+self.data_args['size'][1]]

        loss1 = self.loss1(y_hat, y)
        loss2 = self.loss2(y_hat, y)

        total_loss = (loss1  + loss2) / 2

        self.log('val_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.model_args['learning_rate'],
            # weight_decay=self.model_args['weight_decay']
            )

        # Define the scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.model_args['epochs'], eta_min=0)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer, 
        #     max_lr=self.model_args['learning_rate']*10,
        #     steps_per_epoch=self.data_args['steps_per_epoch'], 
        #     epochs=self.model_args['epochs']
        #     )
        
        # Return both optimizer and scheduler
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Metric to monitor
                "interval": "step",    # Frequency of the scheduler
                "frequency": 1,         # How often to check the monitor
                "strict": True,         # Enforce the availability of the monitored metric
            },
        }