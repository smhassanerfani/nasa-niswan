from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src import ConvLSTM, SimVP_Model, Conv3D, SAConvLSTM, FMEncoder, UNet 
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

        # Progressive training parameters
        self.progressive_training = model_args.get('progressive_training', True)
        self.stage1_epochs = model_args.get('stage1_epochs', 0)  # Teacher forcing epochs
        self.initial_rollout_steps = model_args.get('initial_rollout_steps', 3)
        self.max_rollout_steps = model_args.get('max_rollout_steps', 8)
        self.num_epochs_interval = model_args.get('num_epochs_interval', 10)

        if 'UNet' == self.model_args['model_name']:

            self.model = UNet(
                in_channels=self.model_args['in_channels'], 
                out_channels=self.model_args['out_channels'], 
            )

        if 'ConvLSTM' == self.model_args['model_name']:

            class Model(nn.Module):
                def __init__(self, image_size, model_args, data_args):
                    super(Model, self).__init__()
                    self.encoder = FMEncoder(forcing_channels=data_args['levels']*8, dropout=0.0)
                    self.decoder = ConvLSTM(
                        image_size=image_size,
                        in_channels=model_args['in_channels'], 
                        out_channels=model_args['out_channels'], 
                        hidden_channels=model_args['encoder_channels'],
                        kernel_size=model_args['kernel_size']
                    )

                def forward(self, e, f):
                    x = self.encoder(e, f)
                    y = self.decoder(x)
                    return y
            
            self.model = Model(
                image_size=image_size, 
                model_args=self.model_args,
                data_args=self.data_args
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

    def _apply_padding(self, data):
        """
        Apply cyclic padding along the longitude (W dimension)
        and reflective padding along the latitude (H dimension).
        """
        pad_lat = self.data_args['padding'][0]  # top and bottom
        pad_lon = self.data_args['padding'][1]  # left and right

        H = data.shape[-2]  # latitude
        W = data.shape[-1]  # longitude

        # --- Cyclic padding along longitude (axis=4) ---
        if (pad_lon <= W):
            data = torch.concatenate([data[..., -pad_lon:], data, data[..., :pad_lon]], axis=-1)
        else:
            raise AttributeError(
                f"Requested longitude padding ({pad_lon}) exceeds image width ({W})."
            )

        # --- Reflective padding along latitude (axis=3) ---
        pad_top = pad_lat
        pad_bottom = pad_lat

        if (pad_top <= H) and (pad_bottom <= H):
            top = torch.flip(data[..., 1:pad_top+1, :], dims=[-2])
            bottom = torch.flip(data[..., -pad_bottom-1:-1, :], dims=[-2])
            data = torch.concatenate((top, data, bottom), axis=-2)
        else:
            raise AttributeError(
                f"Requested latitude padding ({pad_top}, {pad_bottom}) exceeds image height ({H})."
            )

        return data

    def get_current_rollout_steps(self):
        """Calculate current rollout steps based on epoch and training stage"""
        if not self.progressive_training:
            return 1  # Default single-step prediction
            
        if self.current_epoch < self.stage1_epochs:
            return 1  # Stage 1: Teacher forcing
        
        else:
            # Stage 2: Progressive rollout - increases based on self.num_epochs_interval
            steps_since_stage2 = self.current_epoch - self.stage1_epochs
            rollout_steps = min(self.initial_rollout_steps + (steps_since_stage2 // self.num_epochs_interval), self.max_rollout_steps)
            return rollout_steps

    def autoregressive_forward(self, e, f, num_steps):
        """Perform autoregressive rollout for num_steps"""
        predictions = []
        
        # Initial prediction using first timestep
        current_e = e[:, 0]  # [batch, channels, 1, H, W]
        current_f = f[:, 0]  # [batch, channels, levels, H, W]

        for step in range(num_steps):
            # Predict next timestep
            pred = self.forward(current_e, current_f)
            predictions.append(pred)
            # For next step, use ground truth inputs if available
            # e.shape: [batch, time, channels, 1, H, W]
            if step + 1 < num_steps and step + 1 < e.shape[1]:
                current_e = e[:, step + 1]
                current_f = f[:, step + 1].clone()  # Clone to avoid in-place modification issues

                # Apply cropping/padding before feeding to model
                pred = pred[..., 
                            self.data_args['padding'][0]:self.data_args['padding'][0]+self.data_args['size'][0], 
                            self.data_args['padding'][1]:self.data_args['padding'][1]+self.data_args['size'][1]]

                pred = self._apply_padding(pred) # pred shape: [batch, 1, levels, H, W]
                
                # Replace last channel with predicted BCB
                # current_f[:, -1] shape: [batch, levels, H, W]
                current_f[:, -1] = pred.squeeze(1)  # Remove channel dimension

        return torch.stack(predictions, dim=1)  # [batch, num_steps, channels, levels, H, W]

    def on_train_epoch_start(self):
        """Update rollout steps at the beginning of each epoch"""
        if self.progressive_training:
            new_steps = self.get_current_rollout_steps()
            print('new steps', new_steps)

            # Update datamodule
            if hasattr(self.trainer, 'datamodule'):
                print('updating dataloader')
                self.trainer.datamodule.set_rollout_steps(new_steps)
                # self.trainer.should_reload_dataloaders = True
            
            # Log current training stage
            stage = "Teacher Forcing" if self.current_epoch < self.stage1_epochs else "Progressive Rollout"
            self.log('rollout_steps', new_steps, on_epoch=True)
            self.log('training_stage', 1 if stage == "Teacher Forcing" else 2, on_epoch=True)

            lr = self.trainer.optimizers[0].param_groups[0]['lr']
            print(f"Epoch {self.current_epoch}: {stage} - {new_steps}-step prediction - Learning Rate: {lr:.4e}")


    @classmethod
    def load_pretrained(cls, checkpoint_path, **kwargs):
        """Load from Lightning checkpoint"""
        return cls.load_from_checkpoint(checkpoint_path, **kwargs)

    def forward(self, e, f):
        return self.model(e, f)

    def training_step(self, batch, batch_idx):
        e, f, y = batch
        
        if self.progressive_training:
            num_steps = self.get_current_rollout_steps()
            
            if num_steps == 1:
                # Single step prediction (teacher forcing or first step)
                y_hat = self.forward(e[:, 0], f[:, 0])
                y_target = y[:, 0]
            else:
                # Multi-step autoregressive rollout
                y_hat = self.autoregressive_forward(e, f, num_steps)
                y_target = y[:, :num_steps]

        y_hat = y_hat[..., 
                      self.data_args['padding'][0]:self.data_args['padding'][0]+self.data_args['size'][0], 
                      self.data_args['padding'][1]:self.data_args['padding'][1]+self.data_args['size'][1]]
        
        total_loss = sum(loss(y_hat, y_target) for loss in self.losses) / len(self.losses)

        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        e, f, y = batch

        if self.progressive_training:
            num_steps = self.get_current_rollout_steps()
            
            if num_steps == 1:
                # Single step prediction (teacher forcing or first step)
                y_hat = self.forward(e[:, 0], f[:, 0])
                y_target = y[:, 0]
            else:
                # Multi-step autoregressive rollout
                y_hat = self.autoregressive_forward(e, f, num_steps)
                y_target = y[:, :num_steps]

        y_hat = y_hat[..., 
                      self.data_args['padding'][0]:self.data_args['padding'][0]+self.data_args['size'][0], 
                      self.data_args['padding'][1]:self.data_args['padding'][1]+self.data_args['size'][1]]

        total_loss = sum(loss(y_hat, y_target) for loss in self.losses) / len(self.losses)

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
            # Warm-up: start from 10% of lr and ramp to full lr over 10 epochs
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **self.model_args['scheduler_params']['warmup_scheduler'])
            
            # Main scheduler: decay from lr=1e-3 using cosine over the remaining epochs
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **self.model_args['scheduler_params']['cosine_scheduler'])

            constant_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, **self.model_args['scheduler_params']['constant_scheduler'])
            
            # Combine Schedulers
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler, constant_scheduler],
                milestones=self.model_args['scheduler_params']['warmup_milestones']  # switch from warm-up to cosine
            )
                
            interval = 'epoch'
        elif self.model_args['scheduler'] == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                **self.model_args['scheduler_params']
            )
            interval = 'step'
        
        elif self.model_args['scheduler'] == 'ConstantLR':
            scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, **self.model_args['scheduler_params']['constant_scheduler'])
            interval = 'epoch'

        else:
            return optimizer

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
