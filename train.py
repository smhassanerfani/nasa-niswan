import os
import yaml
import argparse
from datetime import datetime

from model import STMLightning
from dataset import E33OMAModule

import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(42)

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

# Set environment variables to help manage CUDA memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def main(args):

    # Retrieve hyperparameters
    with open(args.config_filepath, 'r') as config_filepath:
        hyperparams = yaml.load(config_filepath, Loader=yaml.FullLoader)
        
    model_args = hyperparams['model_args']
    data_args = hyperparams['data_args']
    
    # Initialize model (model_args=model_args, data_args=data_args)
    model = STMLightning(model_args=model_args, data_args=data_args)
    dm = E33OMAModule(data_args)

    # Get the current date and time
    now = datetime.now()
    formatted_date_time = now.strftime("%m%d%Y-%H%M")

    # Initialize training
    model_args['log_name'] = model_args['model_name'] + '-' + formatted_date_time
    tb_logger = TensorBoardLogger(save_dir=model_args['log_dir'], name=model_args['log_name'])
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min')

    trainer = pl.Trainer(
        max_epochs=model_args['epochs'],
        accelerator="gpu",
        devices=1,
        precision=32,
        logger=tb_logger,
        callbacks=[checkpoint_callback]
        )
    
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-filepath', help='Provide the filepath string to the model config...', default='configs/ConvLSTM-02102025.yaml')
    
    args = parser.parse_args()
    main(args)