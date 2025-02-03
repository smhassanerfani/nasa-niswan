import os

from model import STMLightning
from dataset import E33OMAModule

import torch

from pytorch_lightning import Trainer
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.loggers import TensorBoardLogger

torch.cuda.empty_cache()
torch.set_float32_matmul_precision('medium')

# Set environment variables to help manage CUDA memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

if __name__ == "__main__":
    logger = TensorBoardLogger("tb_logs", name="Conv3D-V02-02032025")

    model = STMLightning(in_channels=5, out_channels=1, kernel_size=(2, 3, 3))
    # model = STMLightning(in_channels=5, out_channels=[64, 32, 16], kernel_size=[5, 3, 3])
    dm = E33OMAModule(sequence_length=64, padding=(100, 154), batch_size=16, num_workers=4)

    trainer = Trainer(
        logger=logger,
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision=32
        )
    
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    
    # # Create a Tuner object
    # tuner = Tuner(trainer)

    # # Find the optimal learning rate
    # lr_finder = tuner.lr_find(model, datamodule=dm, min_lr=1e-5, max_lr=1e-1)

    # # Plot the learning rate finder results
    # fig = lr_finder.plot(suggest=True)
    # fig.show()

    # # Update the model's learning rate
    # new_lr = lr_finder.suggestion()
    # model.hparams.lr = new_lr  # Assuming your model uses `hparams.lr` for the optimizer
    # print(f"Suggested learning rate: {new_lr}")

    # Re-train the model with the new learning rate
    # trainer.fit(model, datamodule=dm)
