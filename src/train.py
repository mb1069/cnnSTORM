import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import LambdaLR

from src.models import convolutional
from src.data_module import DataModule
import argparse

validation_split = 0.2
epochs = 1000
batch_size = 256

im_size = 32


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss


class LitModel(pl.LightningModule):
    def __init__(self, im_size):
        super().__init__()
        self.model = self.make_model(im_size)
        self.loss = nn.MSELoss()
        self.training_losses = []
        self.validation_losses = []

        self.val_mse = pl.metrics.MeanSquaredError()

    @staticmethod
    def make_model(im_size):
        return convolutional.get_model(im_size)

    def forward(self, x):
        return self.model(x).squeeze(dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        return {'loss': loss, 'preds': y_hat, 'target': y}

    def validation_step_end(self, outputs):
        # update and log
        self.val_mse(outputs['preds'], outputs['target'])
        self.log('val_loss', self.val_mse)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6)
        return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rname')
    args = parser.parse_args()

    dm = DataModule(batch_size)

    wandb_logger = WandbLogger(project='smlm_z', name=args.rname)
    wandb.init()
    wandb.save(__file__)

    model = LitModel(im_size=im_size)

    count_gpus = torch.cuda.device_count() if torch.cuda.is_available() else None
    backend = 'dp' if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else None

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min',
    )

    trainer = pl.Trainer(max_epochs=epochs, gpus=count_gpus, distributed_backend=backend, logger=wandb_logger,
                         callbacks=[early_stop_callback])

    trainer.fit(model, dm)

    model_name = (args.rname or 'default') + '.pth'
    trainer.save_checkpoint(model_name)
    wandb.save(model_name)
