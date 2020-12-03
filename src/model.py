import pytorch_lightning as pl
import torch.nn as nn

im_size = 32


def linear_relu(in_features, out_features):
    return nn.Linear(in_features=in_features, out_features=out_features), nn.ReLU()


class ResNet256(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = self.make_model()

    def make_model(self):
        return nn.Sequential(
            nn.Flatten(),
            *linear_relu(im_size * im_size, 256),
            *linear_relu(256, 128),
            *linear_relu(128, 64),
            *linear_relu(64, 32),
            nn.Linear(32, 1)
        )


model = ResNet256()

