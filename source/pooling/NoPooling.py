import torch
from pytorch_lightning import LightningModule


class NoPooling(LightningModule):
    """
    Performs average pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(NoPooling, self).__init__()

    def forward(self, encoder_outputs, attention_mask=None):
        return encoder_outputs.pooler_output
