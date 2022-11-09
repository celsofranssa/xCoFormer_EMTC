import torch
from pytorch_lightning import LightningModule
from transformers import RobertaModel

class RoBERTaEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions):
        super(RoBERTaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(
            architecture,
            output_attentions=output_attentions
        )

    def forward(self, feature):
        return self.encoder(
            feature,
            torch.where(feature != 1, 1, 0)
        )

