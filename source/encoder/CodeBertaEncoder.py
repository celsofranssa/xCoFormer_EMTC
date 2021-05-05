from pytorch_lightning import LightningModule
from transformers import RobertaModel

from source.pooling.AveragePooling import AveragePooling


class CodeBertaEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(CodeBertaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(
            hparams.architecture,
            output_attentions=hparams.output_attentions
        )
        self.pooling = AveragePooling()

    def forward(self, features):

        attention_mask = (features != 1).int()
        hidden_states = self.encoder(features, attention_mask).last_hidden_state

        return self.pooling(
            attention_mask,
            hidden_states
        )
