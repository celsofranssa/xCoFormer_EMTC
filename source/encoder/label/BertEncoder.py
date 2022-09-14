from pytorch_lightning import LightningModule
from transformers import BertModel

class BertEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions, pooling):
        super(BertEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(
            architecture,
            output_attentions=output_attentions
        )
        self.pooling = pooling

    def forward(self, labels, label_masks):
        attention_mask = (labels > 0).int()
        encoder_outputs = self.encoder(labels, attention_mask)

        return self.pooling(
            encoder_outputs,
            label_masks

        )
