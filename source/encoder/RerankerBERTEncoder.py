import torch
from pytorch_lightning import LightningModule
from transformers import BertModel


class RerankerBERTEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions):
        super(RerankerBERTEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(
            architecture,
            output_attentions=output_attentions
        )

    def forward(self, features):
        return self.encoder(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            token_type_ids=features["token_type_ids"]
        )
