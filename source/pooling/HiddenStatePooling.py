import torch
from pytorch_lightning import LightningModule


class HiddenStatePooling(LightningModule):
    """
    Performs average pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(HiddenStatePooling, self).__init__()

    def forward(self, encoder_outputs, attention_mask):
        """
        """
        hidden_states = encoder_outputs.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # put 0 over PADs
        return hidden_states * attention_mask
