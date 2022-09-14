import torch
from pytorch_lightning import LightningModule


class LabelMaxPooling(LightningModule):
    """
    Performs max pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(LabelMaxPooling, self).__init__()

    def forward(self, encoder_outputs, label_masks):

        hidden_state = encoder_outputs.last_hidden_state
        # print(f"\nLabel hidden_state shape: {hidden_state.shape}\n")
        # print(f"\nLabel label_masks shape: {label_masks.shape}\n")
        m = (label_masks.unsqueeze(-1) == 0).float() * (-1e30)
        label_rpr = m + hidden_state.unsqueeze(1).repeat(1, label_masks.shape[1], 1, 1)
        label_rpr = label_rpr.max(dim=2)[0]
        # print(f"\nLabel label_rpr shape: {label_rpr.shape}\n")
        return label_rpr
