import torch
from pytorch_lightning import LightningModule


class HiddenStatePooling(LightningModule):
    """
    Performs average pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(HiddenStatePooling, self).__init__()
        # self.linear = torch.nn.Linear(768, 768, bias=True)

    def forward(self, encoder_outputs, attention_mask):
        """
        """
        hidden_states = encoder_outputs.last_hidden_state
        # hidden_states = self.linear(hidden_states)
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

        # put 0 over PADs
        return hidden_states * attention_mask
        # return torch.nn.functional.normalize(a, p=2, dim=2)
