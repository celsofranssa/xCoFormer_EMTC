import torch
from pytorch_lightning import LightningModule


class MaxPooling(LightningModule):
    """
    Performs max pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, attention_mask, hidden_states):
        """
        :param attention_mask:
        :param hidden_states:
        :return:
        """
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states[attention_mask == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(hidden_states, 1)[0]


>>>>>>> 97699ed445479fbe726b44e18cda87a2b04578c2
