import torch
from torch import nn
from pytorch_metric_learning import miners, losses

from source.miner.RelevanceMiner import RelevanceMiner


class NTXentLoss(nn.Module):

    def __init__(self, params):
        super(NTXentLoss, self).__init__()
        self.miner = RelevanceMiner(params.miner)
        self.criterion = losses.NTXentLoss(temperature=params.criterion.temperature)

    def forward(self, text_idx, text_rpr, label_idx, label_rpr):

        """
        Computes the NTXentLoss.
        """
        miner_outs = self.miner.mine(text_ids=text_idx, label_ids=torch.flatten(label_idx))
        return self.criterion(text_rpr, None, miner_outs, label_rpr, None)

