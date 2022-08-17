import torch
from torch import nn
from pytorch_metric_learning import miners, losses

class NTXentLoss(nn.Module):

    def __init__(self, params):
        super(NTXentLoss, self).__init__()
        self.miner = miners.MultiSimilarityMiner(epsilon=params.miner.epsilon)
        self.criterion = losses.NTXentLoss(temperature=params.criterion.temperature)

    def forward(self, cls_ids, text_rpr, label_rpr):
        """
        Computes the NTXentLoss.
        """
        cls = torch.cat([cls_ids,cls_ids])
        rpr = torch.cat([text_rpr,label_rpr])
        pairs = self.miner(rpr, cls)
        return self.criterion(rpr, cls, pairs)
