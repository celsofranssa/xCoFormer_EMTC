import pickle

import torch
from pytorch_metric_learning.miners import BaseMiner


class RelevanceMiner(BaseMiner):

    def __init__(self, params):
        super().__init__()
        with open(f"{params.relevance_map.dir}relevance_map.pkl", "rb") as relevance_map_file:
            self.relevance_map = pickle.load(relevance_map_file)

    def mine(self, text_ids, _, label_ids, __):
        a1, p, a2, n = [], [], [], []
        for i, text_idx in enumerate(text_ids.tolist()):
            for j, label_idx in enumerate(label_ids.tolist()):
                if label_idx >= 0:
                    if label_idx in self.relevance_map[text_idx]:
                        a1.append(i)
                        p.append(j)
                    else:
                        a2.append(i)
                        n.append(j)

        return torch.tensor(a1), torch.tensor(p), torch.tensor(a2), torch.tensor(n)

    def output_assertion(self, output):
        pass
