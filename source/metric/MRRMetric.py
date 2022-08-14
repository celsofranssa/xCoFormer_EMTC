import pickle
import torch
from ranx import Qrels, Run, evaluate
from torchmetrics import Metric, RetrievalMRR

class MRRMetric(Metric):
    def __init__(self, params):
        super(MRRMetric, self).__init__()
        self.run={}
        self.relevance_map = self._load_relevance_map()

    def _load_relevance_map(self):
        with open(f"{self.params.relevance_map.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for text_idx, labels_ids in data.items():
            d = {}
            for label_idx in labels_ids:
                d[f"label_{label_idx}"] = 1.0
            relevance_map[f"text_{text_idx}"] = d
        return relevance_map

    def similarities(self, x1, x2):
        """
        Calculates the cosine similarity matrix for every pair (i, j),
        where i is an embedding from x1 and j is another embedding from x2.

        :param x1: a tensors with shape [batch_size, hidden_size].
        :param x2: a tensors with shape [batch_size, hidden_size].
        :return: the cosine similarity matrix with shape [batch_size, batch_size].
        """
        x1 = x1 / torch.norm(x1, dim=1, p=2, keepdim=True)
        x2 = x2 / torch.norm(x2, dim=1, p=2, keepdim=True)
        return torch.matmul(x1, x2.t())


    def update(self, text_ids, text_rpr, label_ids, label_rpr):
        similarities = self.similarities(text_rpr, label_rpr)
        for row, text_idx in enumerate(text_ids.tolist()):
            for col, label_idx in enumerate(label_ids.tolist()):
                self.run[f"text_{text_idx}"]={"label_{label_idx}": similarities[row][col].item()}


    def compute(self):
        qrels = Qrels(self.relevance_map)
        run = Run(self.run)
        return evaluate(qrels, run, ["mrr"])