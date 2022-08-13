import pickle
import torch
from torchmetrics import Metric, RetrievalMRR

class MRRMetric(Metric):
    def __init__(self, params):
        super(MRRMetric, self).__init__()
        self.retrieval_mrr = RetrievalMRR()
        self._load_relevance_map(f"{params.relevance_map.dir}relevance_map.pkl")

    def _load_relevance_map(self, relevance_map_path):
        with open(relevance_map_path, "rb") as relevance_map_file:
            self.relevance_map = pickle.load(relevance_map_file)

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

    def flatten(self, tensor):
        return torch.flatten(tensor)

    def update(self, text_idx, text_rpr, label_idx, label_rpr):
        pairs = torch.cartesian_prod(text_idx, label_idx)
        target = torch.tensor([y.item() in self.relevance_map[x.item()] for x, y in pairs])
        scores = self.flatten(
            self.similarities(text_rpr, label_rpr)
        )
        indexes = text_idx.repeat_interleave(text_idx.shape[0])
        self.retrieval_mrr.update(scores, target, indexes)

    def compute(self):
        return self.retrieval_mrr.compute()