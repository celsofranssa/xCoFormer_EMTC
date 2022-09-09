import pickle

import nmslib
from ranx import Qrels, Run, evaluate
from torchmetrics import Metric


class MRRMetric(Metric):
    def __init__(self, params):
        super(MRRMetric, self).__init__()
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.texts = []
        self.labels = []

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

    def update(self, text_idx, text_rpr, labels_ids, labels_rpr):

        for text_idx, text_rpr, labels_ids, labels_rpr in zip(
                text_idx.tolist(),
                text_rpr.tolist(),
                labels_ids.tolist(),
                labels_rpr.tolist()):

            self.texts.append({"text_idx": text_idx, "text_rpr": text_rpr})
            for label_idx, label_rpr in zip(labels_ids, labels_rpr):
                if label_idx >= 0: # PAD labels have idx = -1
                    self.labels.append({"label_idx": self.label_idx, "label_rpr": label_rpr})

    def init_index(self):

        # initialize a new index, using a HNSW index on l2 space
        index = nmslib.init(method='hnsw', space='l2')

        for label in self.labels:
            index.addDataPoint(id=label["label_idx"], data=label["label_rpr"])

        index.createIndex(self.params.eval.index)
        return index

    def retrieve(self, index, num_nearest_neighbors):
        ranking = {}
        index.setQueryTimeParams({'efSearch': 2048})
        for text in self.texts:
            text_idx = text["text_idx"]
            retrieved_ids, distances = index.knnQuery(text["text_rpr"], k=num_nearest_neighbors)
            for label_idx, distance in zip(retrieved_ids, distances):
                if f"text_{text_idx}" not in ranking:
                    ranking[f"text_{text_idx}"] = {}
                ranking[f"text_{text_idx}"][f"label_{label_idx}"] = 1.0 / (distance + 1e-9)

        return ranking

    def compute(self):
        # index
        index = self.init_index()

        # retrive
        ranking = self.retrieve(index, num_nearest_neighbors=self.params.eval.num_nearest_neighbors)

        # eval
        return evaluate(
            Qrels({key: value for key, value in self.relevance_map.items() if key in ranking.keys()}),
            Run(ranking),
            ["mrr"]
        )

    def reset(self) -> None:
        self.texts = []
        self.labels = []
