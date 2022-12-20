import pickle
import nmslib
import torch
from omegaconf import OmegaConf
from ranx import Qrels, Run, evaluate
from torchmetrics import Metric


class MRRMetric(Metric):
    def __init__(self, params):
        super(MRRMetric, self).__init__(compute_on_cpu=True)
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

    def update(self, text_idx, text_rpr, label_idx, label_rpr):

        for text_idx, text_rpr in zip(
                text_idx.tolist(),
                text_rpr.tolist()):
            self.texts.append({"text_idx": text_idx, "text_rpr": text_rpr})

        # print(f"\nlabels_ids ({labels_ids.shape}):\n {labels_ids}\n")
        # print(f"\nlabels_rpr ({labels_rpr.shape}):\n {labels_rpr}\n")

        for label_idx, label_rpr in zip(
                label_idx.tolist(),
                label_rpr.tolist()):
            # print(f"\nlabel_idx:\n {label_idx}\n")
            # print(f"\nlabel_rpr:\n {label_rpr}\n")
            if label_idx >= 0:  # PAD labels have idx = -1
                self.labels.append({"label_idx": label_idx, "label_rpr": label_rpr})

    def init_index(self):

        # initialize a new index, using a HNSW index on l2 space
        index = nmslib.init(method='hnsw', space='l2')

        for label in self.labels:
            index.addDataPoint(id=label["label_idx"], data=label["label_rpr"])

        index.createIndex(
            index_params=OmegaConf.to_container(self.params.index),
            print_progress=False
        )



        return index

    def retrieve(self, index, num_nearest_neighbors):
        ranking = {}
        # index.setQueryTimeParams({'efSearch': 2048})
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
        ranking = self.retrieve(index, num_nearest_neighbors=self.params.num_nearest_neighbors)

        # print(f"\n\nranking ({len(ranking)}):\n{ranking}\n")

        # eval
        m = evaluate(
            Qrels({key: value for key, value in self.relevance_map.items() if key in ranking.keys()}),
            Run(ranking),
            ["mrr"]
        )

        # print(f"MRR: {m}")
        return m

    def reset(self) -> None:
        self.texts = []
        self.labels = []
