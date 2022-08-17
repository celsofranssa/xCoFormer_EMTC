import pickle
from pathlib import Path

import nmslib
import pandas as pd
import torch
from omegaconf import OmegaConf
from ranx import Qrels, Run, evaluate
from tqdm import tqdm


class EvalHelper:
    def __init__(self, params):
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.labels_cls = self._load_labels_cls()
        self.texts_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()

    def _get_metrics(self):
        metrics = []
        for metric in self.params.eval.metrics:
            for threshold in self.params.eval.thresholds:
                metrics.append(f"{metric}@{threshold}")
        return metrics

    def _load_relevance_map(self):
        with open(f"{self.params.data.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for text_idx, labels_ids in data.items():
            d = {}
            for label_idx in labels_ids:
                d[f"label_{label_idx}"] = 1.0
            relevance_map[f"text_{text_idx}"] = d
        return relevance_map

    def _load_labels_cls(self):
        with open(f"{self.params.data.dir}label_cls.pkl", "rb") as label_cls_file:
            return pickle.load(label_cls_file)

    def _load_texts_cls(self):
        with open(f"{self.params.data.dir}text_cls.pkl", "rb") as text_cls_file:
            return pickle.load(text_cls_file)

    def _load_predictions(self, fold):

        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold}/").glob("*.prd")
        )

        text_predictions = []
        label_predictions = []
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            text_predictions.extend(  # only text prediction
                filter(lambda prediction: prediction["modality"] == "text", torch.load(path))
            )
            label_predictions.extend(  # only label prediction
                filter(lambda prediction: prediction["modality"] == "label", torch.load(path))
            )

        return text_predictions, label_predictions

    def init_index(self, label_predictions, cls):
        M = 256
        efC = 2048
        num_threads = 12
        index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 2}

        # initialize a new index, using a HNSW index on Cosine Similarity
        index = nmslib.init(method='hnsw', space='l2')


        for prediction in tqdm(label_predictions, desc="Adding data to index"):
            label_idx = prediction["idx"]
            if cls in self.labels_cls[label_idx]:
                index.addDataPoint(id=label_idx, data=prediction["rpr"])

        index.createIndex(index_time_params)
        return index

    def retrieve(self, index, text_predictions, cls, num_nearest_neighbors):
        # retrieve
        ranking = {}
        index.setQueryTimeParams({'efSearch':2048})
        for prediction in tqdm(text_predictions, desc="Searching"):
            text_idx = prediction["idx"]
            if cls in self.texts_cls[text_idx]:
                retrieved_ids, distances = index.knnQuery(prediction["rpr"], k=num_nearest_neighbors)
                for label_idx, distance in zip(retrieved_ids, distances):
                    if f"text_{text_idx}" not in ranking:
                        ranking[f"text_{text_idx}"] = {}
                    ranking[f"text_{text_idx}"][f"label_{label_idx}"] = 1.0 / (distance + 1e-9)

        return ranking

    def _get_ranking(self, text_predictions, label_predictions, cls, num_nearest_neighbors):
        # index data
        index = self.init_index(label_predictions, cls)

        # retrieve
        return self.retrieve(index, text_predictions, cls, num_nearest_neighbors)

    def perform_eval(self):
        results = []
        rankings = []
        for fold_id in self.params.data.folds:
            print(
                f"Evaluating {self.params.model.name} over {self.params.data.name} (fold {fold_id}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            text_predictions, label_predictions = self._load_predictions(fold_id)

            for cls in self.params.eval.label_cls:
                ranking = self._get_ranking(text_predictions, label_predictions, cls=cls,
                                            num_nearest_neighbors=self.params.eval.num_nearest_neighbors)
                # filtered_dictionary = {key: value for key, value in self.relevance_map.items() if key in ranking.keys()}
                # qrels = Qrels(filtered_dictionary, name=cls)
                # run = Run(ranking, name=cls)
                # result = evaluate(qrels, run, self.metrics, threads=12)
                result = evaluate(
                    Qrels(
                        {key: value for key, value in self.relevance_map.items() if key in ranking.keys()}
                    ),
                    Run(ranking),
                    self.metrics
                )
                result["fold"] = fold_id
                result["cls"] = cls

                results.append(result)
                rankings.append(ranking)

        self._checkpoint_results(results)
        self._checkpoint_rankings(rankings)

    def _checkpoint_results(self, results):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        pd.DataFrame(results).to_csv(
            self.params.result.dir + self.params.model.name + "_" + self.params.data.name + ".rts",
            sep='\t', index=False, header=True)

    def _checkpoint_rankings(self, rankings):
        with open(
            self.params.ranking.dir + self.params.model.name + "_" + self.params.data.name + ".rnk",
            "wb") as rankings_file:
            pickle.dump(rankings, rankings_file)

