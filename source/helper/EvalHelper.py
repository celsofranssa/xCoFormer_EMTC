import json
import pickle
from pathlib import Path
import nmslib
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


class EvalHelper:
    def __init__(self, params):
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.labels_cls = self._load_labels_cls()
        self.texts_cls = self._load_texts_cls()

    def _load_relevance_map(self):
        with open(f"{self.params.data.dir}relevance_map.pkl", "rb") as relevances_file:
            return pickle.load(relevances_file)

    def _load_labels_cls(self):
        with open(f"{self.params.data.dir}labels_cls.pkl", "rb") as labels_cls_file:
            return pickle.load(labels_cls_file)

    def _load_texts_cls(self):
        with open(f"{self.params.data.dir}texts_cls.pkl", "rb") as texts_cls_file:
            return pickle.load(texts_cls_file)

    def mrr_at_k(self, positions, k, num_samples):
        """
        Evaluates the MMR considering only the positions up to k.
        :param positions:
        :param k:
        :param num_samples:
        :return:
        """
        # positions_at_k = [p for p in positions if p <= k]
        positions_at_k = [p if p <= k else 0 for p in positions]
        rrank = 0.0
        for pos in positions_at_k:
            if pos != 0:
                rrank += 1.0 / pos

        return rrank / num_samples

    def mrr(self, ranking):
        """
        Evaluates the MMR considering only the positions up to k.
        :param positions:
        :param num_samples:
        :return:
        """
        return np.mean(ranking)

    def recall_at_k(self, positions, k, num_samples):
        """
        Evaluates the Recall considering only the positions up to k
        :param positions:
        :param k:
        :param num_samples:
        :return:
        """
        return 1.0 * sum(i <= k for i in positions) / num_samples

    def checkpoint_stats(self, stats):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        stats.to_csv(
            self.params.stat.dir + self.params.model.name + "_" + self.params.data.name + ".stat",
            sep='\t', index=False, header=True)

    def checkpoint_ranking(self, ranking):
        ranking_path = f"{self.params.ranking.dir}" \
                       f"{self.params.model.name}_" \
                       f"{self.params.data.name}.rnk"
        with open(ranking_path, "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)


    def load_predictions(self, fold):

        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold}/").glob("*.prd")
        )

        # for path in tqdm(predictions_paths, desc="Loading predictions"):
        #     prediction = torch.load(path)


        text_predictions = []
        label_predictions = []
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            text_predictions.extend(  # only eval over test split
                filter(lambda prediction: prediction["modality"] == "text", torch.load(path))
            )
            label_predictions.extend(  # only eval over test split
                filter(lambda prediction: prediction["modality"] == "label", torch.load(path))
            )

        return text_predictions, label_predictions

    def init_index(self, label_predictions):
        M = 30
        efC = 100
        num_threads = 4
        index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}

        # initialize a new index, using a HNSW index on Cosine Similarity
        index = nmslib.init(method='hnsw', space='cosinesimil')

        for prediction in tqdm(label_predictions, desc="Indexing"):
            index.addDataPoint(id=prediction["idx"], data=prediction["rpr"])

        index.createIndex(index_time_params)
        return index



    def retrieve(self, index, text_predictions, k):
        # retrieve
        ranking = {}
        for prediction in tqdm(text_predictions, desc="Searching"):
            text_idx = prediction["idx"]
            retrieved_ids, _ = index.knnQuery(prediction["rpr"], k=k)
            retrieved_ids = retrieved_ids.tolist()
            ranking[text_idx] = self._get_relevant_rank(text_idx, retrieved_ids)

        return ranking

    def _get_relevant_rank(self, text_idx, retrieved_ids):
        for position, label_idx in enumerate(retrieved_ids):
            if label_idx in self.relevance_map[text_idx]:
                return position+1
        return 1e9

    def get_ranking(self, text_predictions, label_predictions, num_nearest_neighbors):
        # index data
        index = self.init_index(label_predictions)

        # retrieve
        return self.retrieve(index, text_predictions, k=num_nearest_neighbors)


    def perform_eval(self):
        rankings = {}
        thresholds = [1, 5, 10]
        label_cls = ["all", "full", "few", "tail"]
        stats = []

        for fold in self.params.data.folds:
            print(
                f"Evaluating {self.params.model.name} over {self.params.data.name} (fold {fold}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            all_text_predictions, all_label_predictions = self.load_predictions(fold)

            for cls in label_cls:
                stat = {}
                label_predictions = self.filter_label_predictions(all_label_predictions, cls)
                text_predictions = self.filter_text_predictions(all_text_predictions, cls)
                ranking = self.get_ranking(text_predictions, label_predictions, num_nearest_neighbors=thresholds[-1])
                stat["fold"] = fold
                stat["cls"] = cls
                for k in thresholds:
                    stat[f"MRR@{k}"] = self.mrr_at_k(ranking.values(), k, len(ranking))
                    stat[f"RCL@{k}"] = self.recall_at_k(ranking.values(), k, len(ranking))
                stats.append(stat)



        self.checkpoint_stats(
            pd.DataFrame(
                stats,
                columns=["fold", "cls", "MRR@1", "MRR@5", "MRR@10", "RCL@1", "RCL@5", "RCL@10"]
            ).sort_values(by=["cls"])
        )
        self.checkpoint_ranking(rankings)

    def filter_label_predictions(self, label_predictions, label_cls):
        if label_cls == "all":
            return label_predictions
        return filter(
            lambda prediction:
            label_cls == self.labels_cls[prediction["idx"]],
            label_predictions
        )
    def filter_text_predictions(self, text_predictions, label_cls):
        if label_cls == "all":
            return text_predictions
        return filter(
            lambda prediction:
            label_cls in self.texts_cls[prediction["idx"]],
            text_predictions
        )

