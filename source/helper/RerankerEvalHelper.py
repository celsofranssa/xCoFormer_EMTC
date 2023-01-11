import pickle
from pathlib import Path

import nmslib
import pandas as pd
import torch
from omegaconf import OmegaConf
from ranx import Qrels, Run, evaluate
from tqdm import tqdm


class RerankerEvalHelper:
    def __init__(self, params):
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.labels_cls = self._load_labels_cls()
        self.texts_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()


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

    def _get_metrics(self):
        metrics = []
        for metric in self.params.eval.metrics:
            for threshold in self.params.eval.thresholds:
                metrics.append(f"{metric}@{threshold}")
        return metrics

    def _load_labels_cls(self):
        labels_cls = {}
        with open(f"{self.params.data.dir}label_cls.pkl", "rb") as label_cls_file:
            for label_idx, cls in pickle.load(label_cls_file).items():
                labels_cls[f"label_{label_idx}"] = cls
        return labels_cls

    def _load_texts_cls(self):
        texts_cls = {}
        with open(f"{self.params.data.dir}text_cls.pkl", "rb") as text_cls_file:
            for text_idx, cls in pickle.load(text_cls_file).items():
                texts_cls[f"text_{text_idx}"] = cls
        return texts_cls

    # def _load_rankings(self):
    #     print(f"{self.params.ranking.dir}{self.params.ranking.name}.rnk")
    #     with open(f"{self.params.ranking.dir}{self.params.ranking.name}.rnk", "rb") as ranking_file:
    #         rankings = pickle.load(ranking_file)
    #
    #     for fold_idx in self.params.data.folds:
    #             ranking = rankings[fold_idx]["all"]
    #             for text_idx, labels_scores in ranking.items():
    #                 min_socore, max_score = min(labels_scores.values()), max(labels_scores.values())
    #                 for label_idx, score in labels_scores.items():
    #                     ranking[text_idx][label_idx] = (ranking[text_idx][label_idx] - min_socore) / (max_score - min_socore)
    #     return rankings

    def _load_predictions(self, fold_idx):
        ranking = {}
        print(f"Prediction dir: {self.params.prediction.dir}fold_{fold_idx}/")
        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold_idx}/").glob("*.prd")
        )
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            for prediction in torch.load(path):
                text_idx = prediction["text_idx"]
                label_idx = prediction["label_idx"]
                score = prediction["pred_cls"]
                if f"text_{text_idx}" not in ranking:
                    ranking[f"text_{text_idx}"] = {}
                if score > ranking[f"text_{text_idx}"].get(f"label_{label_idx}", -1e9):
                    ranking[f"text_{text_idx}"][f"label_{label_idx}"] = score

        return ranking

    # def _re_score(self, ranking, fold_idx):
    #     for text_idx, labels_score in ranking.items():
    #         for label_idx, score in labels_score.items():
    #             score = 0.5 * (score + self.rankings[fold_idx]["all"][text_idx][label_idx])
    #             ranking[text_idx][label_idx]=score

    def _filter_ranking(self, ranking, cls):
        ranking = {key: value for key, value in ranking.items() if cls in self.texts_cls[key]}
        for text_idx, labels_scores in ranking.items():
            labels_scores = {key: value for key, value in labels_scores.items() if cls in self.labels_cls[key]}
            ranking[text_idx] = labels_scores
        return ranking

    def perform_eval(self):
        results = []
        rankings = []
        for fold_idx in self.params.data.folds:
            print(
                f"Evaluating {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling "
                f"params\n "
                f"{OmegaConf.to_yaml(self.params)}\n")

            ranking = self._load_predictions(fold_idx)

            for cls in self.params.eval.label_cls:
                cls_ranking = self._filter_ranking(ranking, cls)
                result = evaluate(
                    Qrels(
                        {key: value for key, value in self.relevance_map.items() if key in cls_ranking.keys()}
                    ),
                    Run(cls_ranking),
                    self.metrics
                )
                result = {k: round(v, 3) for k, v in result.items()}
                result["fold"] = fold_idx
                result["cls"] = cls

                results.append(result)
                rankings.append(ranking)

        self._checkpoint_results(results)
        # self._checkpoint_rankings(rankings)

    def _checkpoint_results(self, results):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        pd.DataFrame(results).to_csv(
            self.params.result.dir + self.params.model.name + "_" + self.params.data.name + ".rts",
            sep='\t', index=False, header=True)

    # def _checkpoint_rankings(self, rankings):
    #     with open(
    #             self.params.ranking.dir + self.params.model.name + "_" + self.params.data.name + ".rnk",
    #             "wb") as rankings_file:
    #         pickle.dump(rankings, rankings_file)

