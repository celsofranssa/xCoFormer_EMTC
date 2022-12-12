import pickle
from collections import Counter
from pathlib import Path
from random import random

import numpy as np
import pandas as pd
from pytorch_lightning import loggers, seed_everything
from sklearn.model_selection import KFold


class PreprocessHelper:

    def __init__(self, params):
        self.params = params

    def get_text_cls(self, label_ids, label_cls):
        cls = []
        for label_idx in label_ids:
            cls.extend(label_cls[label_idx])
        return list(set(cls))

    def train_val_split(self, ids, split_indice=.9):
        random.shuffle(ids)
        return np.split(
            ids,
            [
                int(
                    split_indice * len(ids)
                )
            ]
        )

    def k_fold_split(self, ids, n_splits=5, random_state=72, shuffle=True):
        folds = []

        kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=shuffle)
        for fold_idx, splits in enumerate(kf.split(ids)):
            train_ids, val_ids = self.train_val_split(splits[0].tolist())
            test_ids = splits[1].tolist()

            folds.append({
                "fold_idx": fold_idx,
                "splits": {
                    "train": train_ids.tolist(),
                    "val": val_ids.tolist(),
                    "test": test_ids
                }
            })

        return folds


    def show_folds(self, folds, samples_to_show=10):
        for fold in folds:
            splits = fold['splits']
            print(f"fold: {fold['fold_idx']} ")
            print(f"train({len(splits['train'])}): {splits['train'][:samples_to_show]} ")
            print(f"val({len(splits['val'])}): {splits['val'][:samples_to_show]} ")
            print(f"test({len(splits['test'])}): {splits['test'][:samples_to_show]} \n")

    def len_stats(self, df):
        df = df.copy()
        df['text_len'] = df["text"].apply(lambda x: len(x.split()))
        text_len_summary = df['text_len'].quantile([.5, .7, .8, .9, .95])
        print(pd.DataFrame(text_len_summary))

        df['label_len'] = df["labels"].apply(lambda labels: len(' '.join(labels).split()))
        label_len_summary = df['label_len'].quantile([.5, .7, .8, .9, .95])
        print(pd.DataFrame(label_len_summary))

    def checkpoint_samples(self, samples_df, dataset_dir):
        samples_df = samples_df[["idx", "text_idx", "text", "labels_ids", "labels"]]
        with open(dataset_dir + "samples.pkl", "wb") as samples_file:
            pickle.dump(samples_df.to_dict(orient="records"), samples_file)

    def checkpoint_folds(self, folds, dataset_dir):
        for fold in folds:
            fold_dir = f"{dataset_dir}fold_{fold['fold_idx']}/"
            Path(fold_dir).mkdir(parents=True, exist_ok=True)
            for split, split_ids in fold['splits'].items():
                with open(f"{fold_dir}{split}.pkl", "wb") as split_file:
                    pickle.dump(split_ids, split_file)

    def checkpoint_relevance_map(self, relevance_map, dataset_dir):
        with open(dataset_dir + "relevance_map.pkl", "wb") as relevance_map_file:
            pickle.dump(relevance_map, relevance_map_file)

    def checkpoint_label_cls(self, label_cls, dataset_dir):
        with open(dataset_dir + "label_cls_back.pkl", "wb") as label_cls_file:
            pickle.dump(label_cls, label_cls_file)

    def checkpoint_text_cls(self, text_cls, dataset_dir):
        with open(dataset_dir + "text_cls_back.pkl", "wb") as text_cls_file:
            pickle.dump(text_cls, text_cls_file)

    def perform_preprocess(self):

        # read raw data
        labels_map = {}
        with open("resource/raw_dataset/xmc-base/amazon-3m/output-items.txt") as labels_file:
            for label_idx, label in enumerate(labels_file):
                labels_map[label_idx] = label.strip()
        x_train = open("/content/xmc-base/amazon-3m/X.trn.txt").readlines()
        y_train = open("/content/xmc-base/amazon-3m/Y.trn.txt").readlines()
        x_test = open("/content/xmc-base/amazon-3m/X.tst.txt").readlines()
        y_test = open("/content/xmc-base/amazon-3m/Y.tst.txt").readlines()

        # get samples
        samples = []
        for text, labels_ids in zip(x_train, y_train):
            samples.append({
                "text": text,
                "labels_ids": [int(label_idx) for label_idx in labels_ids.strip().split(",")]
            })
        for text, labels_ids in zip(x_test, y_test):
            samples.append({
                "text": text,
                "labels_ids": [int(label_idx) for label_idx in labels_ids.strip().split(",")]
            })

        samples_df = pd.DataFrame(samples)
        samples_df["labels"] = samples_df["labels_ids"].apply(
            lambda label_ids: [labels_map[label_idx] for label_idx in label_ids])
        samples_df["idx"] = samples_df.index
        samples_df["text_idx"] = samples_df["text"].astype('category').cat.codes
        samples_df = samples_df[["idx", "text_idx", "text", "labels_ids", "labels"]]
        print(samples_df.head())

        # classify labels
        labels_counter = Counter()
        for labels_ids in samples_df['labels_ids']:
            labels_counter.update(labels_ids)

        label_cls = {}
        full, few, tail = [], [], []
        for label_idx, freq in labels_counter.items():
            if freq > 32:
                full.append(label_idx)
                label_cls[label_idx] = ["all", "full"]
            elif freq > 8:
                few.append(label_idx)
                label_cls[label_idx] = ["all", "few"]
            else:
                tail.append(label_idx)
                label_cls[label_idx] = ["all", "tail"]

        print(f"Num frequent labels: {len(full)}\n"
              f"Num few shot labels: {len(few)}\n"
              f"Num tail labels: {len(tail)}\n"
              f"Total num labels: {len(full) + len(few) + len(tail)}")

        # classify texts
        text_cls = {}
        for _, row in samples_df.iterrows():
            text_idx = row["text_idx"]
            label_ids = row["labels_ids"]
            text_cls[text_idx] = self.get_text_cls(label_ids, label_cls)

        self.len_stats(samples_df)

        # folds
        samples_folds = self.k_fold_split(samples_df["idx"])
        self.show_folds(samples_folds, samples_to_show=16)

        # relevance map
        relevance_map = pd.Series(samples_df["labels_ids"].values, index=samples_df["text_idx"]).to_dict()

        # checkpoint
        dataset_dir = "resource/dataset/Amazon-3M/"
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        self.checkpoint_samples(samples_df, dataset_dir)
        self.checkpoint_folds(samples_folds, dataset_dir)
        self.checkpoint_relevance_map(relevance_map, dataset_dir)
        self.checkpoint_label_cls(label_cls, dataset_dir)
        self.checkpoint_text_cls(text_cls, dataset_dir)
