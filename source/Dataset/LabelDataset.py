import pickle
import random

import pandas as pd
import torch
from torch.utils.data import Dataset


class LabelDataset(Dataset):
    """Label Dataset.
    """

    def __init__(self, samples, pseudo_labels, ids_path, tokenizer, labels_max_length, max_labels):
        super(LabelDataset, self).__init__()
        self.labels = []
        self.pseudo_labels = pseudo_labels
        self.tokenizer = tokenizer
        self.labels_max_length = labels_max_length
        self.max_labels = max_labels
        self._load_ids(ids_path)

        samples_df = pd.DataFrame(samples)
        for _, r in samples_df[samples_df["idx"].isin(self.ids)].iterrows():
            for label_idx, label in zip(r["labels_ids"], r["labels"]):
                self.labels.append({
                    "label_idx": label_idx,
                    "label": label + " ".join(x[0] for x in self.sample_pseudo_labels(label_idx, 4))
                })

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

    def sample_pseudo_labels(self, label_idx, k):
        pseudo_labels = []
        if label_idx in self.pseudo_labels and len(self.pseudo_labels[label_idx]) > 0:
            pseudo_labels = random.choices([label for (label, _) in self.pseudo_labels[label_idx]],
                                           [weight for (_, weight) in self.pseudo_labels[label_idx]],
                                           k=min(k, len(self.pseudo_labels[label_idx]))
                                           )
        return pseudo_labels

    def _encode(self, sample):
        return {
            "label_idx": sample["label_idx"],
            "label": torch.tensor(
                self.tokenizer.encode(
                    text=sample["label"], max_length=self.labels_max_length, padding="max_length", truncation=True
                ))
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self._encode(self.labels[idx])
