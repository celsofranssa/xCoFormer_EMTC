import pickle

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
        ctrl = {}

        samples_df = pd.DataFrame(samples)
        for _, r in samples_df[samples_df["idx"].isin(self.ids)].iterrows():
            for label_idx, label in zip(r["labels_ids"], r["labels"]):
                if label_idx not in ctrl:
                    self.labels.append({
                        "label_idx": label_idx,
                        "label": label + " ".join(x[0] for x in pseudo_labels[label_idx])
                    })
                    ctrl[label_idx]=True

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

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
