import math
import pickle
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SiEMTCDataset(Dataset):
    """Fit Dataset.
    """

    def __init__(self, samples, pseudo_labels, ids_paths, tokenizer, text_max_length, label_max_length):
        super(SiEMTCDataset, self).__init__()
        self.samples = []
        self.pseudo_labels = pseudo_labels
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.label_max_length = label_max_length
        self._load_ids(ids_paths)

        for sample in samples:
            if sample["idx"] in self.ids:
                for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                    self.samples.append({
                        "text_idx": sample["text_idx"],
                        "text": sample["text"],
                        "label_idx": label_idx,
                        "label": label + " ".join(x[0] for x in pseudo_labels[label_idx])
                    })

    def _load_ids(self, ids_paths):
        self.ids = []
        for path in ids_paths:
            with open(path, "rb") as ids_file:
                self.ids.extend(pickle.load(ids_file))

    def _encode(self, sample):

        return {
            "text_idx": sample["text_idx"],
            "text": torch.tensor(
                self.tokenizer.encode(
                    text=sample["text"], max_length=self.text_max_length, padding="max_length", truncation=True
                )),
            "label_idx": sample["label_idx"],
            "label": torch.tensor(
                self.tokenizer.encode(
                    text=sample["label"], max_length=self.label_max_length, padding="max_length", truncation=True
                ))
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )


