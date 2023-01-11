import pickle
import random
import torch
from torch.utils.data import Dataset


class LabelDataset(Dataset):
    """Label Dataset.
    """

    def __init__(self, samples, pseudo_labels, ids_paths, tokenizer, label_max_length,):
        super(LabelDataset, self).__init__()
        self.labels = []
        self.pseudo_labels = pseudo_labels
        self.tokenizer = tokenizer
        self.label_max_length = label_max_length
        self._load_ids(ids_paths)

        for sample in samples:
            if sample["idx"] in self.ids:
                for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                    self.labels.append({
                        "label_idx": label_idx,
                        "label": label + " ".join(self._get_pseudo_labels(label_idx))
                    })

    def _get_pseudo_labels(self, label_idx):
        pseudo_labels = []
        if label_idx in self.pseudo_labels and len(self.pseudo_labels[label_idx]) > 0:
            pseudo_labels = random.choices(
                [label for (label, _) in self.pseudo_labels[label_idx]],
                [weight for (_, weight) in self.pseudo_labels[label_idx]],
                k=16
            )
        return pseudo_labels

    def _load_ids(self, ids_paths):
        self.ids = []
        for path in ids_paths:
            with open(path, "rb") as ids_file:
                self.ids.extend(pickle.load(ids_file))

    def _encode(self, sample):

        return {
            "label_idx": sample["label_idx"],
            "label": torch.tensor(
                self.tokenizer.encode(
                    text=sample["label"], max_length=self.label_max_length, padding="max_length", truncation=True
                ))
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self._encode(
            self.labels[idx]
        )
