import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class LabelDataset(Dataset):
    """Eval Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, max_length):
        super(LabelDataset, self).__init__()
        self._filter_and_reshape_samples(
            samples,
            self._load_ids(ids_path)
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _filter_and_reshape_samples(self, samples, ids):
        self.labels = {}
        for sample in tqdm(samples, desc="Reshaping samples"):
            if sample["idx"] in ids:
                for label_idx, label in zip(sample["label_ids"], sample["labels"]):
                    self.labels[label_idx] = label


    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            return pickle.load(ids_file)

    def _encode(self, label_idx, label):
        return {
            "idx": label_idx,
            "modality": "label",
            "sample": torch.tensor(
                self.tokenizer.encode(
                    text=label, max_length=self.max_length, padding="max_length", truncation=True
                )
            )
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label_idx = list(self.labels)[idx]
        return self._encode(
            label_idx=label_idx,
            label=self.labels[label_idx]
        )