import pickle

import torch
from torch.utils.data import Dataset


class EvalDataset(Dataset):
    """Eval Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, max_length, modality):
        super(EvalDataset, self).__init__()
        self._filter_samples(
            samples,
            self._load_ids(ids_path)
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.modality = modality

    def _filter_samples(self, samples, ids):
        self.samples = filter(
            lambda sample:
            sample["idx"] in ids,
            samples
        )


    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            return pickle.load(ids_file)

    def _encode(self, sample):
        if self.modality == "text":
            return {
                "idx": sample["text_idx"],
                "modality": "text",
                "sample": torch.tensor(
                    self.tokenizer.encode(
                        text=sample["text"], max_length=self.max_length, padding="max_length", truncation=True
                    )
                )
            }
        elif self.modality == "label":
            return {
                "idx": sample["label_idx"],
                "modality": "label",
                "sample": torch.tensor(
                    self.tokenizer.encode(
                        text=sample["label"], max_length=self.max_length, padding="max_length", truncation=True
                    )
                )
            }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )