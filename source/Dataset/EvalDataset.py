import pickle

import torch
from torch.utils.data import Dataset


class EvalDataset(Dataset):
    """Eval Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, max_length, modality):
        super(EvalDataset, self).__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.modality = modality
        self._load_ids(ids_path)

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

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
        return len(self.ids)

    def __getitem__(self, idx):
        sample_idx = self.ids[idx]
        return self._encode(
            self.samples[sample_idx]
        )