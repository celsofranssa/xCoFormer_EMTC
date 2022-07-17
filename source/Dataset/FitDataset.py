import pickle

import torch
from torch.utils.data import Dataset


class FitDataset(Dataset):
    """CodeSearch Dataset.
    """

    def __init__(self, samples, ids_path, text_tokenizer, label_tokenizer, text_max_length, label_max_length):
        super(FitDataset, self).__init__()
        self.samples = samples
        self.text_tokenizer = text_tokenizer
        self.label_tokenizer = label_tokenizer
        self.text_max_length = text_max_length
        self.label_max_length = label_max_length
        self._load_ids(ids_path)

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

    def _encode(self, sample):
        return {
            "text": torch.tensor(
                self.text_tokenizer.encode(text=sample["text"], max_length=self.text_max_length, padding="max_length",
                                           truncation=True)
            ),
            "label": torch.tensor(
                self.label_tokenizer.encode(text=sample["label"], max_length=self.label_max_length, padding="max_length",
                                            truncation=True)
            )
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        return self._encode(
            self.samples[sample_id]
        )
