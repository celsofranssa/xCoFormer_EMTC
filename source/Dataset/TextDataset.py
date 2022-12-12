import pickle
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Text Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, text_max_length):
        super(TextDataset, self).__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self._load_ids(ids_path)

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

    def _encode(self, sample):
        return {
            "text_idx": sample["text_idx"],
            "text": torch.tensor(
                self.tokenizer.encode(
                    text=sample["text"], max_length=self.text_max_length, padding="max_length", truncation=True
                ))
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[
                self.ids[idx]
            ]
        )
