import pickle
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Text Dataset.
    """

    def __init__(self, samples, ids_paths, tokenizer, text_max_length):
        super(TextDataset, self).__init__()
        self.texts = []
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self._load_ids(ids_paths)

        for sample in samples:
            if sample["idx"] in self.ids:
                self.texts.append({
                    "text_idx": sample["text_idx"],
                    "text": sample["text"]
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
                ))
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self._encode(
            self.texts[idx]
        )
