import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class TextDataset(Dataset):
    """Eval Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, max_length):
        super(TextDataset, self).__init__()
        self._filter_and_reshape_samples(
            samples,
            self._load_ids(ids_path)
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _filter_and_reshape_samples(self, samples, ids):
        self.texts = {}
        for sample in tqdm(samples, desc="Reshaping samples"):
            if sample["idx"] in ids:
                self.texts[sample["text_idx"]]=sample["text"]


    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            return pickle.load(ids_file)

    def _encode(self, text_idx, text):
        return {
            "idx": text_idx,
            "modality": "text",
            "sample": torch.tensor(
                self.tokenizer.encode(
                    text=text, max_length=self.max_length, padding="max_length", truncation=True
                )
            )
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_idx = list(self.texts)[idx]
        return self._encode(
            text_idx=text_idx,
            text=self.texts[text_idx]
        )