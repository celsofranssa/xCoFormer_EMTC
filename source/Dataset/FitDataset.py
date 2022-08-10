import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class FitDataset(Dataset):
    """Fit Dataset.
    """

    def __init__(self, samples, ids_path, text_tokenizer, label_tokenizer, text_max_length, label_max_length):
        super(FitDataset, self).__init__()
        self._filter_and_reshape_samples(
            samples,
            self._load_ids(ids_path)
        )
        self.text_tokenizer = text_tokenizer
        self.label_tokenizer = label_tokenizer
        self.text_max_length = text_max_length
        self.label_max_length = label_max_length

    def _filter_and_reshape_samples(self, samples, ids):
        self.samples = []

        for sample in tqdm(samples, desc="Reshaping samples"):
            if sample["idx"] in ids:
                for label_idx, label in zip(sample["label_ids"], sample["labels"]):
                    self.samples.append({
                        "idx": sample["idx"],
                        "text_idx": sample["text_idx"],
                        "text": sample["text"],
                        "label_idx": label_idx,
                        "label": label
                    })

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            return pickle.load(ids_file)

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
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )
