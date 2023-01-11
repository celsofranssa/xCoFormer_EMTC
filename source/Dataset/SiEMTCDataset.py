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

        for idx in tqdm(self.ids, desc="Reading samples"):
            for label_idx, label in zip(samples[idx]["labels_ids"], samples[idx]["labels"]):
                self.samples.append({
                    "text_idx": samples[idx]["text_idx"],
                    "text": " ".join(x[0] for x in samples[idx]["keywords"]),  # sample["text"],
                    "label_idx": label_idx,
                    "label": label + " ".join(x[0] for x in pseudo_labels[label_idx])
                })

        #
        # for sample in tqdm(samples, desc="Reading samples"):
        #     if sample["idx"] in self.ids:
        #         for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
        #             a = {
        #                 "text_idx": sample["text_idx"],
        #                 "text": " ".join(x[0] for x in sample["keywords"]), #sample["text"],
        #                 "label_idx": label_idx,
        #                 "label": label + " ".join(x[0] for x in pseudo_labels[label_idx])
        #             }
        #             self.samples.append(a)

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


