import math
import pickle
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RerankerPredictDataset(Dataset):
    """Fit Dataset.
    """

    def __init__(self, samples, rankings, pseudo_labels, tokenizer, text_max_length, label_max_length):
        super(RerankerPredictDataset, self).__init__()
        self.samples = []
        self.pseudo_labels = pseudo_labels
        self.tokenizer = tokenizer
        self.max_length = text_max_length + label_max_length
        texts = {}
        labels = {}

        for sample in tqdm(samples, desc="Reading samples"):
            texts[sample["text_idx"]] = sample["text"]
            for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                labels[label_idx] = label

        ranking = rankings["all"]
        for text_idx, labels_scores in tqdm(ranking.items(), desc="Reading ranking"):
            text_idx = int(text_idx.split("_")[-1])
            for label_idx, score in labels_scores.items():
                label_idx = int(label_idx.split("_")[-1])
                self.samples.append({
                    "text_idx": text_idx,
                    "text": texts[text_idx],
                    "label_idx": label_idx,
                    "label": labels[label_idx],
                    "cls": score
                })

    def _encode(self, sample):
        features = self.tokenizer(text=sample["text"], text_pair=sample["label"], max_length=self.max_length,
                                  padding="max_length", truncation=True)

        features["input_ids"] = torch.tensor(features["input_ids"])
        features["attention_mask"] = torch.tensor(features["attention_mask"])
        features["token_type_ids"] = torch.tensor(features["token_type_ids"])
        features["text_idx"] = sample["text_idx"]
        features["label_idx"] = sample["label_idx"]
        features["cls"] = sample["cls"]

        return features

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )
