import math
import pickle
import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RerankerFitDataset(Dataset):
    """Fit Dataset.
    """

    def __init__(self, samples, pseudo_labels, ids_paths, tokenizer, text_max_length, label_max_length):
        super(RerankerFitDataset, self).__init__()
        self.samples = []
        self.pseudo_labels = pseudo_labels
        self.tokenizer = tokenizer
        self.max_length = text_max_length + label_max_length
        self._load_ids(ids_paths)

        texts = {}
        labels = {}
        labels_ids = []

        for sample_idx in tqdm(self.ids, desc="Reading samples"):
            texts[samples[sample_idx]["text_idx"]] = samples[sample_idx]["text"]
            for label_idx, label in zip(samples[sample_idx]["labels_ids"], samples[sample_idx]["labels"]):
                labels[label_idx] = label
                labels_ids.append(label_idx)

        for sample_idx in tqdm(self.ids, desc="Reshaping data"):
            for label_idx, label in zip(samples[sample_idx]["labels_ids"], samples[sample_idx]["labels"]):
                pos_sample = {
                    "text_idx": samples[sample_idx]["text_idx"],
                    "text": samples[sample_idx]["text"],
                    "label_idx": label_idx,
                    "label": label + " ".join(x[0] for x in pseudo_labels[label_idx]),
                    "cls": 1
                }
                #print(len(pos_sample["label"].split()))
                self.samples.append(pos_sample)
                neg_label_idx = random.choice(labels_ids)
                while neg_label_idx in samples[sample_idx]["labels_ids"]:
                    neg_label_idx = random.choice(labels_ids)
                neg_sample = {
                    "text_idx": samples[sample_idx]["text_idx"],
                    "text": samples[sample_idx]["text"],
                    "label_idx": neg_label_idx,
                    "label": labels[neg_label_idx] + " ".join(x[0] for x in pseudo_labels[neg_label_idx]),
                    "cls": 0
                }
                # print(len(neg_sample["label"].split()))
                self.samples.append(neg_sample)

    def _load_ids(self, ids_paths):
        self.ids = []
        for path in ids_paths:
            with open(path, "rb") as ids_file:
                self.ids.extend(pickle.load(ids_file))

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
