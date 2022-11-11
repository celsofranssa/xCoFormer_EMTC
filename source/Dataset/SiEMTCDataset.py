import pickle

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SiEMTCDataset(Dataset):
    """Fit Dataset.
    """

    def __init__(self, samples, ids_path, tokenizer, text_max_length, labels_max_length,
                 max_labels, pseudo_labels):
        super(SiEMTCDataset, self).__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.labels_max_length = labels_max_length
        self.max_labels = max_labels
        self.pseudo_labels = pseudo_labels
        self._load_ids(ids_path)

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

    def _create_label_mask(self, start, end):
        mask = [0] * self.labels_max_length
        for pst in range(start, end):
            if pst < self.labels_max_length:
                mask[pst] = 1
        return mask

    def _encode_labels(self, labels):
        tokens = ["[CLS]"]
        labels_mask = [[0] * self.labels_max_length]
        labels_offset = 1

        for label in labels:
            tokens.extend(self.tokenizer.tokenize(label))
            labels_mask.append(
                self._create_label_mask(labels_offset, len(tokens))
            )
            labels_offset = len(tokens)

        if len(labels_mask) < self.max_labels:
            labels_mask += [[0] * self.labels_max_length] * (self.max_labels - len(labels_mask))

        # print(f"labels_mask ({len(labels_mask)}")

        tokens.append("[SEP]")
        token_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in tokens]
        if len(token_ids) <= self.labels_max_length:
            token_ids += [0] * (self.labels_max_length - len(token_ids))
        else:
            token_ids = token_ids[:self.labels_max_length]

        return token_ids, labels_mask

    def _encode(self, sample):
        if self.pseudo_labels:
            # print(f"\n\nL: {len(sample['labels'])}")
            # print(f"PL: {len(sample['pseudo_labels'])}")
            sample["labels"].extend(sample["pseudo_labels"])
            # print(f"L+ PL: {len(sample['labels'])}")
            # print(f"T(L+ PL): {len(sample['labels'][:self.max_labels])}\n")
            token_ids, labels_mask = self._encode_labels(sample["labels"][:self.max_labels-1])
        else:
            token_ids, labels_mask = self._encode_labels(sample["labels"][:self.max_labels-1])

        # print(f"\ntoken_ids ({len(token_ids)})\n: {token_ids}")
        # print(f"\nlabels_mask ({len(labels_mask)})\n: {labels_mask}")

        return {
            "text_idx": sample["text_idx"],
            "text": torch.tensor(
                self.tokenizer.encode(
                    text=sample["text"], max_length=self.text_max_length, padding="max_length", truncation=True
                )),
            "labels_ids": torch.tensor(
                [-1] + sample["labels_ids"] + [-1] * (self.max_labels - len(sample["labels_ids"]) - 1)
            ),
            "labels": torch.tensor(token_ids),
            "labels_mask": torch.tensor(labels_mask)
        }
        # if a["labels_mask"].shape[0] > 32:
        #     print(sample["idx"])
        # return a

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[
                self.ids[idx]
            ]
        )
