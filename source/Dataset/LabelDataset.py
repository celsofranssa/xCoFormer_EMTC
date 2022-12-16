import math
import pickle
import random
import torch
from torch.utils.data import Dataset


class LabelDataset(Dataset):
    """Label Dataset.
    """

    def __init__(self, samples, pseudo_labels, ids_path, tokenizer, labels_max_length, max_labels):
        super(LabelDataset, self).__init__()
        self.labels = []
        self.pseudo_labels = pseudo_labels
        self.tokenizer = tokenizer
        self.labels_max_length = labels_max_length
        self.max_labels = max_labels
        self._load_ids(ids_path)

        # with open("resource/dataset/Wiki10-31k/labels_map.pkl", "rb") as f:
        #     labels_map = pickle.load(f)
        #
        # with open("resource/dataset/Wiki10-31k/neighborhood.pkl", "rb") as f:
        #     neighborhood = pickle.load(f)
        #
        # for label_idx, neighbors in neighborhood.items():
        #     labels_ids = [label_idx]
        #     labels_ids.extend(random.choices([label for (label, _) in neighbors],
        #                    [weight for (_, weight) in neighbors],
        #                    k=8 ))
        #     labels = [labels_map[label_idx] for label_idx in labels_ids]
        #     self.labels.append({
        #         "labels_ids": labels_ids,
        #         "labels": labels
        #     })

        for sample in samples:
            if sample["idx"] in self.ids:
                self.labels.append({
                    "labels_ids": sample["labels_ids"],
                    "labels": sample["labels"]
                })

    def _get_pseudo_labels(self, labels_ids):
        pseudo_labels = []
        for label_idx in labels_ids:
            if label_idx in self.pseudo_labels and len(self.pseudo_labels[label_idx]) > 0:
                pseudo_labels.extend(
                    random.choices([label for (label, _) in self.pseudo_labels[label_idx]],
                                   [weight for (_, weight) in self.pseudo_labels[label_idx]],
                                   k=math.ceil((self.max_labels - len(labels_ids)) / len(labels_ids))
                                   )
                )
        return pseudo_labels

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
        sample["labels"].extend(self._get_pseudo_labels(sample["labels_ids"]))

        token_ids, labels_mask = self._encode_labels(sample["labels"][:self.max_labels - 1])

        return {
            "labels_ids": torch.tensor(
                [-1] + sample["labels_ids"] + [-1] * (self.max_labels - len(sample["labels_ids"]) - 1)
            ),
            "labels": torch.tensor(token_ids),
            "labels_mask": torch.tensor(labels_mask)
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self._encode(self.labels[idx])
