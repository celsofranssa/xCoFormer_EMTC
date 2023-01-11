import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class RankingDataset(Dataset):

    def __init__(self, samples, ranking, pseudo_labels, tokenizer, text_max_length, label_max_length):
        super(RankingDataset, self).__init__()
        self.samples = []
        self.pseudo_labels = pseudo_labels
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.label_max_length = label_max_length
        texts = {}
        labels = {}

        for sample in tqdm(samples, desc="Reading samples"):
            texts[sample["text_idx"]] = " ".join(x[0] for x in sample["keywords"])
            for label_idx, label in zip(sample["labels_ids"], sample["labels"]):
                labels[label_idx] = label + " ".join(x[0] for x in pseudo_labels[label_idx])

        for text_idx, labels_scores in tqdm(ranking["all"].items(), desc="Reading ranking"):
            text_idx = int(text_idx.split("_")[-1])
            for label_idx, score in labels_scores.items():
                label_idx = int(label_idx.split("_")[-1])
                self.samples.append({
                    "text_idx": text_idx,
                    "text": texts[text_idx],
                    "label_idx": label_idx,
                    "label": labels[label_idx],

                })

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
