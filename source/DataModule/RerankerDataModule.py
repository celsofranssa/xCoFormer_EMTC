import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.Dataset.RerankerFitDataset import RerankerFitDataset
from source.Dataset.RerankerPredictDataset import RerankerPredictDataset


class RerankerDataModule(pl.LightningDataModule):

    def __init__(self, params, tokenizer, ranking, fold_idx):
        super(RerankerDataModule, self).__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.ranking = ranking
        self.fold_idx = fold_idx

    def prepare_data(self):
        with open(f"{self.params.dir}samples.pkl", "rb") as samples_file:
            self.samples = pickle.load(samples_file)

        with open(f"{self.params.dir}fold_{self.fold_idx}/pseudo_labels.pkl", "rb") as pseudo_labels_file:
            self.pseudo_labels = pickle.load(pseudo_labels_file)

    def setup(self, stage=None):

        if stage == 'fit':
            self.train_dataset = RerankerFitDataset(
                samples=self.samples,
                pseudo_labels=self.pseudo_labels,
                ids_paths=[self.params.dir + f"fold_{self.fold_idx}/train.pkl"],
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

            self.val_dataset = RerankerFitDataset(
                samples=self.samples,
                pseudo_labels=self.pseudo_labels,
                ids_paths=[self.params.dir + f"fold_{self.fold_idx}/val.pkl"],
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

        if stage == 'test' or stage == "predict":
            self.predict_dataset = RerankerPredictDataset(
                samples=self.samples,
                rankings=self.ranking[self.fold_idx],
                pseudo_labels=self.pseudo_labels,
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )

