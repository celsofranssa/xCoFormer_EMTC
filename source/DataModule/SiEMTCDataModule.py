import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.Dataset.LabelDataset import LabelDataset
from source.Dataset.SiEMTCDataset import SiEMTCDataset
from source.Dataset.TextDataset import TextDataset


class SiEMTCDataModule(pl.LightningDataModule):
    """
    EMTC SiEMTCDataModule
    """

    def __init__(self, params, tokenizer, fold):
        super(SiEMTCDataModule, self).__init__()
        self.params = params
        self.tokenizer = tokenizer
        self.fold = fold

    def prepare_data(self):
        with open(f"{self.params.dir}samples.pkl", "rb") as dataset_file:
            self.samples = pickle.load(dataset_file)
        with open(f"{self.params.dir}fold_{self.fold}/pseudo_labels.pkl", "rb") as pseudo_labels_file:
            self.pseudo_labels = pickle.load(pseudo_labels_file)

    def setup(self, stage=None):

        if stage == 'fit':
            self.train_dataset = SiEMTCDataset(
                samples=self.samples,
                pseudo_labels=self.pseudo_labels,
                ids_paths=[self.params.dir + f"fold_{self.fold}/train.pkl"],
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

            self.val_dataset = SiEMTCDataset(
                samples=self.samples,
                pseudo_labels=self.pseudo_labels,
                ids_paths=[self.params.dir + f"fold_{self.fold}/val.pkl"],
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

        if stage == 'test' or stage == "predict":
            self.text_dataset = TextDataset(
                samples=self.samples,
                ids_paths=[self.params.dir + f"fold_{self.fold}/test.pkl"],
                tokenizer=self.tokenizer,
                text_max_length=self.params.text_max_length
            )

            self.label_dataset = LabelDataset(
                samples=self.samples,
                pseudo_labels=self.pseudo_labels,
                ids_paths=[self.params.dir + f"fold_{self.fold}/train.pkl",
                           self.params.dir + f"fold_{self.fold}/val.pkl"
                           ],
                tokenizer=self.tokenizer,
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
        return [DataLoader(self.text_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers),
                DataLoader(self.label_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers),
        ]