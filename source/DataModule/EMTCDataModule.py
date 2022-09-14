import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.Dataset.EMTCDataset import EMTCDataset



class EMTCDataModule(pl.LightningDataModule):
    """
    EMTC DataModule
    """

    def __init__(self, params, text_tokenizer, label_tokenizer, fold):
        super(EMTCDataModule, self).__init__()
        self.params = params
        self.text_tokenizer = text_tokenizer
        self.label_tokenizer = label_tokenizer
        self.fold = fold

    def prepare_data(self):
        self.samples = []
        with open(self.params.dir + f"samples.pkl", "rb") as dataset_file:
            self.samples = pickle.load(dataset_file)

    def setup(self, stage=None):

        if stage == 'fit':
            # samples, ids_path, text_tokenizer, label_tokenizer, text_max_length, labels_max_length,
            #                  max_labels
            self.train_dataset = EMTCDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/train.pkl",
                text_tokenizer=self.text_tokenizer,
                label_tokenizer=self.label_tokenizer,
                text_max_length=self.params.text_max_length,
                labels_max_length=self.params.labels_max_length,
                max_labels=self.params.max_labels
            )

            self.val_dataset = EMTCDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/val.pkl",
                text_tokenizer=self.text_tokenizer,
                label_tokenizer=self.label_tokenizer,
                text_max_length=self.params.text_max_length,
                labels_max_length=self.params.labels_max_length,
                max_labels=self.params.max_labels
            )

        if stage == 'test' or stage == "predict":
            self.predict_dataset = EMTCDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/test.pkl",
                text_tokenizer=self.text_tokenizer,
                label_tokenizer=self.label_tokenizer,
                text_max_length=self.params.text_max_length,
                labels_max_length=self.params.labels_max_length,
                max_labels=self.params.max_labels
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
