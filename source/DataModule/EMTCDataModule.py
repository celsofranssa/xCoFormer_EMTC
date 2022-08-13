import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm

from source.Dataset.EvalDataset import EvalDataset
from source.Dataset.FitDataset import FitDataset
from source.Dataset.LabelDataset import LabelDataset
from source.Dataset.TextDataset import TextDataset


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
            self.train_dataset = FitDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/train.pkl",
                text_tokenizer=self.text_tokenizer,
                label_tokenizer=self.label_tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

            self.val_dataset = FitDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/val.pkl",
                text_tokenizer=self.text_tokenizer,
                label_tokenizer=self.label_tokenizer,
                text_max_length=self.params.text_max_length,
                label_max_length=self.params.label_max_length
            )

        if stage == 'test' or stage is "predict":
            self.text_dataset = TextDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/test.pkl",
                tokenizer=self.text_tokenizer,
                max_length=self.params.text_max_length
            )

            self.label_dataset = LabelDataset(
                samples=self.samples,
                ids_path=self.params.dir + f"fold_{self.fold}/test.pkl",
                tokenizer=self.label_tokenizer,
                max_length=self.params.label_max_length
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
            shuffle=False,
            num_workers=self.params.num_workers
        )

    def predict_dataloader(self):
        return [
            DataLoader(self.text_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers),
            DataLoader(self.label_dataset, batch_size=self.params.batch_size, num_workers=self.params.num_workers)
        ]
