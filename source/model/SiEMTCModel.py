import torch
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule
from transformers import get_constant_schedule_with_warmup, get_scheduler

from source.metric.MRRMetric import MRRMetric
from source.pooling.LabelMaxPooling import LabelMaxPooling
from source.pooling.NoPooling import NoPooling


class SiEMTCModel(LightningModule):
    """Encodes the text and label into an same space of embeddings."""

    def __init__(self, hparams):

        super(SiEMTCModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.encoder = instantiate(hparams.encoder)

        # pooling
        self.text_pool = NoPooling()
        self.label_pool = LabelMaxPooling()

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = MRRMetric(hparams.metric)

    def forward(self, text, labels, labels_mask):
        text_rpr = self.encoder(text)
        labels_rpr = torch.reshape(
            self.label_encoder(labels, labels_mask),
            (labels.shape[0] * self.hparams.max_labels, self.hparams.hidden_size)
        )
        return text_rpr, labels_rpr

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        text_idx, text, labels_ids, labels, labels_mask = batch["text_idx"], batch["text"], batch["labels_ids"], batch[
            "labels"], batch["labels_mask"]
        text_rpr = self.text_pool(self.encoder(text))
        encoded_labels = self.encoder(labels)
        e = self.label_pool(encoded_labels, labels_mask)
        labels_rpr = torch.reshape(
            e,
            (labels.shape[0] * self.hparams.max_labels, self.hparams.hidden_size)
        )

        train_loss = self.loss(text_idx, text_rpr, labels_ids, labels_rpr)

        # log training loss
        self.log('train_LOSS', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        text_idx, text, labels_ids, labels, labels_mask = batch["text_idx"], batch["text"], batch["labels_ids"], batch[
            "labels"], batch["labels_mask"]
        text_rpr = self.text_pool(self.encoder(text))
        encoded_labels = self.encoder(labels)
        e = self.label_pool(encoded_labels, labels_mask)
        labels_rpr = torch.reshape(
            e,
            (labels.shape[0] * self.hparams.max_labels, self.hparams.hidden_size)
        )
        self.mrr.update(text_idx, text_rpr, labels_ids, labels_rpr)

    def validation_epoch_end(self, outs):
        self.log("val_MRR", self.mrr.compute(), prog_bar=True)
        self.mrr.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        text_idx, text, labels_ids, labels, labels_mask = batch["text_idx"], batch["text"], batch["labels_ids"], batch[
            "labels"], batch["labels_mask"]
        text_rpr = self.text_pool(self.encoder(text))
        encoded_labels = self.encoder(labels)
        e = self.label_pool(encoded_labels, labels_mask)
        labels_rpr = torch.reshape(
            e,
            (labels.shape[0] * self.hparams.max_labels, self.hparams.hidden_size)
        )

        return {
            "text_idx": text_idx,
            "text_rpr": text_rpr,
            "labels_ids": torch.flatten(labels_ids),
            "labels_rpr": labels_rpr
        }



    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        # schedulers
        step_size_up = round(0.07 * self.trainer.estimated_stepping_batches)

        # scheduler = get_scheduler(
        #     "linear",
        #     optimizer=optimizer,
        #     num_warmup_steps=0,
        #     num_training_steps=self.trainer.estimated_stepping_batches
        # )

        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular2',
                                                      base_lr=self.hparams.base_lr,
                                                      max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                                      cycle_momentum=False)

        return (
            {"optimizer": optimizer,
             "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}},
        )

