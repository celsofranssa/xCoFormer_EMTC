import torch
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule

from source.metric.mMRRMetric import MRRMetric


class EMTCModel(LightningModule):
    """Encodes the text and label into an same space of embeddings."""

    def __init__(self, hparams):

        super(EMTCModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.text_encoder = instantiate(hparams.text_encoder)
        self.label_encoder = instantiate(hparams.label_encoder)

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = MRRMetric(hparams.metric)

    def forward(self, text, labels):
        text_rpr = self.text_encoder(text)
        labels_rpr = torch.reshape(
            self.label_encoder(labels),
            (self.hparams.batch_size * self.hparams.max_labels, self.hparams.max_lenght)
        )
        return text_rpr, labels_rpr

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        text_idx, text, labels_ids, labels = batch["text_idx"], batch["text"], batch["labels_ids"], batch["labels"]
        text_rpr, labels_rpr = self(text, labels)
        train_loss = self.loss(text_idx, text_rpr, labels_ids, labels_rpr)

        # log training loss
        self.log('train_LOSS', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        text_idx, text, labels_ids, labels = batch["text_idx"], batch["text"], batch["labels_ids"], batch["labels"]
        text_rpr, labels_rpr = self(text, labels)
        self.log("val_MRR", self.mrr(text_idx, text_rpr, labels_ids, labels_rpr), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.mrr.compute()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        text_idx, text, labels_ids, labels = batch["text_idx"], batch["text"], batch["labels_ids"], batch["labels"]
        text_rpr, labels_rpr = self(text, labels)
        return {
            "text_idx": text_idx,
            "text_rpr": text_rpr,
            "labels_ids": labels_ids,
            "labels_rpr": labels_rpr
        }


    def configure_optimizers(self):
        if self.hparams.tag_training:
            return self._configure_tgt_optimizers()
        else:
            return self._configure_std_optimizers()

    def _configure_tgt_optimizers(self):
        # optimizers
        text_optimizer = torch.optim.AdamW(self.text_encoder.parameters(), lr=self.hparams.text_lr, betas=(0.9, 0.999),
                                           eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        label_optimizer = torch.optim.AdamW(self.label_encoder.parameters(), lr=self.hparams.label_lr,
                                            betas=(0.9, 0.999),
                                            eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)
        # schedulers
        step_size_up = round(0.03 * self.trainer.estimated_stepping_batches)


        text_scheduler = torch.optim.lr_scheduler.CyclicLR(text_optimizer, mode='triangular2',
                                                           base_lr=self.hparams.base_lr,
                                                           max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                                           cycle_momentum=False)
        label_scheduler = torch.optim.lr_scheduler.CyclicLR(label_optimizer, mode='triangular2',
                                                            base_lr=self.hparams.base_lr,
                                                            max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                                            cycle_momentum=False)

        return (
            {"optimizer": text_optimizer, "lr_scheduler": text_scheduler, "frequency": self.hparams.text_frequency_opt},
            {"optimizer": label_optimizer, "lr_scheduler": label_scheduler,
             "frequency": self.hparams.label_frequency_opt},
        )

    def _configure_std_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        # schedulers
        step_size_up = round(0.03 * self.num_training_steps)


        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular2',
                                                      base_lr=self.hparams.base_lr,
                                                      max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                                      cycle_momentum=False)

        return (
            {"optimizer": optimizer, "lr_scheduler": scheduler}
        )

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
