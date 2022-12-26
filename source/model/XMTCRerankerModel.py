import torch
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule
from torchmetrics import MetricCollection, F1Score
from transformers import get_linear_schedule_with_warmup

from source.pooling.NoPooling import NoPooling


class XMTCRerankerModel(LightningModule):
    """Encodes the code and qs1 and qs2."""

    def __init__(self, hparams):
        super(XMTCRerankerModel, self).__init__()

        self.save_hyperparameters(hparams)

        self.encoder = instantiate(hparams.encoder)

        # pooling
        self.pool = NoPooling()

        self.dropout = torch.nn.Dropout(hparams.dropout)

        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(hparams.hidden_size, self.hparams.num_classes),
            # torch.nn.LogSoftmax(dim=-1)
        )

        # loss
        self.loss = torch.nn.CrossEntropyLoss()

        # metric val_Wei-F1
        self.val_metrics = self.get_metrics(prefix="val_")

    def get_metrics(self, prefix):
        return MetricCollection(
            metrics={
                "Mac-F1": F1Score(num_classes=self.hparams.num_classes, average="macro"),
                "Wei-F1": F1Score(num_classes=self.hparams.num_classes, average="weighted")
            },
            prefix=prefix)


    def forward(self, features):
        rpr = self.pool(self.encoder(features))
        return self.cls_head(rpr)[:, -1]

    def training_step(self, features, batch_idx):
        true_cls = features["cls"]
        rpr = self.pool(self.encoder(features))
        pred_cls = self.cls_head(
            self.dropout(rpr)
        )
        # log training loss
        train_loss = self.loss(pred_cls, true_cls)
        self.log('train_Loss', train_loss)
        return train_loss

    def validation_step(self, features, batch_idx):
        true_cls = features["cls"]
        rpr = self.pool(self.encoder(features))
        pred_cls = self.cls_head(
            self.dropout(rpr)
        )

        pred_cls = torch.argmax(pred_cls, dim=-1)

        self.log_dict(self.val_metrics(pred_cls, true_cls), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.val_metrics.compute()


    # def configure_optimizers(self):
    #     # optimizer
    #     optimizer = torch.optim.AdamW(
    #         self.parameters(),
    #         lr=self.hparams.lr,
    #         weight_decay=self.hparams.weight_decay,
    #         amsgrad=True)
    #
    #     # scheduler
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer=optimizer,
    #         num_warmup_steps=round(0.03 * self.trainer.estimated_stepping_batches),
    #         num_training_steps=self.trainer.estimated_stepping_batches
    #     )
    #
    #     return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        # schedulers
        step_size_up = round(0.07 * self.trainer.estimated_stepping_batches)

        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, mode='triangular2',
                                                      base_lr=self.hparams.base_lr,
                                                      max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                                      cycle_momentum=False)

        return (
            {"optimizer": optimizer,
             "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}},
        )

