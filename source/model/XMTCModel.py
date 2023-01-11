import torch
from hydra.utils import instantiate
from pytorch_lightning.core.lightning import LightningModule
from source.metric.MRRMetric import MRRMetric
from source.pooling.HiddenStatePooling import HiddenStatePooling


class XMTCModel(LightningModule):
    """Encodes the text and label into an same space of embeddings."""

    def __init__(self, hparams):
        super(XMTCModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.encoder = instantiate(hparams.encoder)

        # pooling
        self.pool = HiddenStatePooling()

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = MRRMetric(hparams.metric)

    def forward(self, text, label):
        return self.get_score(
            self.pool(self.encoder(text), torch.where(text > 0, 1, 0)),
            self.pool(self.encoder(label), torch.where(label > 0, 1, 0))
        )

    def get_score(self, text_rpr, label_rpr):
        m = torch.einsum('b i j, c k j -> b c i k', text_rpr, label_rpr)
        m = torch.max(m, -1).values.sum(dim=-1)
        return torch.nn.functional.normalize(m, p=2, dim=-1)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        text_rpr = self.pool(self.encoder(batch["text"]), torch.where(batch["text"] > 0, 1, 0))
        label_rpr = self.pool(self.encoder(batch["label"]), torch.where(batch["label"] > 0, 1, 0))

        train_loss = self.loss(batch["text_idx"], text_rpr, batch["label_idx"], label_rpr)

        self.log('train_LOSS', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        text_rpr = self.pool(self.encoder(batch["text"]), torch.where(batch["text"] > 0, 1, 0))
        label_rpr = self.pool(self.encoder(batch["label"]), torch.where(batch["label"] > 0, 1, 0))
        self.mrr.update(batch["text_idx"], text_rpr, batch["label_idx"], label_rpr)

    def validation_epoch_end(self, outs):
        self.log("val_MRR", self.mrr.compute(), prog_bar=True)
        self.mrr.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return {
            "text_idx": batch["text_idx"],
            "label_idx": batch["label_idx"],
            "score": self(batch["text"], batch["label"])
        }

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
