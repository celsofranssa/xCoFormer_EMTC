import importlib

import torch
from pytorch_lightning.core.lightning import LightningModule

from source.metric.MRRMetric import MRRMetric


class JointEncoder(LightningModule):
    """Encodes the code and desc into an same space of embeddings."""

    def __init__(self, hparams):

        super(JointEncoder, self).__init__()
        self.hparams = hparams

        # encoders
        self.x1_encoder = self.get_encoder(hparams.x1_encoder, hparams.x1_encoder_hparams)
        self.x2_encoder = self.get_encoder(hparams.x2_encoder, hparams.x2_encoder_hparams)

        # loss function
        self.loss_fn = self.get_loss(hparams.loss, hparams.loss_hparams)  # MultipleNegativesRankingLoss()

        # metric
        self.mrr = MRRMetric()

    def get_encoder(self, encoder, encoder_hparams):
        encoder_module, encoder_class = encoder.rsplit('.', 1)
        encoder_module = importlib.import_module(encoder_module)
        return getattr(encoder_module, encoder_class)(encoder_hparams)

    def get_loss(self, loss, loss_hparams):
        loss_module, loss_class = loss.rsplit('.', 1)
        loss_module = importlib.import_module(loss_module)
        return getattr(loss_module, loss_class)(loss_hparams)

    def forward(self, x1, x2):
        r1 = self.x1_encoder(x1)
        r2 = self.x2_encoder(x2)
        return r1, r2

    def configure_optimizers(self):
        num_batches = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        print("nummm ", num_batches)
        # optimizers
        optimizers = [
            torch.optim.Adam(self.x1_encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=0, amsgrad=True),
            torch.optim.Adam(self.x2_encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=0, amsgrad=True)
        ]
        # schedulers
        steps = 2000
        schedulers = [
            torch.optim.lr_scheduler.CyclicLR(optimizers[0], mode='triangular2', base_lr=1e-7, max_lr=1e-3,
                                              step_size_up=steps, cycle_momentum=False),
            torch.optim.lr_scheduler.CyclicLR(optimizers[1], mode='triangular2', base_lr=1e-7, max_lr=1e-3,
                                              step_size_up=steps, cycle_momentum=False)
        ]
        return optimizers, schedulers

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu,
                       using_native_amp, using_lbfgs):

        # update x1 opt every even steps
        if optimizer_idx == 0:
            if batch_idx % 2 == 0:
                optimizer.step(closure=optimizer_closure)

        # update x2 opt every odd steps
        if optimizer_idx == 1:
            if batch_idx % 2 != 0:
                optimizer.step(closure=optimizer_closure)

    def training_step(self, batch, batch_idx, optimizer_idx):

        x1, x2 = batch["x1"], batch["x2"]
        r1, r2 = self(x1, x2)
        train_loss = self.loss_fn(r1, r2)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch["x1"], batch["x2"]
        r1, r2 = self(x1, x2)
        self.log("val_mrr", self.mrr(r1, r2), prog_bar=True)
        self.log("val_loss", self.loss_fn(r1, r2), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.log('m_val_mrr', self.mrr.compute())

    def test_step(self, batch, batch_idx):
        id, x1, x2 = batch["id"], batch["x1"], batch["x2"]
        r1, r2 = self(x1, x2)
        self.write_prediction_dict({
            "id": id,
            "r1": r1,
            "r2": r2
        }, self.hparams.predictions.path)
        self.log('test_mrr', self.mrr(r1, r2), prog_bar=True)

    def test_epoch_end(self, outs):
        self.log('m_test_mrr', self.mrr.compute())

    def get_x1_encoder(self):
        return self.x1_encoder

    def get_x2_encoder(self):
        return self.x1_encoder
