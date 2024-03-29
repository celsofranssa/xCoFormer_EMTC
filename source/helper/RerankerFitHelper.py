from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, TQDMProgressBar
from transformers import AutoTokenizer

from source.DataModule.SiEMTCDataModule import SiEMTCDataModule
from source.DataModule.RerankerDataModule import RerankerDataModule
from source.model.SiEMTCModel import SiEMTCModel
from source.model.XMTCRerankerModel import XMTCRerankerModel


class RerankerFitHelper:

    def __init__(self, params):
        self.params = params

    def perform_fit(self):
        seed_everything(707, workers=True)
        for fold in self.params.data.folds:

            # Initialize a trainer
            trainer = pl.Trainer(
                fast_dev_run=self.params.trainer.fast_dev_run,
                max_epochs=self.params.trainer.max_epochs,
                precision=self.params.trainer.precision,
                gpus=self.params.trainer.gpus,
                progress_bar_refresh_rate=self.params.trainer.progress_bar_refresh_rate,
                logger=self.get_logger(self.params, fold),
                callbacks=[
                    self.get_model_checkpoint_callback(self.params, fold),  # checkpoint_callback
                    self.get_early_stopping_callback(self.params),  # early_stopping_callback
                    self.get_lr_monitor(),
                    self.get_progress_bar_callback()
                ]
            )

            # datamodule
            datamodule = RerankerDataModule(
                self.params.data,
                self.get_tokenizer(self.params.model.tokenizer),
                ranking=None,
                fold_idx=fold)

            # model
            model = XMTCRerankerModel(self.params.model)

            # Train the ⚡ model
            print(
                f"Fitting {self.params.model.name} over {self.params.data.name} (fold {fold}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")
            trainer.fit(
                model=model,
                datamodule=datamodule
            )

    def get_logger(self, params, fold):
        return loggers.TensorBoardLogger(
            save_dir=params.log.dir,
            name=f"{params.model.name}_{params.data.name}_{fold}_exp"
        )

    def get_model_checkpoint_callback(self, params, fold):
        return ModelCheckpoint(
            monitor="val_Wei-F1",
            dirpath=params.model_checkpoint.dir,
            filename=f"{params.model.name}_{params.data.name}_{fold}",
            save_top_k=1,
            save_weights_only=True,
            mode="max"
        )


    def get_early_stopping_callback(self, params):
        return EarlyStopping(
            monitor='val_Wei-F1',
            patience=params.trainer.patience,
            min_delta=params.trainer.min_delta,
            mode='max'
        )

    def get_tokenizer(self, params):
        return AutoTokenizer.from_pretrained(
            params.architecture
        )

    def get_lr_monitor(self):
        return LearningRateMonitor(logging_interval='step')

    def get_progress_bar_callback(self):
        return TQDMProgressBar(
            refresh_rate=self.params.trainer.progress_bar_refresh_rate,
            process_position=0
        )

