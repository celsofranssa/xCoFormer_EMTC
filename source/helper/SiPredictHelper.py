import pickle

from omegaconf import OmegaConf
import pytorch_lightning as pl
from transformers import AutoTokenizer

from source.DataModule.SiEMTCDataModule import SiEMTCDataModule

from source.callback.RerankerPredictionWriter import RerankerPredictionWriter

from source.model.XMTCModel import XMTCModel


class SiPredictHelper:

    def __init__(self, params):
        self.params = params

    def perform_predict(self):
        for fold_idx in self.params.data.folds:
            # data
            dm = SiEMTCDataModule(
                self.params.data,
                self.get_tokenizer(self.params.model.tokenizer),
                ranking=self._get_ranking(),
                fold_idx=fold_idx)

            # model
            model = XMTCModel.load_from_checkpoint(
                checkpoint_path=f"{self.params.model_checkpoint.dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.ckpt"
            )

            self.params.prediction.fold_idx = fold_idx
            # trainer
            trainer = pl.Trainer(
                gpus=self.params.trainer.gpus,
                callbacks=[RerankerPredictionWriter(self.params.prediction)]
            )

            # predicting
            dm.prepare_data()
            dm.setup("predict")

            print(
                f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")
            trainer.predict(
                model=model,
                datamodule=dm,

            )

    def get_tokenizer(self, params):
        return AutoTokenizer.from_pretrained(
            params.architecture
        )

    def _get_ranking(self):
        with open(f"{self.params.ranking.dir}{self.params.ranking.name}.rnk", "rb") as ranking_file:
            return pickle.load(ranking_file)
