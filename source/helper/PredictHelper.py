from omegaconf import OmegaConf
import pytorch_lightning as pl
from transformers import AutoTokenizer

from source.DataModule.EMTCDataModule import EMTCDataModule
from source.callback.PredictionWriter import PredictionWriter
from source.model.EMTCModel import EMTCModel


class PredictHelper:
    
    def __init__(self, params):
        self.params=params

    def perform_predict(self):
        for fold in self.params.data.folds:
            # data
            dm = EMTCDataModule(
                self.params.data,
                self.get_tokenizer(self.params.model.text_tokenizer),
                self.get_tokenizer(self.params.model.label_tokenizer),
                fold=fold)

            # model
            model = EMTCModel.load_from_checkpoint(
                checkpoint_path=f"{self.params.model_checkpoint.dir}{self.params.model.name}_{self.params.data.name}_{fold}.ckpt"
            )

            self.params.prediction.fold = fold
            # trainer
            trainer = pl.Trainer(
                gpus=self.params.trainer.gpus,
                callbacks=[PredictionWriter(self.params.prediction)]
            )

            # predicting
            dm.prepare_data()
            dm.setup("predict")

            print(f"Predicting {self.params.model.name} over {self.params.data.name} (fold {fold}) with fowling params\n"
                  f"{OmegaConf.to_yaml(self.params)}\n")
            trainer.predict(
                model=model,
                datamodule=dm,

            )

    def get_tokenizer(self, params):
        tokenizer = AutoTokenizer.from_pretrained(
            params.architecture
        )
        if "gpt" in params.architecture:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            params.pad = tokenizer.convert_tokens_to_ids("[PAD]")
        return tokenizer
