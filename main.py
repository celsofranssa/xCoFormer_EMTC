import hydra
import os
from omegaconf import OmegaConf
from source.helper.EvalHelper import EvalHelper
from source.helper.FitHelper import FitHelper
from source.helper.PredictHelper import PredictHelper
from source.helper.RerankerEvalHelper import RerankerEvalHelper
from source.helper.RerankerFitHelper import RerankerFitHelper
from source.helper.RerankerPredictHelper import RerankerPredictHelper
from source.helper.SiFitHelper import SiFitHelper
from source.helper.SiPredictHelper import SiPredictHelper


def fit(params):
    if params.model.type == "reranker":
        fit_helper = RerankerFitHelper(params)
        fit_helper.perform_fit()
    elif params.model.type == "single":
        fit_helper = SiFitHelper(params)
        fit_helper.perform_fit()


def predict(params):
    if params.model.type == "reranker":
        predict_helper = RerankerPredictHelper(params)
        predict_helper.perform_predict()
    elif params.model.type == "single":
        predict_helper = SiPredictHelper(params)
        predict_helper.perform_predict()


def eval(params):
    if params.model.type == "reranker":
        eval_helper = RerankerEvalHelper(params)
        eval_helper.perform_eval()
    elif params.model.type == "single":
        eval_helper = RerankerEvalHelper(params)
        eval_helper.perform_eval()



@hydra.main(config_path="settings", config_name="settings.yaml", version_base=None)
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)
    if "fit" in params.tasks:
        fit(params)

    if "predict" in params.tasks:
        predict(params)

    if "eval" in params.tasks:
        eval(params)


if __name__ == '__main__':
    perform_tasks()
