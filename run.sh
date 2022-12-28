# activate venv and set Python path
source ~/projects/venvs/xCoFormer_EMTC/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer_EMTC/

python main.py \
  tasks=[predict] \
  trainer.precision=16 \
  model=RerankerBERT \
  data=Wiki10-31k \
  data.folds=[0] \
  ranking.name=BM25_Wiki10-31k \
  eval.label_cls=["all"]
