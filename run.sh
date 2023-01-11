# activate venv and set Python path
source ~/projects/venvs/xCoFormer_EMTC/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer_EMTC/

python main.py \
  tasks=[fit] \
  trainer.precision=16 \
  model=RerankerBERT \
  data=Wiki10-31k\
  data.folds=[1,2,3,4]