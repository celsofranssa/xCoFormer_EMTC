# activate venv and set Python path
source ~/projects/venvs/xCoFormer_EMTC/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer_EMTC/

python main.py \
  tasks=[predict,eval] \
  trainer.precision=16 \
  model=SiBERT \
  data=Wiki10-31k \
  data.folds=[0]
