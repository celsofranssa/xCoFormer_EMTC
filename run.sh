# activate venv and set Python path
source ~/projects/venvs/xCoFormer_EMTC/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer_EMTC/

# BERT Wiki10-31k
python main.py \
  tasks=[fit,predict,eval] \
  trainer.precision=16 \
  model=Si_LRG_BERT \
  data=Wiki10-31k \
  data.batch_size=32 \
  data.folds=[0]

# BERT Wiki10-31k
python main.py \
  tasks=[fit,predict,eval] \
  trainer.precision=16 \
  model=Si_LRG_BERT \
  data=Eurlex-4k \
  data.batch_size=32 \
  data.folds=[0]

