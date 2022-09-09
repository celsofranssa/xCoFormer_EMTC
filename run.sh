# activate venv and set Python path
source ~/projects/venvs/xCoFormer_EMTC/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer_EMTC/

# BERT_LC_TGT EURLEX57K
python main.py \
  tasks=[fit] \
  model=BERT_TGT \
  data=EURLEX57K \
  data.batch_size=64 \
  data.folds=[0]