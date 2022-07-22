# activate venv and set Python path
source ~/projects/venvs/xCoFormer_EMTC/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer_EMTC/

# BERT_TGT EURLEX57K
python main.py \
  tasks=[predict] \
  model=BERT_TGT \
  data=EURLEX57K \
  data.folds=[0]

