# activate venv and set Python path
source ~/projects/venvs/xCoFormer_EMTC/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer_EMTC/

# BERT_TGT Wiki10-31k
python main.py \
  tasks=[predict,eval] \
  model=BERT_TGT \
  data=Wiki10-31k \
  data.batch_size=64 \
  data.folds=[0]