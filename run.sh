# activate venv and set Python path
source ~/projects/venvs/xCoFormer_EMTC/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer_EMTC/

# BERT_LC_TGT EURLEX57K
python main.py \
  tasks=[fit] \
  model=BERT_TGT \
  data=Wiki10-31k \
  trainer.max_epochs=2 \
  data.text_max_length=16 \
  data.label_max_length=8
  data.batch_size=64 \
  data.folds=[0]

