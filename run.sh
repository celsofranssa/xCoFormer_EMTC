# activate venv and set Python path
source ~/projects/venvs/xCoFormer_EMTC/bin/activate
export PYTHONPATH=$PATHONPATH:~/projects/xCoFormer_EMTC/

# BERT Wiki10-31k
python main.py \
  tasks=[fit,predict,eval] \
  trainer.precision=16 \
  model=SiBERT \
  data=Eurlex-PSD-4k \
  data.batch_size=64 \
  data.folds=[0]
