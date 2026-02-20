python train.py \
'data.parts=["body"]' \
data.batch_size=32 \
data.context_len=0 \
data.dataset=modiff-2022-gen \
data.pkl_path=/home/mb230/projects/human-modeling-3d/data/modiff-2022-samp-from-text/walking-example-dataset/results.pkl \
training.start_from_ckpt=false 
