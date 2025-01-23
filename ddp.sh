CUDA_VISIBLE_DEVICES=7 python ddp.py \
    --total_epochs 10 \
    --save_every 1 \
    --batch_size 8 \
    --model 'Llama-2-7b' \
    --data_folder 'data' \
    --subject 'math' \
    --lr 1e-3\
    --save_folder 'ckp'\
    --method "letter"\
