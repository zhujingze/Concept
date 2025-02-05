CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8  ddp.py \
    --total_epochs 10 \
    --save_every 1 \
    --batch_size 1 \
    --model 'Llama-2-7b' \
    --data_folder 'data' \
    --subject 'math' \
    --lr 1e-3\
    --save_folder 'ckp'\
    --method "letter"\
