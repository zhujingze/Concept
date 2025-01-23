CUDA_VISIBLE_DEVICES=7 python train.py \
    --model 'Llama-2-7b-hf' \
    --device 'cuda' \
    --data_folder 'data' \
    --subject 'global_facts' \
    --bs 4 \
    --lr 1e-3 \
    --epoch 20\
    --method 'letter'\
    --save_folder "layer_weight"\
