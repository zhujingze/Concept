下最新transformers pandas==2.1.4 torch
先把layer_weight的内容添加到/site-packages/transformers/models/llama/modeling_llama.py中LlamaForCausalLM的定义后面 model = LlamaWithLayerWeights.from_pretrained()调用模型

CUDA_VISIBLE_DEVICES=1,2,5,6 torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12345 your_training_script.py --batch_size 64 --learning_rate 0.001 --epochs 20
