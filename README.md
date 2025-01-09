下最新transformers pandas==2.1.4 torch
先把layer_weight的内容添加到/site-packages/transformers/models/llama/modeling_llama.py中LlamaForCausalLM的定义后面 model = LlamaWithLayerWeights.from_pretrained()调用模型
