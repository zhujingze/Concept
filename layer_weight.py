import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, LlamaModel, CausalLMOutputWithPast
import numpy as np

class LlamaWithLayerWeights(LlamaForCausalLM):
    def __init__(self, config):
        super(LlamaWithLayerWeights, self).__init__(config)
        
        # 定义一个32维的可训练参数，表示每一层的权重
        layer_weights_init = torch.zeros(config.num_hidden_layers)
        layer_weights_init[-1] = 1
        self.layer_weights = nn.Parameter(layer_weights_init)  # 最后一层权重初始化为1，其余为0
        
        self.post_init()

    def get_relative_top_filter(self, scores: torch.FloatTensor, relative_top, min_tokens_to_keep: int = 1):
        scores_normalized = scores.log_softmax(dim=-1) 
        sorted_logits, sorted_indices = torch.sort(scores_normalized, descending=True)
        min_thresh = sorted_logits[..., min_tokens_to_keep-1] 
        probs_max = torch.max(scores_normalized, dim=-1).values
        probs_thresh = probs_max + np.log(relative_top)
        probs_thresh = torch.min(min_thresh, probs_thresh)
        probs_thresh = probs_thresh.unsqueeze(-1)
        return scores_normalized < probs_thresh

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        relative_top: Optional[float] = None,
        relative_top_value: Optional[float] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        relative_top = relative_top if relative_top is not None else 0.0001
        relative_top_value = relative_top_value if relative_top_value is not None else -1000.0
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 获取所有层的 hidden states
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states_last = outputs[0]
        logits_last = self.lm_head(hidden_states_last[:, -num_logits_to_keep:, :])

        # 获取每一层的 hidden states
        hidden_states_all = outputs.hidden_states
        # 限制大小在[0,1]并转换为和为1的权重
        # constrained_layer_weights = torch.clamp(self.layer_weights, min=0.0, max=1.0)
        # constrained_layer_weights = constrained_layer_weights / constrained_layer_weights.abs().sum()
        # 去除输入embedding
        hidden_states_out=hidden_states_all[1:]
        weighted_hidden_states = torch.zeros_like(hidden_states_out[0])
        device = hidden_states_out[0].device
        logits = torch.zeros(weighted_hidden_states.size(0), weighted_hidden_states.size(1), self.config.vocab_size, device=device)

        for layer_idx in range(len(hidden_states_out)):
            weight = self.layer_weights[layer_idx]
            layer_logits = self.lm_head(hidden_states_out[layer_idx]).log_softmax(dim=-1)
            layer_logits = layer_logits * weight
            logits += layer_logits
        logits = logits.log_softmax(dim=-1)

        if relative_top > 0.0:
            relative_top_mask = self.get_relative_top_filter(logits_last, relative_top)
            logits = torch.where(relative_top_mask, relative_top_value, logits)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
