from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from configuration_qwen2_5_vl import Qwen2_5_VLConfig

import time

import torch.nn.functional as F

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import nvtx

from .attention import Qwen2_5_VLAttention,Qwen2_5_VLSdpaAttention
from .mlp import Qwen2MLP

from transformers.utils import logging

from utils.RMSNorm import Qwen2RMSNorm
from .kv_cache import CPUKVCache

logger = logging.get_logger(__name__)

import torch.cuda.nvtx as nvtx



QWEN2_5_VL_ATTENTION_CLASSES = {
    "eager": Qwen2_5_VLAttention,
    "sdpa": Qwen2_5_VLSdpaAttention,
}


class Qwen2_5_VLDecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5_VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_5_VL_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False, 
        cpu_kv_cache: Optional[CPUKVCache] = None,     #new
        is_prefill: Optional[bool] = None,
        layer_idx: Optional[int] = None,
        image_len: Optional[int] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        # torch.cuda.synchronize()
        # layer_start = time.time()

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # torch.cuda.synchronize()
        # attn_start = time.time()
        # Self Attention
        # nvtx.range_push("My ATTN Layer")
        hidden_states, self_attn_weights, present_key_value= self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cpu_kv_cache = cpu_kv_cache,     #new
            is_prefill = is_prefill,
            layer_idx = layer_idx,
            image_len = image_len,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        # nvtx.range_pop()
        # torch.cuda.synchronize()
        # attn_end = time.time()
        # print(f"attn time: {attn_end - attn_start}")


        hidden_states = residual + hidden_states

        # nvtx.range_push("My POST-ATTN Layer")
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # nvtx.range_pop()

        # torch.cuda.synchronize()
        # mlp_start = time.time()
        # nvtx.range_push("My MLP Layer")
        hidden_states = self.mlp(hidden_states)
        # nvtx.range_pop()
        # torch.cuda.synchronize()
        # mlp_end = time.time()
        # print(f"mlp time: {mlp_end - mlp_start}")
        

        hidden_states = residual + hidden_states

        # torch.cuda.synchronize()
        # layer_end = time.time()
        # print(f"layer time: {layer_end - layer_start}")

        outputs = (hidden_states,)

        # 这里我们不保留self attn score 如果需要可以自己hook 不然太浪费空间了
        # 可以直接在attention文件里获取
        if output_attentions:
            # outputs += (self_attn_weights,)
            del self_attn_weights

        if use_cache and cpu_kv_cache is None:
            outputs += (present_key_value,)
        
        if cpu_kv_cache is not None:
            del present_key_value

        # cpu_kv_cache.finalize_append(layer_idx, ev, k_buf, v_buf)

        return outputs
