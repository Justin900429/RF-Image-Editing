from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention


class FluxAttnProcessor2_0WithMemory:
    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "FluxAttnProcessor2_0WithMemory requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        self.first_order_memory = {}
        self.second_order_memory = {}

    def check_memory(self, timestep: int, second_order: bool) -> Optional[torch.Tensor]:
        if second_order:
            return timestep in self.second_order_memory
        else:
            return timestep in self.first_order_memory

    def pop_memory(self, timestep: int, second_order: bool) -> Optional[torch.Tensor]:
        if second_order:
            return self.second_order_memory.pop(timestep, None)
        else:
            return self.first_order_memory.pop(timestep, None)

    def get_memory(self, timestep: int, second_order: bool) -> Optional[torch.Tensor]:
        if second_order:
            return self.second_order_memory.get(timestep, None)
        else:
            return self.first_order_memory.get(timestep, None)

    def set_memory(self, memory: torch.Tensor, timestep: int, second_order: bool):
        if second_order:
            self.second_order_memory[timestep] = memory.to("cpu").clone()
        else:
            self.first_order_memory[timestep] = memory.to("cpu").clone()

    def clear_memory(self):
        self.first_order_memory.clear()
        self.second_order_memory.clear()

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        second_order: bool = False,
        timestep: int = -1,
        inject: bool = False,
    ) -> torch.FloatTensor:
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)

        # For sharing value to insert similar features
        if inject and self.check_memory(timestep, second_order):
            value = self.get_memory(timestep, second_order).to(hidden_states.device)
        else:
            value = attn.to_v(hidden_states)
            if inject:
                self.set_memory(value, timestep, second_order)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from diffusers.models.embeddings import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
