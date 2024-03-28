import torch
import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    compute scaled dot-product attention

    Returns:
        attention: multi-head attention values (batch_size, n_heads, seq_len, d_head)
        attention_probs: attention probabilities matrix (batch_size, n_heads, seq_len, seq_len)
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, mask=None):
        # input is 4 dimension tensor (batch_size, n_heads, seq_len, d_head)
        batch_size, n_heads, seq_len, d_head = K.size()

        # 1. dot product Q with K^{T} to compute similarity
        K_t = K.transpose(2, 3)  # (batch_size, n_heads, d_head, seq_len)
        # Q @ K_t -> (batch_size, n_heads, seq_len, seq_len)
        scaled_dot_prod = (Q @ K_t) / math.sqrt(d_head)  # scaled dot-product

        # 2. apply masking (opt)
        if mask is not None:
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask == 0, -1e10)

        # 3. pass them softmax to make (0, 1) range
        attention_probs = torch.softmax(scaled_dot_prod, dim=-1)

        # 4. multiply with Value
        # attention_probs @ V -> (batch_size, n_heads, seq_len, d_head)
        attention = attention_probs @ V

        return attention, attention_probs