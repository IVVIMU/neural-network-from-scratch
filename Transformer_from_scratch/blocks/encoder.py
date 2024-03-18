import torch
import torch.nn as nn

from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionWiseFeedForward


class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_heads, dropout):
        super(Encoder, self).__init__()

        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, dropout=dropout)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x, src_mask):
        # 1. compute self attention
        attention_out = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        norm1_out = self.dropout1(self.norm1(attention_out + x))

        # 3. position-wise feed forward network
        ffn_out = self.ffn(norm1_out)

        # 4. add and norm
        norm2_out = self.dropout2(self.norm2(ffn_out + norm1_out))

        return norm2_out







