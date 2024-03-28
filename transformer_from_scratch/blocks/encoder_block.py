import torch
import torch.nn as nn

from transformer_from_scratch.layers.layer_norm import LayerNorm
from transformer_from_scratch.layers.multi_head_attention import MultiHeadAttention
from transformer_from_scratch.layers.position_wise_feed_forward import PositionWiseFeedForward


class EncoderBlock(nn.Module):

    def __init__(self, d_model, d_ffn, n_heads, dropout):
        """
        Args:
            d_model: dimension of embeddings
            n_heads: number of heads
            d_ffn: dimension of feed-forward network
            dropout: probability of dropout occurring
        """
        super(EncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = LayerNorm(d_model=d_model)

        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ffn=d_ffn, dropout=dropout)
        self.norm2 = LayerNorm(d_model=d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, src, src_mask):
        """
        Args:
            src: positionally embedded sequences (batch_size, seq_len, d_model)
            src_mask: mask for the sequences (batch_size, 1, 1, seq_len)
        Returns:
            out: sequences after self-attention (batch_size, seq_len, d_model)
        """
        # 1. pass embeddings through multi-head attention
        attention_out, attention_probs = self.attention(q=src, k=src, v=src, mask=src_mask)

        # 2. residual add and norm
        norm1_out = self.norm1(src + self.dropout(attention_out))

        # 3. position-wise feed forward network
        ffn_out = self.ffn(norm1_out)

        # 4. residual add and norm
        out = self.norm2(norm1_out + self.dropout(ffn_out))

        return out, attention_probs







