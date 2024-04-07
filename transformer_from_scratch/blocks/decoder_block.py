import torch
import torch.nn as nn

from transformer_from_scratch.layers.layer_norm import LayerNorm
from transformer_from_scratch.layers.multi_head_attention import MultiHeadAttention
from transformer_from_scratch.layers.position_wise_feed_forward import PositionWiseFeedForward


class DecoderBlock(nn.Module):

    def __init__(self, d_model, d_ffn, n_heads, dropout):
        """
        Args:
            d_model: dimension of embeddings
            d_ffn: dimension of feed-forward network
            n_heads: number of heads
            dropout: probability of dropout occurring
    """
        super(DecoderBlock, self).__init__()

        self.masked_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = LayerNorm(d_model=d_model)

        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm2 = LayerNorm(d_model=d_model)

        self.ffn = PositionWiseFeedForward(d_model=d_model, d_ffn=d_ffn, dropout=dropout)
        self.norm3 = LayerNorm(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, trg, src, trg_mask, src_mask):
        """
        Args:
            trg: embedded sequences (batch_size, trg_seq_len, d_model)
            src: embedded sequences (batch_size, src_seq_len, d_model)
            trg_mask: mask for the sequences (batch_size, 1, trg_seq_len, trg_seq_len)
            src_mask: mask for the sequences (batch_size, 1, 1, src_seq_len)

        Returns:
            trg: sequences after self-attention (batch_size, trg_seq_len, d_model)
            attention_probs: self-attention softmax scores (batch_size, n_heads, trg_seq_len, src_seq_len)
        """
        # 1. compute self attention
        attention_out, masked_attention_probs = self.masked_attention(q=trg, k=trg, v=trg, mask=trg_mask)

        # 2. residual add and norm
        norm1_out = self.norm1(trg + self.dropout(attention_out))
        trg = norm1_out

        # 3. compute encoder-decoder attention
        enc_dec_attention_out, attention_probs = self.attention(q=trg, k=src, v=src, mask=src_mask)

        # 4. add and norm
        norm2_out = self.norm2(trg + self.dropout(enc_dec_attention_out))

        # 5. position-wise feed forward network
        ffn_out = self.ffn(norm2_out)

        # 6. residual add and norm
        out = self.norm3(norm2_out + self.dropout(ffn_out))

        return out, masked_attention_probs, attention_probs