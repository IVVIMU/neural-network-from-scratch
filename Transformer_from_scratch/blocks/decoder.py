import torch
import torch.nn as nn

from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention
from layers.position_wise_feed_forward import PositionWiseFeedForward


class Decoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_heads, dropout):
        super(Decoder, self).__init__()

        self.attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=dropout)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=dropout)

        self.ffn = PositionWiseFeedForward(d_model=d_model, hidden=ffn_hidden, dropout=dropout)
        self.norm3 = self.LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, dec, enc, trg_mask, src_mask):
        # 1. compute self attention
        dec_attention_out = self.attention(q=dec, k=dec, v=dec, mask=trg_mask)

        # 2. add and norm
        norm1_out = self.dropout1(self.norm1(dec_attention_out + dec))

        # 3. compute encoder-decoder attention
        enc_dec_attention_out = self.enc_dec_attention(q=norm1_out, k=enc, v=enc, mask=src_mask)

        # 4. add and norm
        norm2_out = self.dropout2(self.norm2(enc_dec_attention_out + norm1_out))

        # 5. position-wise feed forward network
        ffn_out = self.ffn(norm2_out)

        # 6. add and norm
        norm3_out = self.dropout3(self.norm3(ffn_out + norm2_out))

        return norm3_out