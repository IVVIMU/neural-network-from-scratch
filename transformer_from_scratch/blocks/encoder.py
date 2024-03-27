import torch
import torch.nn as nn

from transformer_from_scratch.embedding.transformer_embedding import TransformerEmbedding
from transformer_from_scratch.blocks.encoder_block import EncoderBlock


class Encoder(nn.Module):

    def __init__(
            self,
            src_pad_idx,
            src_vocab_size,
            max_seq_len,
            d_model,
            ffn_hidden,
            n_heads,
            n_layers,
            dropout,
            device
    ):
        super().__init__()

        self.emb = TransformerEmbedding(
            pad_idx=src_pad_idx,
            vocab_size=src_vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            dropout=dropout,
            device=device
        )

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_heads=n_heads,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
