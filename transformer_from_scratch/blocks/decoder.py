import torch
import torch.nn as nn

from transformer_from_scratch.embedding.transformer_embedding import TransformerEmbedding
from transformer_from_scratch.blocks.decoder_block import DecoderBlock


class Decoder(nn.Module):

    def __init__(
            self,
            trg_pad_idx,
            trg_vocab_size,
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
            pad_idx=trg_pad_idx,
            vocab_size=trg_vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            dropout=dropout,
            device=device
        )

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_heads=n_heads,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.linear = nn.Linear(d_model, trg_vocab_size)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        output = self.linear(trg)

        return output