import torch
import torch.nn as nn

from transformer_from_scratch.blocks.encoder_block import EncoderBlock


class Encoder(nn.Module):

    def __init__(
            self,
            d_model,
            d_ffn,
            n_heads,
            n_layers,
            dropout=0.1,
    ):
        """
        Args:
            d_model:      dimension of embeddings
            d_ffn:        dimension of feed-forward network
            n_heads:      number of heads
            n_layers:     number of encoder layers
            dropout:      probability of dropout occurring
        """
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    n_heads=n_heads,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src, src_mask):
        """
        Args:
            src: embedded sequences (batch_size, seq_len, d_model)
            src_mask:  mask for the sequences (batch_size, 1, 1, seq_len)

        Returns:
            enc_out:  sequences after self-attention (batch_size, seq_len, d_model)
        """

        for layer in self.layers:
            enc_out, attention_probs = layer(src, src_mask)

        return enc_out
