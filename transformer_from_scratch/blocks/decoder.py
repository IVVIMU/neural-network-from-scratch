import torch
import torch.nn as nn

from transformer_from_scratch.blocks.decoder_block import DecoderBlock


class Decoder(nn.Module):

    def __init__(
      self,
      vocab_size,
      d_model,
      d_ffn,
      n_heads,
      n_layers,
      dropout=0.1      
    ):
        """
        Args:
            vocab_size: size of the target vocabulary
            d_model: dimension of embeddings
            d_ffn: dimension of feed-forward network
            n_heads: number of heads
            n_layers: number of encoder layers
            dropout: probability of dropout occurring
        """
        super().__init__()

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    d_ffn=d_ffn,
                    n_heads=n_heads,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.w_o = nn.Linear(d_model, vocab_size)

    def forward(self, trg, src, trg_mask, src_mask):
        """
        Args:
            trg: embedded sequences (batch_size, trg_seq_len, d_model)
            src: encoded sequences from encoder (batch_size, src_seq_len, d_model)
            trg_mask: mask for the sequences (batch_size, 1, trg_seq_len, trg_seq_len)
            src_mask: mask for the sequences (batch_size, 1, 1, src_seq_len)

        Returns:
            output: sequences after decoder (batch_size, trg_seq_len, vocab_size)
            attention_probs: self-attention softmax scores (batch_size, n_heads, trg_seq_len, src_seq_len)
        """
        for layer in self.layers:
            trg, attention_probs = layer(trg, src, trg_mask, src_mask)

        self.attention_probs = attention_probs

        dec_out = self.w_o(trg)

        return dec_out