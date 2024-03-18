import torch
import torch.nn as nn

from model.encoder import Encoder
from model.decoder import Decoder


class Transformer(nn.Module):

    def __init__(
            self,
            src_pad_idx,
            trg_pad_idx,
            trg_sos_idx,
            src_vocab_size,
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

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            device=device
        )

        self.decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            max_seq_len=max_seq_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            device=device
        )

    def make_src_mask(self, src):
        # suppose src -> [batch_size, seq_len]
        # unsqueeze(1) src -> [batch_size, 1, seq_len]
        # unsqueeze(2) src -> [batch_size, 1, 1, seq_len]
        # attention score -> [batch_size, n_heads, seq_len, seq_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        # trg_pad_mask -> [batch_size, 1, 1, seq_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        # trg_sub_mask -> [seq_len, seq_len]
        trg_sub_mask = torch.tril(torch.ones(trg_len, trg_len)).type(torch.ByteTensor).to(self.device)
        # trg_pad_mask broadcast to [batch_size, 1, seq_len, seq_len]
        # duo to trg_pad_mask, trg_sub_mask broadcasting to [batch_size, 1, seq_len, seq_len]
        trg_mask = trg_pad_mask & trg_sub_mask

        return trg_mask
