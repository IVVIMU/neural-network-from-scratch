import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    compute positional encoding
    """

    def __init__(self, d_model, max_seq_len=5000, dropout=0.1):
        """
        Args:
            d_model: dimension of embeddings
            max_seq_len: max sequence length
            dropout: randomly zeroes-out some of the input
        """
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # same size with input matrix (for adding with input matrix)
        positional_encoding = torch.zeros(max_seq_len, d_model)

        pos = torch.arange(0, max_seq_len)
        # 1D -> 2D unsqueeze to represent word's position
        pos = pos.unsqueeze(1) # pos.shape: [seq_len] -> [seq_len, 1]

        _2i = torch.arange(0, d_model, 2)
        # compute positional encoding to consider positional information of words
        positional_encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        positional_encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        # add dimension
        positional_encoding = positional_encoding.unsqueeze(0)

        # buffers are saved in state_dict but not trained by optimizer
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, x):
        """
        Args:
            x: embeddings (batch_size, seq_len, d_model)
        
        Returns:
            embeddings + positional encodings (batch_size, seq_len, d_model)
        """
        x = x + self.positional_encoding[:, : x.size(1)].requires_grad_(False)  # # don't need to compute gradient

        return self.dropout(x)
