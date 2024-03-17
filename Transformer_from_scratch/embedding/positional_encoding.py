import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    compute positional encoding
    """

    def __init__(self, d_model, max_seq_len, device):
        """
        :param d_model: dimension of model
        :param max_seq_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.positional_encoding = torch.zeros(max_seq_len, d_model, device=device)
        self.positional_encoding.requires_grad = False # don't need to compute gradient

        pos = torch.arange(0, max_seq_len, device=device)
        # 1D -> 2D unsqueeze to represent word's position
        pos = pos.float().unsqueeze(dim=1) # pos.shape: [seq_len] -> [seq_len, 1]

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # compute positional encoding to consider positional information of words
        self.positional_encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.positional_encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

    def forward(self, x):
        # self.positional_encoding
        # [max_seq_len=512, d_model=512]

        # [batch_size=128, seq_len=30]
        batch_size, seq_len = x.size()

        return self.positional_encoding[:seq_len, :] # will add with token_embedding
