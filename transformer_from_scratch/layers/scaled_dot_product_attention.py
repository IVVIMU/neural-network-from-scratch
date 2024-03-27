import torch.nn as nn
import math


class ScaledDotProductAttention(nn.Module):
    """
    compute scaled dot-product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor [batch_size, n_heads, seq_len, d_head]
        batch_size, n_heads, seq_len, d_head = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # [batch_size, n_heads, d_head, seq_len]
        # q @ k_t -> [batch_size, n_heads, seq_len, seq_len]
        score = (q @ k_t) / math.sqrt(d_head)  # scaled dot-product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-1e20'))

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        # score @ v -> [batch_size, n_heads, seq_len, d_head]
        v = score @ v

        return v, score