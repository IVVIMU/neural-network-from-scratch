import torch.nn as nn

from layers.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.attention = ScaledDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def split(self, tensor):
        """
        split tensor by number of heads

        :param tensor: [batch_size, seq_len, d_model]
        :return: [batch_size, seq_len, n_heads, d_head]
        """
        batch_size, seq_len, d_model = tensor.size()

        d_head = d_model // self.n_heads
        # [batch_size, seq_len, n_heads, d_head] -> [batch_size, n_heads, seq_len, d_head]
        tensor = tensor.view(batch_size, seq_len, self.n_heads, d_head).transpose(1, 2)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, n_heads, seq_len, d_head]
        :return: [batch_size, seq_len, d_model]
        """
        batch_size, n_heads, seq_len, d_head = tensor.size()
        # make the tensor contiguous in memory
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, n_heads * d_head)

        return tensor

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrix
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        # [batch_size, seq_len, n_heads, d_head]
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scaled dot-product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO: implement visualization

        return out