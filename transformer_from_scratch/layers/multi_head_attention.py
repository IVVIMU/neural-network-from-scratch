import torch.nn as nn

from transformer_from_scratch.layers.scaled_dot_product_attention import ScaledDotProductAttention


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        """
        Args:
            d_model: dimension of embeddings
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.attention = ScaledDotProductAttention()

        self.w_q = nn.Linear(d_model, d_model)  # query weights
        self.w_k = nn.Linear(d_model, d_model)  # key weights
        self.w_v = nn.Linear(d_model, d_model)  # value weights
        self.w_o = nn.Linear(d_model, d_model)  # output weights

    # def split(self, tensor):
    #     """
    #     Args:
    #         tensor: (batch_size, seq_len, d_model)

    #     Returns: 
    #         tensor: (batch_size, seq_len, n_heads, d_head]
    #     """
    #     batch_size, seq_len, d_model = tensor.size()

    #     # [batch_size, seq_len, n_heads, d_head] -> [batch_size, n_heads, seq_len, d_head]
    #     tensor = tensor.view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

    #     return tensor

    # # def concat(self, tensor):
    # #     """
    # #     inverse function of self.split(tensor : torch.Tensor)

    # #     :param tensor: [batch_size, n_heads, seq_len, d_head]
    # #     :return: [batch_size, seq_len, d_model]
    # #     """
    # #     batch_size, n_heads, seq_len, d_head = tensor.size()
    # #     # make the tensor contiguous in memory
    # #     tensor = tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, n_heads * d_head)

    # #     return tensor

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: query vector (batch_size, q_len, d_model)
            k: key vector (batch_size, k_len, d_model)
            v: value vector (batch_size, v_len, d_model)
            mask: mask for decoder

        Returns:
            output: attention values (batch_size, q_len, d_model)
            attention_probs: softmax scores (batch_size, n_heads, q_len, k_len)
        """
        batch_size = k.size(0)

        # 1. calculate query, key and value tensors
        # (batch_size, seq_length, d_model) x (d_model, d_model) = (batch_size, seq_length, d_model)
        Q, K, V = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split each tensor into n-heads to compute attention
        # (batch_size, seq_len, n_heads, d_head) -> (batch_size, n_heads, seq_len, d_head)
        Q = Q.view(batch_size, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.d_head).permute(0, 2, 1, 3)

        # 3. do scaled dot-product (QK^{T}) to compute similarity
        attention, attention_probs = self.attention(Q, K, V, mask=mask)

        # 4. reshape attention back to (batch_size, seq_len, d_model)
        # (batch_size, n_heads, seq_len, d_head) -> (batch_size, seq_len, n_heads, d_head) -> (batch_size, seq_len, d_model)
        attention = attention.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.n_heads * self.d_head)

        # 5. visualize attention map
        # TODO: implement visualization

        out = self.w_o(attention)

        return out, attention_probs