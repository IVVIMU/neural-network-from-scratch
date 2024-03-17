import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_dim: dimension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_dim

        pe = torch.zeros(seq_len, self.embed_dim)
        for pos in range(seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / 10000 ** ((2 * i) / self.embed_dim))
                pe[pos, i + 1] = math.cos(pos / 10000 ** ((2 * (i + 1)) / self.embed_dim))
        # pe.shape -> [seq_len, embed_dim]
        # unsqueeze(0) -> pe.shape -> [1, max_seq_len, embed_dim]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input vector [batch_size, seq_len, embed_dim]
        Returns:
            x: output [batch_size, seq_len, embed_dim]
        """
        # Scaling Embedding
        x = x * math.sqrt(self.embed_dim)
        # Add Positional Encoding
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, n_heads):
        """
        Args:
            embed_dim: dimension of embedding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = self.embed_dim // self.n_heads

        assert (self.head_dim * self.n_heads == self.embed_dim), "Embedding dimension needs to be divisible by n_heads"

        # query, key and value matrix
        self.query_matrix = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key_matrix = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.value_matrix = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(self.n_heads * self.head_dim, self.embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: query vector
            key: key vector
            value: value vector
            mask: mask for encoder

        Returns:
            out: output vector from multi-head attention
        """
        batch_size = key.size(0)
        seq_len = key.size(1)

        # query dimension can change in decoder during inference
        seq_len_query = query.size(1)

        # suppose [batch_size, seq_len, embed_dim] -> [32, 10, 512]
        # when n_heads=8
        # [batch_size, seq_len, n_heads, head_dim] -> [32, 10, 8, 64]
        query = query.view(batch_size, seq_len_query, self.n_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.n_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.n_heads, self.head_dim)

        Q = self.query_matrix(query)
        K = self.key_matrix(key)
        V = self.value_matrix(value)

        # [batch_size, n_heads, seq_len, head_dim] -> [32, 8, 10, 64]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # compute attention
        K = K.transpose(-1, -2) # [batch_size, n_heads, head_dim, seq_len] -> [32, 8, 64, 10]
        dot_product = torch.matmul(Q, K) # [32, 8, 10, 64] x [32, 8, 64, 10]

        # masking
        if mask is not None:
            dot_product = dot_product.masked_fill(mask == 0, float("-1e20"))

        dot_product = dot_product / math.sqrt(self.head_dim)
        scores = F.softmax(dot_product, dim=-1)
        scores = torch.matmul(scores, V) # [32, 8, 10, 10] x [32, 8, 10, 64]
        # [32, 8, 10, 64] -> [32, 10, 8, 64] -> [32, 10, 512]
        concat = scores.transpose(1, 2).contiguous().view(batch_size, seq_len_query, self.n_heads * self.head_dim)

        out = self.fc_out(concat)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, forward_expansion, dropout):
        """
        Args:
            embed_dim: dimension of the embedding
            n_heads: number of attention heads
            forward_expansion: factor which determines output dimension of linear layer
            dropout: rate of dropout
        """
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attention_out = self.attention(query=query, key=key, value=value)
        attention_res_out = attention_out + value
        norm1_out = self.dropout(self.norm1(attention_res_out))

        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_res_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout(self.norm2(feed_fwd_res_out))
        return norm2_out


class Encoder(nn.Module):
    """
    Args:
        src_vocab_size: vocabulary size of source
        embed_dim: dimension of embedding
        seq_len: length of input sequence
        n_heads: number of heads in multi-head attention
        n_layers: number of encoder layer
        forward_expansion: factor which determines of linear in feed forward layer
        dropout: rate of dropout

    Returns:
        out: output of the encoder
    """
    def __init__(
            self,
            src_vocab_size,
            embed_dim,
            seq_len,
            n_heads,
            n_layers,
            forward_expansion,
            dropout=0.2
    ):
        super(Encoder, self).__init__()

        self.word_embedding = Embedding(src_vocab_size, embed_dim)
        self.position_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    forward_expansion=forward_expansion,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x):
        embed_out = self.word_embedding(x)
        out = self.position_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, forward_expansion, dropout=0.2):
        """
        Args:
            embed_dim: dimension of embedding
            n_heads: number of heads in multi-head attention
            forward_expansion: factor which determines output dimension of linear layer
            dropout: rate of dropout
        """
        super(DecoderBlock, self).__init__()

        self.attention = MultiHeadAttention(embed_dim=embed_dim, n_heads=n_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.transformer_block = TransformerBlock(
            embed_dim=embed_dim,
            n_heads=n_heads,
            forward_expansion=forward_expansion,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, x, mask):
        """
        Args:
            query: query vector
            key: key vector
            x: decoder input
            mask: mask to be given for multi-head attention

        Returns:
            out: output for transformer block
        """
        attention = self.attention(x, x, x, mask=mask)
        value = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query=query, key=key, value=value)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_dim,
            seq_len,
            n_heads,
            n_layers,
            forward_expansion,
            dropout=0.2,
    ):
        """
        Args:
            trg_vocab_size: vocabulary size of target
            embed_dim: dimension of embedding
            seq_len: length of input sequence
            n_heads: number of heads in multi-head attention
            n_layers: number of decoder layer
            forward_expansion: factor which determines number of layers in feed forward layer
            dropout: rate of dropout
        """
        super(Decoder, self).__init__()

        self.word_embedding = Embedding(trg_vocab_size, embed_dim)
        self.position_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    forward_expansion=forward_expansion,
                    dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_dim, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, mask):
        """
        Args:
            x: input vector from target
            enc_out: output from encoder layer
            mask: mask for decoder self attention

        Returns:
            out: output vector
        """
        embed_out = self.word_embedding(x)
        out = self.position_encoder(embed_out)
        out = self.dropout(out)

        for layer in self.layers:
            out = layer(enc_out, enc_out, out, mask)

        out = F.softmax(self.fc_out(out), dim=-1)
        return out


class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            embed_dim,
            seq_len,
            n_heads=4,
            n_layers=2,
            forward_expansion=4,
            dropout=0.2,

    ):
        super(Transformer, self).__init__()

        self.trg_vocab_size = trg_vocab_size

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_dim=embed_dim,
            seq_len=seq_len,
            n_heads=n_heads,
            n_layers=n_layers,
            forward_expansion=forward_expansion,
            dropout=dropout
        )

        self.decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            embed_dim=embed_dim,
            seq_len=seq_len,
            n_heads=n_heads,
            n_layers=n_layers,
            forward_expansion=forward_expansion,
            dropout=dropout
        )

    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        device = trg.device
        # return the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len), device=device)).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def decode(self, src, trg):
        """
        for inference

        Args:
            src: input to encoder
            trg: input to decoder

        Returns:
            out_labels: returns final prediction of sequence
        """
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size, seq_len = src.shape[0], src.shape[1]

        out = trg
        for i in range(seq_len):
            out = self.decoder(out, enc_out, mask=trg_mask)

            # take the last token
            out = out[:, -1, :]

            out = out.argmax(-1)
            print(out)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, axis=0)

        return out_labels

    def forward(self, src, trg):
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out = self.decoder(trg, enc_out, trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src = torch.tensor([
        [1, 5, 6, 4, 3, 9, 5, 2, 0, 1],
        [1, 8, 7, 3, 4, 5, 6, 7, 2, 4]
    ]).to(device)

    trg = torch.tensor([
        [1, 7, 4, 3 ,5, 9 ,2, 0, 7, 8],
        [1, 5, 6, 2, 4, 7, 6, 2, 4, 9]
    ]).to(device)

    src_vocab_size = 12
    trg_vocab_size = 12
    model = Transformer(
        src_vocab_size=src_vocab_size,
        trg_vocab_size=trg_vocab_size,
        embed_dim=256,
        seq_len=12,
        n_heads=2,
        n_layers=2
    ).to(device)

    out = model(src, trg)
    print(out.shape)

    src = torch.tensor([[0, 2, 5, 6, 4, 3, 9, 5, 2, 9]]).to(device)
    trg = torch.tensor([[4]]).to(device)
    print(src.shape, trg.shape)
    out_labels = model.decode(src, trg)
    print(out_labels)

