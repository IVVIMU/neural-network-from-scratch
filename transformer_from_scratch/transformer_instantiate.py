import torch
import torch.nn as nn

from transformer_from_scratch.embedding.embeddings import Embeddings
from transformer_from_scratch.embedding.positional_encoding import PositionalEncoding
from transformer_from_scratch.blocks.encoder import Encoder
from transformer_from_scratch.blocks.decoder import Decoder
from transformer_from_scratch.transformer import Transformer


def transformer(
        src_vocab,
        trg_vocab,
        d_model,
        d_ffn,
        n_heads,
        n_layers,
        device,
        dropout=0.1,
        max_seq_len=50
):
    """
    Construct a model when provided parameters.

    Args:
        src_vocab: source vocabulary
        trg_vocab: target vocabulary
        d_model: dimension of embeddings
        d_ffn: dimension of feed-forward network
        n_heads: number of heads
        n_layers: Number of Encoder and Decoders 
        device: cuda or cpu
        dropout: probability of dropout occurring
        max_seq_len: maximum sequence length for positional encodings

    Returns:
        Transformer model based on hyperparameters
    """

    encoder = Encoder(d_model, d_ffn, n_heads, n_layers, dropout)
    decoder = Decoder(len(trg_vocab), d_model, d_ffn, n_heads, n_layers, dropout)

    # create source embedding matrix
    src_embed = Embeddings(len(src_vocab), d_model)

    # create target embedding matrix
    trg_embed = Embeddings(len(trg_vocab), d_model)

    # create a positional encoding matrix
    pos_enc = PositionalEncoding(d_model, max_seq_len, dropout)

    # create the Transformer model
    model = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_embed=nn.Sequential(src_embed, pos_enc),
        trg_embed=nn.Sequential(trg_embed, pos_enc),
        src_pad_idx=src_vocab.get_stoi()["<pad>"],
        trg_pad_idx=trg_vocab.get_stoi()["<pad>"],
        device=device
    )

    # initialize parameters with Xavier/Glorot
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model