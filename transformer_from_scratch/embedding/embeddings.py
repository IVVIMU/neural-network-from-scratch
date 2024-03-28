import torch.nn as nn
import math


class Embeddings(nn.Module):
    """
    Embeddings using torch.nn will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: size of vocabulary
            d_model: dimension of embeddings
        """
        super(Embeddings, self).__init__()

        # embedding look-up table (lut)
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Args:
            x: input Tensor (batch_size, seq_length)
        
        Returns:
            embedding vector
        """
        return self.lut(x) * math.sqrt(self.d_model)