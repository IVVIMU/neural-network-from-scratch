import torch.nn as nn

class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimension of model
        """
        # padding_idx=1 sets the embedding for index 1 (padding token) to a zero vector, ignoring it in learning.
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)