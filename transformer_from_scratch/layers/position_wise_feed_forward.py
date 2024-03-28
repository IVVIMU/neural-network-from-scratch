import torch.nn as nn


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ffn, dropout=0.1):
        """
        Args:
            d_model: dimension of embeddings
            d_ffn: dimension fo feed-forward network
            dropout: probability of dropout occurring
        """
        super(PositionWiseFeedForward, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ffn, d_model)
        )

    def forward(self, x):
        out = self.feed_forward(x)
        return out