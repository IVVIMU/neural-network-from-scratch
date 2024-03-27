import torch.nn as nn


class PositionWiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, d_model)
        )

    def forward(self, x):
        out = self.feed_forward(x)
        return out