import math

import torch.nn as nn
import torch.nn.functional as F


class myModel(nn.Module):

    def __init__(self, encoder, n_features, labels):
        super(myModel, self).__init__()
        self.encoder = encoder
        middle_dim = int(math.sqrt(n_features * labels) * 2)
        self.projector = nn.Sequential(
            nn.Linear(n_features, middle_dim),
            nn.ReLU(),
            nn.Linear(middle_dim, labels),
        )
        self.labels = labels

    def forward(self, x):
        x = self.encoder(x)
        x = F.dropout(x, 0.5)
        x = self.projector(x)

        return x.view(x.size(0), self.labels)