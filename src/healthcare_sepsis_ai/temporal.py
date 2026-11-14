"""Optional PyTorch temporal model definitions, not trained production models.

Inputs have shape (batch, hours, features). Fit means/scales on train patients only.
GRUD uses explicit missingness and time-since-measurement; TCN uses causal padding.
"""
import torch
from torch import nn


class GRUD(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.input_decay = nn.Linear(n_features, n_features)
        self.hidden_decay = nn.Linear(n_features, hidden_size)
        self.cell = nn.GRUCell(n_features * 2, hidden_size)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, values, mask, deltas, population_mean):
        hidden = values.new_zeros((values.shape[0], self.hidden_size))
        last = population_mean.expand(values.shape[0], -1).clone()
        outputs = []
        # Input values must have NaNs replaced with zero; masks distinguish missingness.
        for hour in range(values.shape[1]):
            decay_x = torch.exp(-torch.relu(self.input_decay(deltas[:, hour])))
            decay_h = torch.exp(-torch.relu(self.hidden_decay(deltas[:, hour])))
            m = mask[:, hour]
            x = m * values[:, hour] + (1 - m) * (decay_x * last + (1 - decay_x) * population_mean)
            hidden = self.cell(torch.cat([x, m], dim=-1), hidden * decay_h)
            last = m * values[:, hour] + (1 - m) * last
            outputs.append(self.head(hidden).squeeze(-1))
        return torch.stack(outputs, dim=1)


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)

    def forward(self, x):
        return x + torch.relu(self.conv(nn.functional.pad(x, (self.padding, 0))))


class TCN(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32):
        super().__init__()
        self.project = nn.Conv1d(n_features * 2, hidden_size, 1)
        self.blocks = nn.Sequential(*[CausalBlock(hidden_size, d) for d in [1, 2, 4]])
        self.head = nn.Conv1d(hidden_size, 1, 1)

    def forward(self, values, mask):
        x = torch.cat([values, mask], dim=-1).transpose(1, 2)
        return self.head(self.blocks(self.project(x))).squeeze(1)
