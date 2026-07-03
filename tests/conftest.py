import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn

from neural_observatory import Observatory, ObservatoryConfig


class TinyInplaceReLUNet(nn.Module):
    """Network using inplace=True to test for tensor-aliasing bugs."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 16)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


@pytest.fixture
def inplace_model():
    torch.manual_seed(0)
    return TinyInplaceReLUNet()


@pytest.fixture
def default_config():
    return ObservatoryConfig(max_observations=50)


def run_training_steps(obs: Observatory, model: nn.Module, n_steps: int, batch_size: int = 4):
    """Run n_steps of a tiny CPU training loop through the given Observatory."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    
    for step in range(n_steps):
        # IMPORTANT: Call obs.step() *before* forward/backward so hooks 
        # stamp the correct current iteration, not the previous one.
        obs.step(step=step, epoch=0)
        x = torch.randn(batch_size, 8)
        y = torch.randint(0, 4, (batch_size,))
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()