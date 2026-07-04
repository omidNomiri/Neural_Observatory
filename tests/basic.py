"""
Simple example demonstrating Neural Observatory usage.

This script trains a basic MLP on synthetic data and shows how to
integrate Observatory into a standard PyTorch training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Assuming the package is installed via `pip install -e .`
from neural_observatory import Observatory
from collections import OrderedDict


def create_simple_model():
    # Using OrderedDict so layers have readable names in the report
    # instead of just '0', '1', '2', etc.
    return nn.Sequential(OrderedDict([
        ("fc1", nn.Linear(10, 32)),
        ("relu1", nn.ReLU()),
        ("fc2", nn.Linear(32, 16)),
        ("relu2", nn.ReLU()),
        ("fc3", nn.Linear(16, 2)),
    ]))


def create_toy_data(num_samples=100, batch_size=16):
    X = torch.randn(num_samples, 10)
    y = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size)


def main():
    print("=" * 60)
    print("Neural Observatory — Simple Example")
    print("=" * 60)

    model = create_simple_model()
    print(f"✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # 1. Initialize Observatory
    obs = Observatory(model)
    
    # 2. Start watching (attaches hooks)
    obs.watch()
    print(f"✓ Started monitoring — {len(obs.registry_summary()['analyzers'])} analyzers registered")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data_loader = create_toy_data()

    # 3. Training Loop
    print("\nRunning training loop...")
    for epoch in range(2):
        for batch_idx, (X, y) in enumerate(data_loader):
            # NOTE: obs.step() is called *before* the forward/backward
            # pass, not after. HookManager reads the current step/epoch
            # at the moment each hook fires (i.e. during model(X) and
            # loss.backward()), so calling step() afterward would stamp
            # every observation from this batch with the *previous*
            # batch's step number instead of this one.
            obs.step(step=epoch * len(data_loader) + batch_idx, epoch=epoch)

            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            if batch_idx % 3 == 0:
                print(f"  Epoch {epoch}, Batch {batch_idx}/{len(data_loader)}: loss={loss.item():.4f}")

    print("✓ Training completed")

    # 4. Stop monitoring before generating report
    obs.stop()
    print("✓ Monitoring stopped")

    # 5. Generate and display report
    print("\n" + "-" * 60)
    print("Console Report:")
    print("-" * 60)
    
    report = obs.report()
    obs.console_report()

    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()