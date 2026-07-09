"""
NLP example demonstrating Neural Observatory's Embedding Drift detection.

This script trains a simple text classifier with a high learning rate
to intentionally induce rapid changes in the embedding weights, triggering
the EmbeddingDriftAnalyzer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from neural_observatory import Observatory, ObservatoryConfig


class SimpleNLPModel(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=16, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.fc1 = nn.Linear(embed_dim, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        x = x.mean(dim=1)      # Global average pooling -> (batch_size, embed_dim)
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)


def create_toy_nlp_data(num_samples=100, seq_len=5, vocab_size=100, batch_size=16):
    # Random integer tokens representing word indices
    X = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, 2, (num_samples,))
    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size)


def main():
    print("=" * 60)
    print("Neural Observatory — NLP Embedding Drift Example")
    print("=" * 60)

    model = SimpleNLPModel()
    print(f"✓ Created NLP model with {sum(p.numel() for p in model.parameters())} parameters")

    # Configure Observatory to keep raw tensors (needed for cosine similarity)
    # and set strict thresholds for embedding drift.
    config = ObservatoryConfig(
        stats_only_mode=False,          # MUST be False to calculate cosine similarity
        parameter_sample_rate=1,        # Snapshot parameters every step
        embedding_drift_warning=0.85,   # Warn if similarity drops below 0.85
        embedding_drift_critical=0.60   # Critical if drops below 0.60
    )

    # 1. Initialize Observatory
    obs = Observatory(model, config=config)
    
    # 2. Start watching
    obs.watch()
    print(f"✓ Started monitoring — {len(obs.registry_summary()['analyzers'])} analyzers registered")

    # Using a very high learning rate to force embeddings to drift quickly
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1.5)
    data_loader = create_toy_nlp_data()

    # 3. Training Loop
    print("\nRunning training loop with high LR to induce drift...")
    for epoch in range(2):
        for batch_idx, (X, y) in enumerate(data_loader):
            # IMPORTANT: Call step() before forward pass
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
    print("✓ Look for 'EMBEDDING DRIFT' in the report above!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()