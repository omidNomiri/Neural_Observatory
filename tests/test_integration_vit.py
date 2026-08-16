"""Integration tests for Vision Transformers (ViT) and LLMs."""
import torch
import torch.nn as nn

from neural_observatory import Observatory, ObservatoryConfig


class TinyViT(nn.Module):
    """A minimal Vision Transformer for testing purposes."""
    def __init__(self, image_size=32, patch_size=8, num_classes=4, dim=16, heads=2):
        super().__init__()
        self.patch_dim = patch_size * patch_size * 3  # 3 for RGB channels
        self.num_patches = (image_size // patch_size) ** 2
        
        # 1. Embedding layer for patches (we want to test EmbeddingDrift here)
        self.patch_embed = nn.Linear(self.patch_dim, dim)
        
        # 2. MultiheadAttention layer (we want to test AttentionHealth here)
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        b, c, h, w = x.shape
        
        # Extract patches
        patches = x.unfold(2, 8, 8).unfold(3, 8, 8)  # (b, c, h/8, w/8, 8, 8)
        patches = patches.contiguous().view(b, c, -1, 8*8)  # (b, c, num_patches, 64)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(b, self.num_patches, -1)
        
        # Linear embedding
        x = self.patch_embed(patches)  # (b, num_patches, dim)
        
        # Attention
        attn_out, _ = self.attention(x, x, x)
        
        # Classification head
        x = attn_out.mean(dim=1)  # Global average pooling
        return self.fc(x)


def test_vit_attention_and_embedding_monitoring():
    """Ensure Observatory hooks correctly into ViT's MultiheadAttention and Embedding."""
    model = TinyViT()
    
    # Enable raw tensor storage so analyzers can compute entropy and cosine similarity
    config = ObservatoryConfig(
        stats_only_mode=False, 
        max_observations=50,
        neural_collapse_layer="patch_embed",  # Monitor this layer for NC
    )
    
    obs = Observatory(model, config=config)
    obs.watch()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Run a tiny training loop on fake image data
    for step in range(3):
        obs.step(step=step, epoch=0, targets=torch.randint(0, 4, (4,)))
        
        # Fake batch of 4 RGB images, 32x32
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 4, (4,))
        
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    obs.stop()
    report = obs.report()

    # 1. Verify Attention weights were captured
    # Since we changed the architecture to use metadata flags instead of name suffixes
    attn_layer_obs = obs.activation_collector.get("attention")
    assert len(attn_layer_obs) > 0, "No observations for attention layer!"
    assert any(o.metadata.get("is_attention_weights") for o in attn_layer_obs), \
        "MultiheadAttention weights were not captured by hooks!"

    # 2. Verify Embedding Drift analyzer ran on the patch embedding
    drift_results = [r for r in report.results if r.analyzer_name == "embedding_drift"]
    assert len(drift_results) > 0, "EmbeddingDriftAnalyzer did not run on ViT patch embeddings."

    # 3. Verify Attention Health analyzer ran
    attn_results = [r for r in report.results if r.analyzer_name == "attention_health"]
    assert len(attn_results) > 0, "AttentionHealthAnalyzer did not run on ViT attention weights."

    # 4. Verify Neural Collapse analyzer ran (since we configured it)
    nc_results = [r for r in report.results if r.analyzer_name == "neural_collapse"]
    assert len(nc_results) > 0, "NeuralCollapseAnalyzer did not run on the configured layer."