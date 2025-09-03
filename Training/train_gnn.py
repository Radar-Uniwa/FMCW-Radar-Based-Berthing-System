import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ─── Experimental training script ───────────────────────────────────────────────
class RadarNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, extra_feats,
                 conv_type='gcn', dropout_p=0.0):
        super().__init__()
        # Choose convolution type
        if conv_type == 'gat':
            self.conv1 = GATConv(in_channels, hidden_channels, heads=1)
            self.conv2 = GATConv(hidden_channels, hidden_channels, heads=1)
        else:  # 'gcn'
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.lin = torch.nn.Linear(hidden_channels + extra_feats, 2)
        self.extra_feats = extra_feats

    def forward(self, x, edge_index, batch, cluster_feats=None):
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = self.dropout(h)
        g = global_mean_pool(h, batch)
        if self.extra_feats > 0:
            if cluster_feats.dim() == 1:
                B = cluster_feats.numel() // self.extra_feats
                cluster_feats = cluster_feats.view(B, self.extra_feats)
            g = torch.cat([g, cluster_feats], dim=1)
        return self.lin(g)

def train_and_evaluate(config):
    # Load data
    base = config['base_path']
    train_data = torch.load(os.path.join(base, "training_set.pt"),# Adjust filename as needed
                            map_location='cpu', weights_only=False)
    val_data   = torch.load(os.path.join(base, "validation_set.pt"),# Adjust filename as needed
                            map_location='cpu', weights_only=False)
    test_data  = torch.load(os.path.join(base, "test_set.pt"),# Adjust filename as needed
                            map_location='cpu', weights_only=False)

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=config['batch_size'])
    test_loader  = DataLoader(test_data,  batch_size=config['batch_size'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample = train_data[0]
    extra_feats = sample.cluster_feats.numel() if hasattr(sample, 'cluster_feats') else 0
    model = RadarNet(
        in_channels=5,
        hidden_channels=config['hidden_channels'],
        extra_feats=extra_feats,
        conv_type=config['conv_type'],
        dropout_p=config['dropout']
    ).to(device)

    # Optimizer & scheduler
    if config['optimizer'] == 'adamw':
        optimizer = AdamW(model.parameters(),
                          lr=config['lr'], weight_decay=config['weight_decay'])
    else:
        optimizer = Adam(model.parameters(), lr=config['lr'])

    scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=3) if config['use_scheduler'] else None

    best_val = float('inf')
    patience_ctr = 0

    # Training loop
    for epoch in range(1, config['epochs']+1):
        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch,
                        getattr(batch, 'cluster_feats', None))
            loss = F.cross_entropy(out, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch,
                            getattr(batch, 'cluster_feats', None))
                val_loss += F.cross_entropy(out, batch.y.view(-1)).item()

        if scheduler:
            scheduler.step(val_loss)

        print(f"[{config['name']}] Epoch {epoch} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience_ctr = 0
            torch.save(model.state_dict(), os.path.join(base, f"best_{config['name']}.pt"))
        else:
            patience_ctr += 1
            if patience_ctr >= config['patience']:
                print(f"[{config['name']}] Early stopping")
                break

    # Load best and evaluate on test
    model.load_state_dict(torch.load(os.path.join(base, f"best_{config['name']}.pt")))
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch,
                        getattr(batch, 'cluster_feats', None))
            pred = out.argmax(dim=1)
            correct += (pred == batch.y.view(-1)).sum().item()
            total += batch.num_graphs
    acc = correct/total * 100
    print(f"[{config['name']}] Test Accuracy: {acc:.2f}%")
    return acc

if __name__ == "__main__":
    BASE_PATH = r"base/path/to/data"  # Adjust as needed
    experiments = [
        dict(name="baseline_gcn",
             conv_type="gcn",
             dropout=0.0,
             optimizer="adam",
             lr=1e-3,
             weight_decay=0,
             use_scheduler=False,
             hidden_channels=64,
             batch_size=32,
             epochs=30,
             patience=5,
             base_path=BASE_PATH),
        dict(name="reg_gcn",
             conv_type="gcn",
             dropout=0.2,
             optimizer="adamw",
             lr=1e-3,
             weight_decay=1e-4,
             use_scheduler=True,
             hidden_channels=64,
             batch_size=32,
             epochs=30,
             patience=5,
             base_path=BASE_PATH),
        dict(name="reg_gat",
             conv_type="gat",
             dropout=0.2,
             optimizer="adamw",
             lr=1e-3,
             weight_decay=1e-4,
             use_scheduler=True,
             hidden_channels=64,
             batch_size=32,
             epochs=30,
             patience=5,
             base_path=BASE_PATH),
    ]

    results = {}
    for cfg in experiments:
        acc = train_and_evaluate(cfg)
        results[cfg['name']] = acc

    print("\nSummary of results:")
    for name, acc in results.items():
        print(f"  {name}: {acc:.2f}%")
