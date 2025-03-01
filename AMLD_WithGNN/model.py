import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GATConv, Linear

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, int(hidden_channels / 4), heads=1, concat=False, dropout=0.6)
        self.lin = Linear(int(hidden_channels / 4), out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_attr):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.lin(x)
        x = self.sigmoid(x)
        return x

# ✅ Simple test if model.py is run directly
if __name__ == "__main__":
    print("🚀 Testing GAT Model...")
    
    # Dummy input data
    in_channels = 16
    hidden_channels = 32
    out_channels = 1
    heads = 8
    num_nodes = 10  # Example graph with 10 nodes

    # Create a dummy model
    model = GAT(in_channels, hidden_channels, out_channels, heads)

    # Create fake data
    x = torch.rand((num_nodes, in_channels))  # Node features
    edge_index = torch.randint(0, num_nodes, (2, 20))  # Random edges
    edge_attr = torch.rand(20, 5)  # Random edge features

    # Forward pass
    output = model(x, edge_index, edge_attr)
    print("✅ Model output shape:", output.shape)  # Should print (10, 1)
