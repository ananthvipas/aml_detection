import torch
import networkx as nx
import matplotlib.pyplot as plt
from model import GAT
from dataset import AMLtoGraph

# ✅ Load dataset
dataset = AMLtoGraph(r'D:\DHI\AntiMoneyLaunderingDetectionWithGNN-main\data')
data = dataset[0]

# ✅ Get input feature size dynamically
in_channels = data.num_features  

# ✅ Load trained model
model = GAT(in_channels=in_channels, hidden_channels=16, out_channels=1, heads=8)
model.load_state_dict(torch.load("aml_gnn_model.pth"))
model.eval()

# ✅ Select a subset of nodes
num_sample_nodes = 20  # Adjust based on dataset size
x = data.x[:num_sample_nodes]  # Select first 20 nodes

# ✅ Filter edges to only include edges within `num_sample_nodes`
mask = (data.edge_index[0] < num_sample_nodes) & (data.edge_index[1] < num_sample_nodes)
edge_index = data.edge_index[:, mask]  # Keep only valid edges
edge_attr = data.edge_attr[mask]  # Keep only valid edge attributes

# ✅ Predict
with torch.no_grad():
    predictions = model(x, edge_index, edge_attr)
    labels = (predictions > 0.5).float()  # Convert probabilities to 0 or 1

# ✅ Print Results
print("\n🔍 **Prediction Results:**")
for i, label in enumerate(labels):
    print(f"Transaction {i + 1} → Predicted Class: {int(label.item())}")  # 0 = Normal, 1 = Money Laundering

# ✅ Create a Graph Plot
G = nx.Graph()

# Add nodes with color coding
for i in range(num_sample_nodes):
    color = "red" if labels[i].item() == 1 else "green"
    G.add_node(i, color=color)

# Add edges
for src, dst in edge_index.T.tolist():
    G.add_edge(src, dst)

# Get colors for nodes
node_colors = [G.nodes[n]["color"] for n in G.nodes]

# ✅ Plot Graph
plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color=node_colors, edge_color="gray", node_size=500, font_size=10)
plt.title("🔍 AML Transaction Graph (Red = Money Laundering, Green = Normal)")
plt.show()
