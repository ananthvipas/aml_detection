import torch
from model import GAT
from dataset import AMLtoGraph

# ✅ Load dataset to get the correct input feature size
dataset = AMLtoGraph(r'D:\DHI\AntiMoneyLaunderingDetectionWithGNN-main\data')
data = dataset[0]  # Load dataset

# ✅ Get the correct number of input features dynamically
in_channels = data.num_features  

# ✅ Load trained model
model = GAT(in_channels=in_channels, hidden_channels=16, out_channels=1, heads=8)
model.load_state_dict(torch.load("aml_gnn_model.pth"))
model.eval()

# ✅ Select a batch of transactions for testing
num_sample_nodes = 10  # Adjust based on dataset size
x = data.x[:num_sample_nodes]  # Select first few nodes
edge_index = data.edge_index  # Use full graph structure
edge_attr = data.edge_attr  # Use full edge attributes

# ✅ Predict
with torch.no_grad():
    predictions = model(x, edge_index, edge_attr)
    labels = (predictions > 0.5).float()  # Convert probabilities to 0 or 1

# ✅ Print Results
print("\n🔍 **Prediction Results:**")
for i, label in enumerate(labels):
    print(f"Transaction {i + 1} → Predicted Class: {int(label.item())}")  # 0 = Normal, 1 = Money Laundering
