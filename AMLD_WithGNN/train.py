import torch
from model import GAT
from dataset import AMLtoGraph
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader

# Select CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Dataset
dataset = AMLtoGraph(r'D:\DHI\AMLD_WithGNN\data')
data = dataset[0]
epoch = 100

print("Input Features:", data.num_features)

# Define Model
model = GAT(in_channels=data.num_features, hidden_channels=16, out_channels=1, heads=8)
model = model.to(device)

# Loss Function & Optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Using Adam for better convergence

# Split Dataset
split = T.RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0)
data = split(data)

# Data Loaders
train_loader = NeighborLoader(
    data,
    num_neighbors=[30] * 2,
    batch_size=256,
    input_nodes=data.train_mask,
)

test_loader = NeighborLoader(
    data,
    num_neighbors=[30] * 2,
    batch_size=256,
    input_nodes=data.val_mask,
)

# Training Loop
for i in range(epoch):
    total_loss = 0
    model.train()
    
    for batch in train_loader:
        batch = batch.to(device)  # Move batch to GPU/CPU
        optimizer.zero_grad()
        
        pred = model(batch.x, batch.edge_index, batch.edge_attr)
        ground_truth = batch.y.unsqueeze(1)  # Ensure correct shape

        loss = criterion(pred, ground_truth)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # Print loss and evaluate every 10 epochs
    if i % 10 == 0:
        print(f"Epoch: {i:03d}, Loss: {total_loss:.4f}")

        # Model Evaluation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for test_data in test_loader:
                test_data = test_data.to(device)
                pred = model(test_data.x, test_data.edge_index, test_data.edge_attr)
                
                pred_labels = (pred > 0.5).float()  # Convert probabilities to 0 or 1
                correct += (pred_labels == test_data.y.unsqueeze(1)).sum().item()
                total += len(test_data.y)

        accuracy = correct / total if total > 0 else 0  # Avoid division by zero
        print(f"\U0001F50D Model Test Accuracy: {accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), "aml_gnn_model.pth")
print("✅ Model saved successfully!")

# Ensure optimizer is defined (This might not be needed again)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Save model again (Redundant but as per request)
torch.save(model.state_dict(), "aml_gnn_model.pth")
print("✅ Model saved successfully! (Final)")


model.load_state_dict(torch.load("aml_gnn_model.pth"))
model.eval()