import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ✅ Load dataset function with debugging
def load_dataset(filepath):
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip().str.lower()
    
    # Ensure column names exist
    required_cols = {'from bank', 'to bank', 'account', 'account.1', 'amount paid', 'is laundering'}
    if not required_cols.issubset(df.columns):
        print("Error: Missing columns in dataset!")
        print("Found columns:", df.columns)
        return None

    # Process dataset
    df['sender'] = df['from bank'].astype(str) + " - " + df['account'].astype(str)
    df['receiver'] = df['to bank'].astype(str) + " - " + df['account.1'].astype(str)
    df['amount'] = df['amount paid']
    df['label'] = df['is laundering']

    # ✅ Check dataset
    print("\n✅ Dataset Loaded Successfully!")
    print(df[['sender', 'receiver', 'amount', 'label']].head())
    print("Dataset Shape:", df.shape)

    return df[['sender', 'receiver', 'amount', 'label']]

# ✅ Build transaction graph
def build_transaction_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['sender'], row['receiver'], amount=row['amount'])

    # ✅ Debug Graph
    print("\n✅ Transaction Graph Created!")
    print(f"Total Nodes: {G.number_of_nodes()}, Total Edges: {G.number_of_edges()}")
    
    return G

# ✅ Extract graph features
def extract_graph_features(G):
    features = {}
    for node in G.nodes():
        features[node] = {
            'degree': G.degree(node),
            'clustering': nx.clustering(G, node),
        }

    feature_df = pd.DataFrame.from_dict(features, orient='index')
    
    # ✅ Debug extracted features
    print("\n✅ Extracted Graph Features:\n", feature_df.head())
    
    return feature_df

# ✅ Optimized Graph Visualization
def plot_transaction_graph(G):
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    nx.draw(
        G, pos, with_labels=True, node_size=800, node_color="lightblue", 
        edge_color="gray", font_size=10, font_weight="bold", arrows=True, width=1
    )
    
    plt.title("Transaction Network Graph", fontsize=14)
    plt.show()

# ✅ Feature Distribution Plot
def plot_feature_distribution(data):
    plt.figure(figsize=(10, 5))
    plt.hist(data['degree'], bins=20, alpha=0.7, label="Node Degree")
    plt.xlabel("Degree")
    plt.ylabel("Count")
    plt.title("Node Degree Distribution")
    plt.legend()
    plt.show()

# ✅ GNN Model for Future Use
class GNN(torch.nn.Module):
    def __init__(self, num_features):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 2)  # Binary classification
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ✅ Main Execution
if __name__ == "__main__":
    dataset_path = "aml_dataset.xlsx"

    df = load_dataset(dataset_path)
    if df is None:
        print("❌ Error: Failed to load dataset. Exiting.")
        exit()

    transaction_graph = build_transaction_graph(df)
    features_df = extract_graph_features(transaction_graph)
    
    labels = df[['sender', 'label']].drop_duplicates().set_index('sender')

    # ✅ Merge features and labels
    data = features_df.join(labels, how="inner").dropna()
    if data.empty:
        print("❌ Error: No valid data after merging features & labels. Exiting.")
        exit()

    X = data.drop(columns=['label'])
    y = data['label']

    # ✅ Debug Final Data
    print("\n✅ Final Processed Data:\n", data.head())

    # ✅ Prevent train-test split error
    if len(X) == 0 or len(y) == 0:
        print("❌ Error: No valid samples found. Check data preprocessing!")
        exit()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # ✅ Train Random Forest Model
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # ✅ Print Classification Report
    print("\n✅ Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))

    # ✅ Plot Graphs
    plot_transaction_graph(transaction_graph)
    plot_feature_distribution(features_df)
