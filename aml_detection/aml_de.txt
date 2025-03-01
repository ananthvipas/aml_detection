import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def load_dataset(filepath):
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.strip().str.lower()
    df['sender'] = df['from bank'].astype(str) + " - " + df['account'].astype(str)
    df['receiver'] = df['to bank'].astype(str) + " - " + df['account.1'].astype(str)
    df['amount'] = df['amount paid']
    df['label'] = df['is laundering']  # Assuming 'Is Laundering' column exists
    return df[['sender', 'receiver', 'amount', 'label']]

def build_transaction_graph(df):
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['sender'], row['receiver'], amount=row['amount'])
    return G

def extract_graph_features(G):
    features = {}
    for node in G.nodes():
        features[node] = {
            'degree': G.degree(node),
            'clustering': nx.clustering(G, node),
        }
    return pd.DataFrame.from_dict(features, orient='index')

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

if __name__ == "__main__":
    dataset_path = "aml_dataset.xlsx"
    df = load_dataset(dataset_path)
    transaction_graph = build_transaction_graph(df)
    features_df = extract_graph_features(transaction_graph)
    labels = df[['sender', 'label']].drop_duplicates().set_index('sender')
    
    # Merge features and labels
    data = features_df.join(labels).dropna()
    X = data.drop(columns=['label'])
    y = data['label']
    
    # Train Random Forest Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))


    