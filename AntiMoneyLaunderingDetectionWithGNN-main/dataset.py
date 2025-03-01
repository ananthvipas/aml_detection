import os
import datetime
from typing import Callable, Optional
import pandas as pd
from sklearn import preprocessing
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset

pd.set_option('display.max_columns', None)

class AMLtoGraph(InMemoryDataset):
    def __init__(self, root: str, edge_window_size: int = 10,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.edge_window_size = edge_window_size
        super().__init__(root, transform, pre_transform)
        # If processed data already exists, this will load it.
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        
    @property
    def raw_file_names(self) -> str:
        # The raw CSV file should be placed in data/raw/
        return 'HI-Small_Trans.csv'
    
    @property
    def processed_file_names(self) -> str:
        return 'data.pt'
    
    @property
    def num_nodes(self) -> int:
        return self.data.edge_index.max().item() + 1
    
    def df_label_encoder(self, df, columns):
        le = preprocessing.LabelEncoder()
        for i in columns:
            df[i] = le.fit_transform(df[i].astype(str))
        return df

    def preprocess(self, df):
        df = self.df_label_encoder(df, ['Payment Format', 'Payment Currency', 'Receiving Currency'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Timestamp'] = df['Timestamp'].apply(lambda x: x.value)
        df['Timestamp'] = (df['Timestamp'] - df['Timestamp'].min()) / (df['Timestamp'].max() - df['Timestamp'].min())
        
        # Combine bank and account information
        df['Account'] = df['From Bank'].astype(str) + '_' + df['Account']
        df['Account.1'] = df['To Bank'].astype(str) + '_' + df['Account.1']
        df = df.sort_values(by=['Account'])
        
        receiving_df = df[['Account.1', 'Amount Received', 'Receiving Currency']]
        paying_df = df[['Account', 'Amount Paid', 'Payment Currency']]
        receiving_df = receiving_df.rename({'Account.1': 'Account'}, axis=1)
        currency_ls = sorted(df['Receiving Currency'].unique())
        
        return df, receiving_df, paying_df, currency_ls

    def get_all_account(self, df):
        ldf = df[['Account', 'From Bank']]
        rdf = df[['Account.1', 'To Bank']]
        suspicious = df[df['Is Laundering'] == 1]
        s1 = suspicious[['Account', 'Is Laundering']]
        s2 = suspicious[['Account.1', 'Is Laundering']]
        s2 = s2.rename({'Account.1': 'Account'}, axis=1)
        suspicious = pd.concat([s1, s2], join='outer')
        suspicious = suspicious.drop_duplicates()

        ldf = ldf.rename({'From Bank': 'Bank'}, axis=1)
        rdf = rdf.rename({'Account.1': 'Account', 'To Bank': 'Bank'}, axis=1)
        df_accounts = pd.concat([ldf, rdf], join='outer')
        df_accounts = df_accounts.drop_duplicates()

        df_accounts['Is Laundering'] = 0
        df_accounts.set_index('Account', inplace=True)
        df_accounts.update(suspicious.set_index('Account'))
        df_accounts = df_accounts.reset_index()
        return df_accounts

    def paid_currency_aggregate(self, currency_ls, paying_df, accounts):
        for i in currency_ls:
            temp = paying_df[paying_df['Payment Currency'] == i]
            # Calculate mean per account
            accounts['avg paid ' + str(i)] = temp.groupby('Account')['Amount Paid'].transform('mean')
        return accounts

    def received_currency_aggregate(self, currency_ls, receiving_df, accounts):
        for i in currency_ls:
            temp = receiving_df[receiving_df['Receiving Currency'] == i]
            accounts['avg received ' + str(i)] = temp.groupby('Account')['Amount Received'].transform('mean')
        accounts = accounts.fillna(0)
        return accounts

    def get_edge_df(self, accounts, df):
        accounts = accounts.reset_index(drop=True)
        accounts['ID'] = accounts.index
        mapping_dict = dict(zip(accounts['Account'], accounts['ID']))
        df['From'] = df['Account'].map(mapping_dict)
        df['To'] = df['Account.1'].map(mapping_dict)
        df = df.drop(['Account', 'Account.1', 'From Bank', 'To Bank'], axis=1)
        
        edge_index = torch.stack([torch.from_numpy(df['From'].values), torch.from_numpy(df['To'].values)], dim=0)
        df = df.drop(['Is Laundering', 'From', 'To'], axis=1)
        edge_attr = torch.from_numpy(df.values).to(torch.float)
        return edge_attr, edge_index

    def get_node_attr(self, currency_ls, paying_df, receiving_df, accounts):
        node_df = self.paid_currency_aggregate(currency_ls, paying_df, accounts)
        node_df = self.received_currency_aggregate(currency_ls, receiving_df, node_df)
        node_label = torch.from_numpy(node_df['Is Laundering'].values).to(torch.float)
        node_df = node_df.drop(['Account', 'Is Laundering'], axis=1)
        node_df = self.df_label_encoder(node_df, ['Bank'])
        node_df = torch.from_numpy(node_df.values).to(torch.float)
        return node_df, node_label

    def process(self):
        try:
            print("🚀 Processing dataset...")
            # Build the path for the raw file
            raw_path = os.path.join(self.raw_dir, self.raw_file_names)
            print(f"✅ Loading CSV from: {raw_path}")
            df = pd.read_csv(raw_path)
            print(f"✅ Data Loaded! Shape: {df.shape}")

            df, receiving_df, paying_df, currency_ls = self.preprocess(df)
            print("✅ Preprocessing Complete!")

            accounts = self.get_all_account(df)
            print(f"✅ Accounts Processed! Total accounts: {len(accounts)}")

            node_attr, node_label = self.get_node_attr(currency_ls, paying_df, receiving_df, accounts)
            edge_attr, edge_index = self.get_edge_df(accounts, df)
            print(f"✅ Nodes: {node_attr.shape}, Labels: {node_label.shape}")
            print(f"✅ Edges: {edge_index.shape}, Edge Attributes: {edge_attr.shape}")

            data = Data(x=node_attr,
                        edge_index=edge_index,
                        y=node_label,
                        edge_attr=edge_attr)

            data_list = [data]
            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            data, slices = self.collate(data_list)
            processed_path = os.path.join(self.processed_dir, self.processed_file_names)
            torch.save((data, slices), processed_path)
            print(f"✅ Data saved at {processed_path}")
        except Exception as e:
            print("❌ An error occurred during processing:", e)

if __name__ == '__main__':
    # Before creating the dataset, check if the raw file exists
    raw_csv = os.path.join("data", "raw", "HI-Small_Trans.csv")
    if not os.path.exists(raw_csv):
        print(f"❌ ERROR: File not found at {raw_csv}")
    else:
        print(f"✅ File found! Loading {raw_csv}")
    
    # Create the dataset object which will process the data if not already processed.
    dataset = AMLtoGraph(os.path.join("data"))
