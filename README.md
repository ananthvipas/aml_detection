# Anti-Money Laundering Detection with Graph Attention Network (GAT)

This repository provides an implementation of a Graph Attention Network (GAT) for detecting money laundering activities using transaction data. The model is trained on structured transaction data to identify suspicious activities.

## Dataset

We utilize the IBM Anti-Money Laundering dataset, which is available on Kaggle:\
[IBM Transactions for Anti-Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.7+
- NumPy
- PyTorch
- PyTorch Geometric (PyG)
- pandas

You can install the dependencies using `pip` or `conda`. To install PyG, use the following command:

```bash
pip install torch_geometric
```

### Project Structure

Ensure your project directory follows this structure:

```bash
├── data
│   ├── raw          # Raw dataset files
│   ├── processed    # Automatically created after preprocessing
├── dataset.py       # Handles data loading and preprocessing
├── model.py         # Defines the GAT model architecture
├── train.py         # Trains the model
├── anti-money-laundering-detection-with-gnn.ipynb  # Data analysis and visualization
```

## Data Analysis & Visualization

The Jupyter Notebook [`anti-money-laundering-detection-with-gnn.ipynb`](anti-money-laundering-detection-with-gnn.ipynb) contains:

- Exploratory Data Analysis (EDA)
- Feature engineering techniques
- Data visualization
- Dataset design details

## Data Preprocessing

All preprocessing is handled by [`dataset.py`](dataset.py), which utilizes `torch_geometric.data.InMemoryDataset` to process transaction records efficiently. Ensure the dataset is placed in the `data/raw` directory before running the script.

## Model Training

Modify line 8 in [`train.py`](train.py) to set the correct dataset path, e.g.,:

```python
DATA_PATH = '/path/to/AntiMoneyLaunderingDetectionWithGNN/data'
```

### Training Configuration

The model is trained using the following hyperparameters:

- **Epochs**: 100
- **Train Batch Size**: 256
- **Test Batch Size**: 256
- **Learning Rate**: 0.0001
- **Optimizer**: SGD

To start training, run:

```bash
python train.py
```

## Model Selection

The default model used is a **Graph Attention Network (GAT)**, defined in [`model.py`](model.py). You can modify this file to experiment with different Graph Neural Network (GNN) architectures.

## Contributions

Contributions are welcome! Feel free to submit issues or pull requests to improve this project.

## License

This project is licensed under the MIT License.

