import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import flwr as fl
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

# Load the full dataset
def load_data():
    df = pd.read_excel("HV Circuit Breaker Maintenance Data.xlsx")

    # Drop the product_variant column (not used as feature) and encode other categoricals
    df = df.drop(columns=["Product_variant"])
    df["Breaker_status"] = LabelEncoder().fit_transform(df["Breaker_status"])
    df["Heater_status"] = LabelEncoder().fit_transform(df["Heater_status"])
    df["Last_trip_coil_energized"] = LabelEncoder().fit_transform(df["Last_trip_coil_energized"])

    X = df.drop(columns=["Maintenance_required"]).values
    y = df["Maintenance_required"].values

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# Partition dataset using IID strategy
def get_partition(partition_id: int, num_partitions: int = 5):
    X, y = load_data()
    total_samples = len(X)

    # Set a seed to ensure consistent shuffling
    np.random.seed(42)
    indices = np.random.permutation(total_samples)

    # Calculate size of each partition
    partition_size = total_samples // num_partitions
    start = partition_id * partition_size
    end = (partition_id + 1) * partition_size if partition_id < num_partitions - 1 else total_samples
    part_indices = indices[start:end]

    X_part, y_part = X[part_indices], y[part_indices]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X_part, y_part, test_size=0.2, random_state=42)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                  torch.tensor(y_train, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(y_test, dtype=torch.float32))
    return train_dataset, test_dataset

# Define simple model
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x)) 
