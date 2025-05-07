# -*- coding: utf-8 -*-
# Federated Learning with HV Circuit Breaker Maintenance Data - Functions

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -------------------------------
# Data Preprocessing Functions
# -------------------------------

def preprocess_breaker_data(df):
    """
    Preprocess the HV Circuit Breaker Maintenance Data.

    Args:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple: Processed features (X), labels (y), and the scaler object.
    """
    df = df.copy()

    # Convert categorical columns to numerical
    df['Breaker_status'] = df['Breaker_status'].map({'Closed': 0, 'Open': 1}).astype(np.float32)
    df['Heater_status'] = df['Heater_status'].map({'Off': 0, 'On': 1}).astype(np.float32)
    df['Last_trip_coil_energized'] = df['Last_trip_coil_energized'].str.replace('TC', '').astype(np.float32)

    # Separate features and labels
    y = df['Maintenance_required'].values.reshape(-1, 1).astype(np.float32)
    features = df.drop(['Product_variant', 'Maintenance_required'], axis=1)

    # Scale numerical features
    numerical_cols = features.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    features[numerical_cols] = scaler.fit_transform(features[numerical_cols])

    # Convert all to float32
    X = features.astype(np.float32).values
    return X, y, scaler


# -------------------------------
# Federated Data Preparation
# -------------------------------

def create_client_dataset(X, y):
    """
    Create a TensorFlow Dataset for a single client.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.

    Returns:
        tf.data.Dataset: Batched dataset for the client.
    """
    return tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(X, dtype=tf.float32),
         tf.convert_to_tensor(y, dtype=tf.float32))
    ).batch(32)


def create_iid_federated_data(X, y, num_clients=5):
    """
    Create IID federated data by evenly splitting the dataset among clients.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        num_clients (int): Number of clients.

    Returns:
        list: List of TensorFlow datasets for each client.
    """
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    client_datasets = []
    for i in range(num_clients):
        start = i * len(X) // num_clients
        end = (i + 1) * len(X) // num_clients
        client_datasets.append(create_client_dataset(X_shuffled[start:end], y_shuffled[start:end]))
    return client_datasets


def create_non_iid_federated_data(df, num_clients=5):
    """
    Create Non-IID federated data by grouping data based on product variants.

    Args:
        df (pd.DataFrame): Input DataFrame.
        num_clients (int): Number of clients.

    Returns:
        list: List of TensorFlow datasets for each client.
    """
    client_data = [[] for _ in range(num_clients)]

    for i, (variant, group) in enumerate(df.groupby('Product_variant')):
        client_idx = i % num_clients
        X, y, _ = preprocess_breaker_data(group)
        client_data[client_idx].append((X, y))

    federated_datasets = []
    for client in client_data:
        if client:
            X_client = np.concatenate([x for x, _ in client])
            y_client = np.concatenate([y for _, y in client])
            federated_datasets.append(create_client_dataset(X_client, y_client))

    return federated_datasets


def create_client_data(X, y, num_clients=5, partition_type='iid'):
    """
    Create client data based on the specified partition type.

    Args:
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        num_clients (int): Number of clients.
        partition_type (str): Partition type ('iid', 'label_skew', 'feature_skew').

    Returns:
        list: List of client data tuples (X, y).
    """
    client_data = []

    if partition_type == 'iid':
        data_per_client = len(X) // num_clients
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        for i in range(num_clients):
            start, end = i * data_per_client, (i + 1) * data_per_client
            client_data.append((X[indices[start:end]], y[indices[start:end]]))

    elif partition_type == 'label_skew':
        labels = y.flatten().astype(int)
        unique_labels = np.unique(labels)
        label_chunks = np.array_split(unique_labels, num_clients)

        for label_set in label_chunks:
            idx = np.where(np.isin(labels, label_set))[0]
            np.random.shuffle(idx)
            client_data.append((X[idx], y[idx]))

    elif partition_type == 'feature_skew':
        kmeans = KMeans(n_clusters=num_clients, random_state=42).fit(X)
        clusters = kmeans.labels_
        for i in range(num_clients):
            idx = np.where(clusters == i)[0]
            client_data.append((X[idx], y[idx]))

    return client_data


# -------------------------------
# Model Creation Functions
# -------------------------------

def create_model(X_train):
    """
    Create a simple feedforward neural network model.

    Args:
        X_train (np.ndarray): Training features.

    Returns:
        tf.keras.Model: Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_model_fedbn(X_train):
    """
    Create a simple feedforward neural network model for FedBN (no batch normalization).

    Args:
        X_train (np.ndarray): Training features.

    Returns:
        tf.keras.Model: Keras model.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model


# -------------------------------
# Federated Learning Processes
# -------------------------------

def model_fn(X_train):
    """
    Wrap the Keras model for TFF with input specifications and loss function.

    Args:
        X_train (np.ndarray): Training features.

    Returns:
        tff.learning.Model: TFF model.
    """
    keras_model = create_model(X_train)
    input_spec = (
        tf.TensorSpec(shape=[None, X_train.shape[1]], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
    )
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

def build_iid_fedaverage_process(X_train):
    """
    Build the federated averaging process for IID data.

    Args:
        X_train (np.ndarray): Training features.

    Returns:
        tff.learning.templates.IterativeProcess: Federated averaging process.
    """
    no_arg_model_fn = lambda: model_fn(X_train)
    return tff.learning.algorithms.build_weighted_fed_avg(
        no_arg_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.01)
    )

def build_non_iid_fedaverage_process(X_train, MU=0.1):
    """
    Build the federated averaging process for Non-IID data with a custom FedProx optimizer.

    Args:
        X_train (np.ndarray): Training features.
        MU (float): Proximal term weight for FedProx.

    Returns:
        tff.learning.templates.IterativeProcess: Federated averaging process.
    """
    no_arg_model_fn = lambda: model_fn(X_train)
    return tff.learning.algorithms.build_weighted_fed_avg(
        no_arg_model_fn,
        client_optimizer_fn=lambda: fedprox_optimizer(0.001, MU),
        server_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.01)
    )


def fedprox_optimizer(learning_rate=0.001, MU=0.1):
    """
    Create a custom FedProx optimizer with a proximal term.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        MU (float): Proximal term weight.

    Returns:
        tf.keras.optimizers.Optimizer: Custom FedProx optimizer.
    """
    class FedProxOptimizer(tf.keras.optimizers.Adam):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.global_weights = None

        def minimize(self, loss, var_list, global_weights=None):
            if global_weights is not None:
                self.global_weights = [tf.identity(w) for w in global_weights]
            return super().minimize(loss, var_list)

        def apply_gradients(self, grads_and_vars, name=None):
            if self.global_weights is not None:
                # Add proximal term to gradients
                new_grads_and_vars = []
                for (g, v), w in zip(grads_and_vars, self.global_weights):
                    if g is not None:
                        g += MU * (v - w)  # Proximal term
                    new_grads_and_vars.append((g, v))
                grads_and_vars = new_grads_and_vars
            return super().apply_gradients(grads_and_vars, name)

    return FedProxOptimizer(learning_rate)


def convert_to_tff_client_data(client_data):
    """
    Convert client data into TensorFlow Federated (TFF) format.

    Args:
        client_data (list): List of client datasets (features and labels).

    Returns:
        list: List of TFF datasets for each client.
    """
    def client_dataset_fn(x, y):
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(16)

    return [client_dataset_fn(x, y) for x, y in client_data]


def train_fedavg(client_data, num_rounds=20, X_train=None):
    """
    Train a federated model using FedAvg.

    Args:
        client_data (list): List of client datasets.
        num_rounds (int): Number of training rounds.
        X_train (np.ndarray): Training features.

    Returns:
        list: Training history containing metrics for each round.
    """
    federated_data = convert_to_tff_client_data(client_data)
    no_arg_model_fn = lambda: model_fn(X_train)
    fedavg_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=no_arg_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
    )
    state = fedavg_process.initialize()
    history = []
    for round_num in range(1, num_rounds + 1):
        state, metrics = fedavg_process.next(state, federated_data)
        history.append(metrics)
        print(f"Round {round_num}, Metrics: {metrics}")
    return history


def fedprox_loss_fn(base_loss_fn, global_weights, model, mu):
    """
    Create a custom loss function for FedProx with a proximal term.

    Args:
        base_loss_fn (callable): Base loss function (e.g., BinaryCrossentropy).
        global_weights (list): Global model weights.
        model (tf.keras.Model): Local model.
        mu (float): Proximal term weight.

    Returns:
        callable: Custom loss function with a proximal term.
    """
    def loss(y_true, y_pred):
        base_loss = base_loss_fn(y_true, y_pred)
        prox_term = tf.add_n([
            tf.reduce_sum(tf.square(var - gw))
            for var, gw in zip(model.trainable_variables, global_weights)
        ])
        return base_loss + (mu / 2.0) * prox_term
    return loss


def train_fedprox(client_data, num_rounds=20, mu=0.1, X_train=None):
    """
    Train a federated model using FedProx.

    Args:
        client_data (list): List of client datasets.
        num_rounds (int): Number of training rounds.
        mu (float): Proximal term weight.
        X_train (np.ndarray): Training features.

    Returns:
        list: Training history containing metrics for each round.
    """
    federated_data = convert_to_tff_client_data(client_data)
    no_arg_model_fn = lambda: model_fn(X_train)

    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=no_arg_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

    state = iterative_process.initialize()
    history = []

    for round_num in range(1, num_rounds + 1):
        state, metrics = iterative_process.next(state, federated_data)
        history.append(metrics)
        print(f"Round {round_num}, Metrics: {metrics}")

    return history

def model_fn_fedbn(X_train):
    """
    Wrap the Keras model for TFF with input specifications and loss function for FedBN.

    Args:
        X_train (np.ndarray): Training features.

    Returns:
        tff.learning.Model: TFF model for FedBN.
    """
    keras_model = create_model_fedbn(X_train)
    input_spec = (
        tf.TensorSpec(shape=[None, X_train.shape[1]], dtype=tf.float32),
        tf.TensorSpec(shape=[None, 1], dtype=tf.float32)
    )
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )


def train_fedbn(client_data, num_rounds=20, X_train=None):
    """
    Train a federated model using FedBN (Federated Batch Normalization).

    Args:
        client_data (list): List of client datasets.
        num_rounds (int): Number of training rounds.
        X_train (np.ndarray): Training features.

    Returns:
        list: Training history containing metrics for each round.
    """
    federated_data = convert_to_tff_client_data(client_data)
    no_arg_model_fn_fedbn = lambda: model_fn_fedbn(X_train)

    iterative_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=no_arg_model_fn_fedbn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.01),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )

    state = iterative_process.initialize()
    history = []

    for round_num in range(1, num_rounds + 1):
        state, metrics = iterative_process.next(state, federated_data)
        history.append(metrics)
        print(f"Round {round_num}, Metrics: {metrics}")

    return history


# -------------------------------
# Visualization and Summarization
# -------------------------------

def plot_training_curves(histories, labels, title="Training Metrics"):
    """
    Plot training accuracy over rounds for multiple algorithms.

    Args:
        histories (list): List of training histories.
        labels (list): List of algorithm labels.
        title (str): Plot title.
    """
    rounds = len(histories[0])
    metrics_per_round = {
        label: [round['train']['binary_accuracy'] for round in hist]
        for hist, label in zip(histories, labels)
    }

    plt.figure(figsize=(12, 5))
    for label, acc in metrics_per_round.items():
        plt.plot(range(1, rounds + 1), acc, label=label)
    plt.xlabel("Rounds")
    plt.ylabel("Train Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def summarize_iid_accuracy(histories, labels):
    """
    Summarize the final training accuracy for IID data.

    Args:
        histories (list): List of training histories.
        labels (list): List of algorithm labels.

    Returns:
        pd.DataFrame: Summary table of final accuracies.
    """
    final_acc = [rounds[-1]['client_work']['train']['binary_accuracy'] for rounds in histories]
    df = pd.DataFrame({
        "Algorithm": labels,
        "Final Train Accuracy": final_acc
    })
    return df


def plot_training_curves_client_work(histories, labels, title="Training Metrics"):
    """
    Plot training accuracy over rounds for client work.

    Args:
        histories (list): List of training histories.
        labels (list): List of algorithm labels.
        title (str): Plot title.
    """
    rounds = len(histories[0])
    metrics_per_round = {
        label: [round['client_work']['train']['binary_accuracy'] for round in hist]
        for hist, label in zip(histories, labels)
    }

    plt.figure(figsize=(12, 5))
    for label, acc in metrics_per_round.items():
        plt.plot(range(1, rounds + 1), acc, label=label)
    plt.xlabel("Rounds")
    plt.ylabel("Train Accuracy")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def summarize_non_iid_accuracy(histories, labels):
    """
    Summarize the final training accuracy for Non-IID data.

    Args:
        histories (list): List of training histories.
        labels (list): List of algorithm labels.

    Returns:
        pd.DataFrame: Summary table of final accuracies.
    """
    final_acc = [rounds[-1]['client_work']['train']['binary_accuracy'] for rounds in histories]
    df = pd.DataFrame({
        "Algorithm": labels,
        "Final Train Accuracy": final_acc
    })
    return df