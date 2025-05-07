# -*- coding: utf-8 -*-
# Federated Learing with HV Circuit Breaker Maintenance Data Advanced - Functions

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
from collections import OrderedDict

# -------------------------------
# Non-IID Data Partitioning
# -------------------------------

def create_dirichlet_non_iid(df, num_clients=5, alpha=0.5, label_col='Maintenance_required'):
    """
    Create Non-IID client datasets using a Dirichlet distribution.

    Args:
        df (pd.DataFrame): Input DataFrame.
        num_clients (int): Number of clients.
        alpha (float): Dirichlet distribution parameter controlling label distribution skew.
        label_col (str): Column name for labels.

    Returns:
        list: List of TensorFlow datasets for each client.
    """
    label_indices = {}
    for label in df[label_col].unique():
        label_indices[label] = np.where(df[label_col] == label)[0]

    client_indices = [[] for _ in range(num_clients)]
    for label, indices in label_indices.items():
        np.random.shuffle(indices)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        splits = np.split(indices, proportions)
        for client_id, idx in enumerate(splits):
            client_indices[client_id] += idx.tolist()

    client_datasets = []
    for client_idx in client_indices:
        df_client = df.iloc[client_idx]
        labels = df_client.pop(label_col)
        dataset = tf.data.Dataset.from_tensor_slices((OrderedDict(df_client.items()), labels)).shuffle(100).batch(10)
        client_datasets.append(dataset)
    return client_datasets


def create_feature_skew_non_iid(df, num_clients=5, feature='SF6_density', label_col='Maintenance_required'):
    """
    Create Non-IID client datasets by sorting data based on a specific feature.

    Args:
        df (pd.DataFrame): Input DataFrame.
        num_clients (int): Number of clients.
        feature (str): Feature column to sort by.
        label_col (str): Column name for labels.

    Returns:
        list: List of TensorFlow datasets for each client.
    """
    df_sorted = df.sort_values(by=feature)
    client_dataframes = np.array_split(df_sorted, num_clients)
    client_datasets = []
    for df_client in client_dataframes:
        labels = df_client.pop(label_col)
        dataset = tf.data.Dataset.from_tensor_slices((OrderedDict(df_client.items()), labels)).shuffle(100).batch(10)
        client_datasets.append(dataset)
    return client_datasets


# -------------------------------
# Model Creation Functions
# -------------------------------

def model_fn(input_spec):
    """
    Create a Keras model for federated learning.

    Args:
        input_spec (tuple): Input specification for the model.

    Returns:
        tf.keras.Model: Keras model.
    """
    feature_spec = input_spec[0]
    inputs = {name: tf.keras.Input(shape=(1,), name=name) for name in feature_spec.keys()}
    x = tf.keras.layers.Concatenate()(list(inputs.values()))
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def model_fn_with_bn(input_spec):
    """
    Create a Keras model with Batch Normalization for FedBN.

    Args:
        input_spec (tuple): Input specification for the model.

    Returns:
        tf.keras.Model: Keras model with Batch Normalization.
    """
    feature_spec = input_spec[0]
    inputs = {name: tf.keras.Input(shape=(1,), name=name) for name in feature_spec.keys()}
    x = tf.keras.layers.Concatenate()(list(inputs.values()))
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# -------------------------------
# Federated Learning Algorithms
# -------------------------------

def build_fedavg(input_spec, momentum=0.0):
    """
    Build the FedAvg algorithm for federated learning.

    Args:
        input_spec (tuple): Input specification for the model.
        momentum (float): Momentum parameter for the optimizer.

    Returns:
        tff.learning.templates.IterativeProcess: Federated averaging process.
    """
    def client_opt_fn():
        if momentum == 0.0:
            return tf.keras.optimizers.SGD(learning_rate=0.02)
        else:
            return tf.keras.optimizers.SGD(learning_rate=0.02, momentum=momentum)

    return tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=lambda: tff.learning.models.from_keras_model(
            keras_model=model_fn(input_spec),
            input_spec=input_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        ),
        client_optimizer_fn=client_opt_fn,
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )


def build_fedprox(input_spec, proximal_mu=0.1):
    """
    Build the FedProx algorithm for federated learning.

    Args:
        input_spec (tuple): Input specification for the model.
        proximal_mu (float): Proximal term weight for FedProx.

    Returns:
        tff.learning.templates.IterativeProcess: Federated proximal process.
    """
    return tff.learning.algorithms.build_weighted_fed_prox(
        model_fn=lambda: tff.learning.models.from_keras_model(
            keras_model=model_fn(input_spec),
            input_spec=input_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        ),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        proximal_strength=proximal_mu
    )


def build_fedbn(input_spec):
    """
    Build the FedBN algorithm for federated learning.

    Args:
        input_spec (tuple): Input specification for the model.

    Returns:
        tff.learning.templates.IterativeProcess: Federated Batch Normalization process.
    """
    return tff.learning.algorithms.build_weighted_fed_avg(
        model_fn=lambda: tff.learning.models.from_keras_model(
            keras_model=model_fn_with_bn(input_spec),
            input_spec=input_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()]
        ),
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )


# -------------------------------
# Training and Evaluation
# -------------------------------

def train_process(algorithm, client_datasets, num_rounds=20):
    """
    Train a federated learning algorithm.

    Args:
        algorithm (tff.learning.templates.IterativeProcess): Federated learning algorithm.
        client_datasets (list): List of client datasets.
        num_rounds (int): Number of training rounds.

    Returns:
        list: List of accuracy metrics for each round.
    """
    state = algorithm.initialize()
    acc_list = []
    for round_num in range(1, num_rounds + 1):
        state, metrics = algorithm.next(state, client_datasets)
        acc = metrics['client_work']['train']['binary_accuracy']
        acc_list.append(acc)
        print(f"Round {round_num}: Accuracy={acc:.4f}")
    return acc_list


def compare_algorithms(client_datasets, title):
    """
    Compare the performance of FedAvg and FedAvg with Momentum.

    Args:
        client_datasets (list): List of client datasets.
        title (str): Title for the comparison plot.

    Returns:
        None
    """
    input_spec = client_datasets[0].element_spec

    # Build algorithms
    fedavg = build_fedavg(input_spec)
    fedavg_momentum = build_fedavg(input_spec, momentum=0.9)

    # Train algorithms
    acc_fedavg = train_process(fedavg, client_datasets)
    acc_fedavg_momentum = train_process(fedavg_momentum, client_datasets)

    # Plot results
    rounds = list(range(1, len(acc_fedavg) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, acc_fedavg, label='FedAvg')
    plt.plot(rounds, acc_fedavg_momentum, label='FedAvg + Momentum')
    plt.title(f'Accuracy over Rounds - {title}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()


def compare_all_algorithms(client_datasets, title, num_rounds=10):
    """
    Compare the performance of multiple federated learning algorithms.

    Args:
        client_datasets (list): List of client datasets.
        title (str): Title for the comparison plot.
        num_rounds (int): Number of training rounds.

    Returns:
        None
    """
    input_spec = client_datasets[0].element_spec

    # Build algorithms
    fedavg = build_fedavg(input_spec)
    fedavg_momentum = build_fedavg(input_spec, momentum=0.9)
    fedprox = build_fedprox(input_spec)
    fedbn = build_fedbn(input_spec)

    # Train algorithms
    acc_fedavg = train_process(fedavg, client_datasets, num_rounds)
    acc_fedavg_momentum = train_process(fedavg_momentum, client_datasets, num_rounds)
    acc_fedprox = train_process(fedprox, client_datasets, num_rounds)
    acc_fedbn = train_process(fedbn, client_datasets, num_rounds)

    # Plot results
    rounds = list(range(1, len(acc_fedavg) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, acc_fedavg, marker='o', label='FedAvg')
    plt.plot(rounds, acc_fedavg_momentum, marker='x', label='FedAvg + Momentum')
    plt.plot(rounds, acc_fedprox, marker='s', label='FedProx')
    plt.plot(rounds, acc_fedbn, marker='^', label='FedBN')
    plt.title(f'Accuracy over Rounds - {title}')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.show()


def train_process_per_client(algorithm, client_datasets, num_rounds=10):
    """
    Train a federated learning algorithm and track per-client accuracy.

    Args:
        algorithm (tff.learning.templates.IterativeProcess): Federated learning algorithm.
        client_datasets (list): List of client datasets.
        num_rounds (int): Number of training rounds.

    Returns:
        list: Per-client accuracy metrics for each round.
    """
    state = algorithm.initialize()
    per_round_client_accuracies = []

    for round_num in range(1, num_rounds + 1):
        state, metrics = algorithm.next(state, client_datasets)
        client_metrics = metrics['client_work']['train']

        # Save per-client accuracy
        round_client_acc = client_metrics['binary_accuracy']
        per_round_client_accuracies.append(round_client_acc)

        print(f"Round {round_num}: Global Accuracy={round_client_acc:.4f}")

    return per_round_client_accuracies


def plot_per_client(per_client_acc_list, title):
    """
    Plot per-client average accuracy over rounds.

    Args:
        per_client_acc_list (list): List of per-client accuracy metrics.
        title (str): Title for the plot.

    Returns:
        None
    """
    rounds = list(range(1, len(per_client_acc_list) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, per_client_acc_list, marker='o')
    plt.title(f'Per-Client Average Accuracy Over Rounds - {title}')
    plt.xlabel('Round')
    plt.ylabel('Average Client Accuracy')
    plt.grid(True)
    plt.show()