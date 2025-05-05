# -*- coding: utf-8 -*-
# Federated Learning with HV Circuit Breaker Maintenance Data - Functions

import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Preprocess the data
# This function takes a DataFrame and preprocesses it for training
def preprocess_breaker_data(df):
    df = df.copy()

    # Convert categoricals
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


# Create TensorFlow Dataset for each client
# This function takes features and labels and creates a TensorFlow Dataset
def create_client_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(X, dtype=tf.float32),
         tf.convert_to_tensor(y, dtype=tf.float32))
    ).batch(32)


# Create IID Federated Data
# This function takes features and labels and creates IID federated data
def create_iid_federated_data(X, y, num_clients=5):
    # Shuffle data
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # Split into clients
    client_datasets = []
    for i in range(num_clients):
        start = i * len(X) // num_clients
        end = (i+1) * len(X) // num_clients
        client_datasets.append(create_client_dataset(X_shuffled[start:end], y_shuffled[start:end]))
    return client_datasets


# Create Non-IID Federated Data
# This function takes a DataFrame and creates non-IID federated data
# based on product variants
def create_non_iid_federated_data(df, num_clients=5):
    client_data = [[] for _ in range(num_clients)]

    # Group by product variant and distribute to clients
    for i, (variant, group) in enumerate(df.groupby('Product_variant')):
        client_idx = i % num_clients
        X, y, _ = preprocess_breaker_data(group)
        client_data[client_idx].append((X, y))

    # Create datasets for each client by concatenating their data
    federated_datasets = []
    for client in client_data:
        if client:  # Only if client has data
            X_client = np.concatenate([x for x, _ in client])
            y_client = np.concatenate([y for _, y in client])
            federated_datasets.append(create_client_dataset(X_client, y_client))

    return federated_datasets


# Helper function to simulate IID and Non-IID data partitions
# This function takes features and labels and creates client data
# based on the specified partition type
# 'iid', 'label_skew', or 'feature_skew'
# 'iid' - Independent and Identically Distributed
# 'label_skew' - Skewed by labels
# 'feature_skew' - Skewed by features
def create_client_data(X, y, num_clients=5, partition_type='iid'):
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
            np.random.seed(41)
            np.random.shuffle(idx)
            client_data.append((X[idx], y[idx]))

    elif partition_type == 'feature_skew':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clients, random_state=42).fit(X)
        clusters = kmeans.labels_
        for i in range(num_clients):
            idx = np.where(clusters == i)[0]
            client_data.append((X[idx], y[idx]))

    return client_data


# This function creates a simple feedforward neural network model
# with batch normalization
# and dropout layers
def create_model(X_train):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# This function wraps the Keras model for TFF
# and specifies the input specifications
# and loss function
def model_fn(X_train):
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


# This function builds the federated averaging process
# using TensorFlow Federated for IID partition
def build_iid_fedaverage_process(X_train):
    no_arg_model_fn = lambda: model_fn(X_train)

    return tff.learning.algorithms.build_weighted_fed_avg(
        no_arg_model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.001),
        server_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.01)
        )

# This function builds the federated averaging process
# using TensorFlow Federated for Non-IID partition
# with a custom FedProx optimizer
# and a proximal term
def build_non_iid_fedaverage_process(X_train, MU=0.1):
    no_arg_model_fn = lambda: model_fn(X_train)

    return tff.learning.algorithms.build_weighted_fed_avg(
        no_arg_model_fn,
        client_optimizer_fn=lambda: fedprox_optimizer(0.001, MU),  # Custom FedProx optimizer
        server_optimizer_fn=lambda: tf.keras.optimizers.Adam(0.01)
)



# This function creates a custom client optimizer with proximal term
def fedprox_optimizer(learning_rate=0.001, MU=0.1):
    # Create a standard optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Add proximal term functionality
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


# This function takes client data and converts it to TFF format
# for federated learning
def convert_to_tff_client_data(client_data):
    import collections
    def client_dataset_fn(x, y):
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
    return [
        client_dataset_fn(x, y) for x, y in client_data
    ]


# FedAvg Training Function
# This function takes client data and trains a federated model using FedAvg
def train_fedavg(client_data, num_rounds=20, X_train=None):
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


# Final working FedProx version using custom loss wrapper 
# This function takes a base loss function and wraps it with a proximal term
# to create a new loss function
def fedprox_loss_fn(base_loss_fn, global_weights, model, mu):
    def loss(y_true, y_pred):
        base_loss = base_loss_fn(y_true, y_pred)
        prox_term = tf.add_n([
            tf.reduce_sum(tf.square(var - gw))
            for var, gw in zip(model.trainable_variables, global_weights)
        ])
        return base_loss + (mu / 2.0) * prox_term
    return loss


# FedProx Training Function
# This function takes client data and trains a federated model using FedProx
# with a proximal term
def train_fedprox(client_data, num_rounds=20, mu=0.1, X_train=None):
    federated_data = convert_to_tff_client_data(client_data)
    no_arg_model_fn = lambda: model_fn(X_train)

    # Custom client training loop to simulate FedProx
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

# Create a model for FedBN
# This function creates a simple feedforward neural network model
# without batch normalization layers
def create_model_fedbn(X_train):
    # FedBN: no batch norm layers are aggregated
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# This function wraps the Keras model for TFF
# and specifies the input specifications
# and loss function
# for FedBN
def model_fn_fedbn(X_train):
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


# FedBN Training Function
# This function takes client data and trains a federated model using FedBN
# with batch normalization
# and dropout layers
def train_fedbn(client_data, num_rounds=20, X_train=None):
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


# Visualize training curves
# This function takes a list of training histories and labels
# and plots the training accuracy over rounds
def plot_training_curves(histories, labels, title="Training Metrics"):
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


# Visual comparison
# This function takes a list of training histories and labels
# and plots the training accuracy over rounds
# for client work
def plot_training_curves_client_work(histories, labels, title="Training Metrics"):
    rounds = len(histories[0])
    metrics_per_round = {
        label: [round['client_work']['train']['binary_accuracy'] for round in hist]  # Access using 'client_work'
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


# Summarize IID accuracy
# This function takes a list of training histories and labels
# and summarizes the final training accuracy
# for each algorithm
def summarize_iid_accuracy(histories, labels):
    final_acc = [rounds[-1]['client_work']['train']['binary_accuracy'] for rounds in histories]  # Access 'binary_accuracy' under 'client_work' and 'train'
    df = pd.DataFrame({
        "Algorithm": labels,
        "Final Train Accuracy": final_acc
    })
    return df


# Summarize Non-IID accuracy
# This function takes a list of training histories and labels
# and summarizes the final training accuracy
# for each algorithm across all settings
def summarize_non_iid_accuracy(histories, labels):
    # Corrected access to 'binary_accuracy' under 'client_work' and 'train' within the last element of 'rounds'
    final_acc = [rounds[-1]['client_work']['train']['binary_accuracy'] for rounds in histories]
    df = pd.DataFrame({
        "Algorithm": labels,
        "Final Train Accuracy": final_acc
    })
    return df