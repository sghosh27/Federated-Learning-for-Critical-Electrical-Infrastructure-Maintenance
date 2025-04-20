import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from task import Net, get_partition
import numpy as np
from flwr.common import Context

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flower client class for simulation
class SimulatedClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.cid = int(cid)
        self.train_data, self.test_data = get_partition(self.cid)
        self.model = Net(self.train_data[0][0].shape[0]).to(DEVICE)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(),
                              [torch.tensor(p) for p in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()

        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=32, shuffle=True)
        for epoch in range(3):
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).view(-1, 1)
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config), len(self.train_data), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_fn = nn.BCELoss()
        loss, correct, total = 0.0, 0, 0

        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=32)
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).view(-1, 1)
                preds = self.model(X_batch)
                loss += loss_fn(preds, y_batch).item()
                correct += (preds.round() == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total
        return loss / len(test_loader), len(self.test_data), {"accuracy": accuracy}


# Evaluation function for server
def get_eval_fn():
    _, test_data = get_partition(0)  # Just pick one for global validation
    input_size = test_data[0][0].shape[0]

    def evaluate(weights):
        model = Net(input_size).to(DEVICE)
        state_dict = dict(zip(model.state_dict().keys(),
                              [torch.tensor(p) for p in weights]))
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
        loss_fn = nn.BCELoss()
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).view(-1, 1)
                preds = model(X_batch)
                loss += loss_fn(preds, y_batch).item()
                correct += (preds.round() == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total
        print(f"🔎 Global Evaluation - Loss: {loss / len(test_loader):.4f}, Accuracy: {accuracy:.4f}")
        return loss / len(test_loader), {"accuracy": accuracy}

    return evaluate

# Start the simulation
def main():
    num_clients = 5

    def client_fn(cid: str):
        return SimulatedClient(cid).to_client()

    #def client_fn(cid: str):
    #    return SimulatedClient(cid)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        #fraction_eval=1.0,
        min_fit_clients=num_clients,
        #min_eval_clients=num_clients,
        min_available_clients=num_clients,
        #eval_fn=get_eval_fn(),
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
