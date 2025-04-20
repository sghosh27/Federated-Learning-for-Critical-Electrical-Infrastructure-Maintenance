import sys
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from task import get_partition, Net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model.to(DEVICE)
        self.train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = nn.BCELoss()

        for epoch in range(3):  # few local epochs
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).view(-1, 1)
                optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss_fn = nn.BCELoss()
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE).view(-1, 1)
                y_pred = self.model(X_batch)
                loss += loss_fn(y_pred, y_batch).item()
                preds = (y_pred > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total
        return loss / len(self.test_loader), total, {"accuracy": accuracy}

def main():
    client_id = int(sys.argv[1])
    train_data, test_data = get_partition(client_id)
    input_size = train_data[0][0].shape[0]
    model = Net(input_size)

    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient(model, train_data, test_data))

if __name__ == "__main__":
    main()
