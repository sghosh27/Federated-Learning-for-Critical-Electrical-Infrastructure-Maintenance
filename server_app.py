import flwr as fl
from task import Net, load_data
import torch

# Get model input size
X, _ = load_data()
input_size = X.shape[1]

# Define evaluation function (server-side aggregation validation)
def get_evaluate_fn():
    _, test_dataset = load_data(), None
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(_, dtype=torch.float32).view(-1, 1)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_tensor, y_tensor), batch_size=32)

    def evaluate(weights):
        model = Net(input_size)
        params_dict = zip(model.state_dict().keys(), weights)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        loss_fn = torch.nn.BCELoss()
        loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch)
                loss += loss_fn(y_pred, y_batch).item()
                preds = (y_pred > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

        accuracy = correct / total
        return loss / len(test_loader), {"accuracy": accuracy}

    return evaluate

# Configure server with FedAvg
def main():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_eval=1.0,
        min_fit_clients=5,
        min_eval_clients=5,
        min_available_clients=5,
        eval_fn=get_evaluate_fn()
    )

    fl.server.start_server(server_address="127.0.0.1:8080", config=fl.server.ServerConfig(num_rounds=5), strategy=strategy)

if __name__ == "__main__":
    main()
