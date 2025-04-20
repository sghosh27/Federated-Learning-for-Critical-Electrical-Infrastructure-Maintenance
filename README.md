# Federated-Learning-for-Critical-Electrical-Infrastructure-Maintenance

Implementation of the paper "Federated Learning for Critical Electrical Infrastructure - Handling Non-IID Data for Predictive Maintenance of Substation Equipment" by Soham Ghosh and Gaurav Mittal.


├── federated_cb_maintenance \
│   ├── __init__.py
│   ├── client_app.py             # Defines your ClientApp \
│   ├── server_app.py             # Defines your ServerApp \
│   └── task.py                   # Defines your model, training and data loading \
│   └── simulation_app.py         # Runs entire federated learning experiment in simulation mode \
├── pyproject.toml                # Project metadata like dependencies and configs \
└── README.md \
