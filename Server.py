import flwr as fl
import sys
import numpy as np

strategy=fl.server.strategy.FedAvg()

x=fl.server.start_server(
        server_address = "127.0.0.1:16080", 
        config=fl.server.ServerConfig(num_rounds=2), strategy=strategy)


