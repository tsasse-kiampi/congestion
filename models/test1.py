import torch
from torch_geometric.data import Data, Batch

# Example graphs
graph1 = Data(x=torch.randn(4, 5), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
graph2 = Data(x=torch.randn(3, 5), edge_index=torch.tensor([[0, 1, 2], [1, 0, 1]]))

# Batch the graphs
batch = Batch.from_data_list([graph1, graph2])

# Accessing batch attributes
print("Batch x shape:", batch.x.shape)  # Shape: [total_nodes, node_features]
print("Batch edge_index shape:", batch.edge_index.shape)  # Shape: [2, total_edges]
print(batch.batch)
print(batch)