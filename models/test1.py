import torch
from torch_geometric.data import Data, Batch
from torch import nn

embeddings = nn.Embedding(7, 4)
# print(embeddings.weight)

x1 = nn.Parameter(torch.randn(3,5))
x2 = nn.Parameter(torch.randn(4,5))
print(x1.grad)

# x1 = embeddings(torch.tensor([1,2,3,4]))
# x2 = embeddings(torch.tensor([5,6,6]))


# Example graphs
graph1 = Data(x=x1, edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]))
graph2 = Data(x=x2, edge_index=torch.tensor([[0, 1, 2], [1, 0, 1]]))

# Batch the graphs
batch = Batch.from_data_list([graph1, graph2])

# output = None
# for b in batch:
#     output += b.x.sum()

# Accessing batch attributes
# print(output)
print("Batch x shape:", batch.x)  # Shape: [total_nodes, node_features]
print("Batch edge_index shape:", batch.edge_index.shape)  # Shape: [2, total_edges]
# print(batch.batch)
# output = batch.x.sum()
# output.backward()

# batch.x.sum().backward()
# print(x2.grad)

# print(batch.x.grad_fn.bac)