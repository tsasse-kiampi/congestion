from model import CongestionLearnableEmbedding, CongestionWrapperEncoder
import torch
from parameters import CONGESTION_EMBEDDING_DIM

num_nodes = 10

cong = torch.randint(1, 10, size=(10, num_nodes)).unsqueeze(0)
print(cong)


edge_index = torch.randint(high=num_nodes-1, size=(2, num_nodes**2//10))
print(edge_index)
x = CongestionWrapperEncoder(CongestionLearnableEmbedding, num_nodes, CONGESTION_EMBEDDING_DIM, CONGESTION_EMBEDDING_DIM, 1, 0.1)
# print(x(cong, edge_index))
print(x(cong, edge_index).shape)