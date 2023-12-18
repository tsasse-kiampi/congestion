from model import CongestionWrapperEncoder1, CustomLearnableEmbedding
import torch
from parameters import CONGESTION_EMBEDDING_DIM, CONGESTION_SPACE_CARDINAL

num_nodes = 10

cong = torch.randint(1, 10, size=(10, num_nodes)).unsqueeze(0)
cong = torch.cat([cong,cong])
print(cong)
print(cong.shape)

edge_index = torch.randint(high=num_nodes-1, size=(2, num_nodes**2//10))
print(edge_index)
x = CongestionWrapperEncoder1(CustomLearnableEmbedding, num_nodes, CONGESTION_SPACE_CARDINAL, CONGESTION_EMBEDDING_DIM, CONGESTION_EMBEDDING_DIM, 1, 0.1)
# print(x(cong, edge_index))
print(x(cong, edge_index).shape)