from model import CongestionWrapperEncoder1, CustomLearnableEmbedding, GATDecoder1
import torch
from parameters import CONGESTION_EMBEDDING_DIM, CONGESTION_SPACE_CARDINAL

num_nodes = 10

cong = torch.rand(2, 1, 50).unsqueeze(0)
# cong = torch.cat([cong,cong])

target_cong = torch.randint(1, 10, size=(1, num_nodes)).unsqueeze(0)
target_cong = torch.cat([target_cong,target_cong])

print(cong)
print(cong.shape)
print(target_cong.shape, 'target')

edge_index = torch.randint(high=num_nodes-1, size=(2, num_nodes**2//10))

print(edge_index)
x, _ = GATDecoder1()(cong, edge_index, target_cong)
# print(x(cong, edge_index))
print(x)