from torchinfo import summary
from model import CustomLearnableEmbedding, CongestionWrapperEncoder1, CongestionModel, CongestionTransformerDecoder, GATDecoder1
import torch
from parameters import CONGESTION_EMBEDDING_DIM
num_nodes = 10
cong = torch.randint(1, 10, size=(10, num_nodes)).unsqueeze(0)
cong = torch.cat([cong,cong,cong])
print(cong)

target_cong = torch.randint(1, 10, size=(1, num_nodes)).unsqueeze(0)
target_cong = torch.cat([target_cong,target_cong, target_cong])

edge_index = torch.randint(high=num_nodes-1, size=(2, num_nodes**2//10))

# x = CongestionWrapperEncoder(CongestionLearnableEmbedding, num_nodes, CONGESTION_EMBEDDING_DIM, CONGESTION_EMBEDDING_DIM, 1, 0.1)
# print(x(cong, edge_index))
# x(cong, edge_index).shape

model = CongestionModel(CustomLearnableEmbedding, CongestionWrapperEncoder1, CongestionTransformerDecoder, GATDecoder1)
x, _ = model(cong, edge_index, target_cong)
print(x.indices)
print(_, 'loss')

device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")
model.to(device)

# Print a summary of the model
summary(model, input_names=["x", "edge_index"],  depth=3, col_names=[ "num_params", "trainable"])
summary(model, input_names=["x", "edge_index"])
# summary(model, input_size=(1, 10, 32, 32), device="cpu", depth=None, col_names=["output_size", "num_params"])
