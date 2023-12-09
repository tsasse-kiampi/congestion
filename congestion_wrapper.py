import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import Data


edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
x = torch.rand(3, 16)  

data = Data(x=x, edge_index=edge_index)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=1, dropout=0.6):
        super().__init__()
        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    def forward(self, data):
        x = self.gat_conv(data.x, data.edge_index)
        return x


input_features = 16
output_features = 8
num_heads = 2

graph_attention_layer = GraphAttentionLayer(input_features, output_features, heads=num_heads)

output = graph_attention_layer(data)

print(output)