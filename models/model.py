import torch
from torch import nn
from torch.nn import functional
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from parameters import DIM, ATTENTION_BLOCKS, CONGESTION_EMBEDDING_DIM, CONGESTION_SPACE_CARDINAL, DEFAULT_TERMINAL_NUMBER, MAX_CONGESTION, ATTENTION_HEADS, DROPOUT

def congestion_to_index(congestion): #TODO maybe indices are precalculated in dataloader?
    return DEFAULT_TERMINAL_NUMBER*round(congestion)/MAX_CONGESTION

class CongestionNonLearnableEmbedding(nn.Module):
    r"""Tokenization of congestion
        
        Each indice of embeddings represents congestion in hours (and minutes)
    """
    def __init__(self, num_tokens=CONGESTION_SPACE_CARDINAL, embedding_dim=CONGESTION_EMBEDDING_DIM):
        super().__init__()
        self.embeddings = nn.Embedding(num_tokens, embedding_dim)

    def forward(self, input_tokens):
        return self.embeddings(input_tokens)
    
class CustomLearnableEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        
        # Define a learnable embedding matrix
        self.embedding = nn.Parameter(torch.randn(num_embeddings, embedding_dim))  #TODO different init
        
    def forward(self, input):
        # Custom forward function using the embedding matrix
        embedded = self.embedding[input]
        return embedded



class CongestionWrapperEncoder0(nn.Module):

    def __init__(self,
                embeddings: CustomLearnableEmbedding,
                node_number=DEFAULT_TERMINAL_NUMBER,
                in_channels=CONGESTION_EMBEDDING_DIM,
                out_channels=CONGESTION_EMBEDDING_DIM,
                heads=ATTENTION_HEADS,
                dropout=DROPOUT):
        
        super().__init__()

        self.congestion_embeddings = embeddings(node_number, in_channels)

        self.node_number = node_number
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)


    def forward(self, x, adjacency):
        # probably it's inefficient because not vectorized, look best RNN solutions 

        # x = self.congestion_embeddings(self.congestions)

        # flattened_output = torch.stack([self.gat_conv(terminal, edge_dim=self.adjacency).view(-1) for terminal in x])
        # return flattened_output

        ##################
        # yes, vector-parallelism is possible here
        
        batch_dim = x.size(0)
        days_num = x.size(1)

        x = self.congestion_embeddings(x)
        x = x.view(-1, self.node_number, self.out_channels)
        x = list(x)
        data_list = [Data(x=x_, edge_index=adjacency) for x_ in x] 
        batch_loader = Batch.from_data_list(data_list)
        x = self.gat_conv(batch_loader.x, edge_index=batch_loader.edge_index)
        x = x.view(batch_dim, days_num, -1)
        return x 

class CongestionWrapperEncoder1(nn.Module):

    def __init__(self,
                embeddings: CustomLearnableEmbedding,
                node_number=DEFAULT_TERMINAL_NUMBER,
                congestion_space_dimension=CONGESTION_SPACE_CARDINAL,
                in_channels=CONGESTION_EMBEDDING_DIM,
                out_channels=CONGESTION_EMBEDDING_DIM,
                heads=ATTENTION_HEADS,
                dropout=DROPOUT):
        
        super().__init__()

        self.congestion_embeddings = embeddings(congestion_space_dimension, in_channels)

        self.node_number = node_number
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)


    def forward(self, x, adjacency):
        # probably it's inefficient because not vectorized, look best RNN solutions 

        # x = self.congestion_embeddings(self.congestions)

        # flattened_output = torch.stack([self.gat_conv(terminal, edge_dim=self.adjacency).view(-1) for terminal in x])
        # return flattened_output

        ##################
        # yes, vector-parallelism is possible here
        
        batch_dim = x.size(0)
        days_num = x.size(1)

        x = self.congestion_embeddings(x)
        x = x.view(-1, self.node_number, self.out_channels)
        x = list(x)
        data_list = [Data(x=x_, edge_index=adjacency) for x_ in x] 
        batch_loader = Batch.from_data_list(data_list)
        x = self.gat_conv(batch_loader.x, edge_index=batch_loader.edge_index)
        x = x.view(batch_dim, days_num, -1)
        return x 



class CongestionWrapperDecoder(nn.Module):

    def __init__(self):
        super().__init__()

class DecoderLinear(CongestionWrapperDecoder):
    
    def __init__(self, 
                 embeds: CongestionNonLearnableEmbedding,
                 embedding_dim=CONGESTION_EMBEDDING_DIM,
                 terminal_number=DEFAULT_TERMINAL_NUMBER):
        super().__init__(embeds)

        self.terminal_number = terminal_number
        self.lin_layers =[nn.Linear(embedding_dim, terminal_number) for _ in range(terminal_number)]
    
    def forward(self, x, targets): 
        loss = 0

        congestions = torch.tensor(self.terminal_number)

        for layer, number in enumerate(self.lin_layers):
            logits = layer(x)
            if targets is not None:
                loss += functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            congestions[number] = logits.max(1, keepdims=True)

        if loss != 0:
            loss /= self.terminal_number

        return congestions, loss
    

class DecoderGAT(CongestionWrapperDecoder):
    
    def __init__(self, 
                 embeds: CongestionNonLearnableEmbedding,
                 terminal_number=DEFAULT_TERMINAL_NUMBER,
                 embedding_dim=CONGESTION_EMBEDDING_DIM,
                 in_channels=CONGESTION_EMBEDDING_DIM,
                 out_channels=CONGESTION_EMBEDDING_DIM,
                 heads=ATTENTION_HEADS,
                 dropout=DROPOUT,
                 ):
        super().__init__(embeds)

        self.terminal_number = terminal_number
        self.gat_conv_decoder = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

    
    def forward(self, x, targets): 
        loss = 0

        x = x.view(-1, CONGESTION_EMBEDDING_DIM, MAX_CONGESTION)
        logits = layer(x)
        congestions = torch.tensor(self.terminal_number)

        for layer, number in enumerate(logits):
            if targets is not None:
                loss += functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            congestions[number] = logits.max(1, keepdims=True)

        if loss != 0:
            loss /= self.terminal_number

        return congestions, loss
    


class GATDecoder1(CongestionWrapperDecoder):

    def __init__(self,
                node_number=DEFAULT_TERMINAL_NUMBER,
                congestion_space_dimension=CONGESTION_SPACE_CARDINAL,
                in_channels=CONGESTION_EMBEDDING_DIM,
                out_channels=CONGESTION_EMBEDDING_DIM,
                heads=ATTENTION_HEADS,
                dropout=DROPOUT):
        
        super().__init__()


        self.node_number = node_number
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gat_conv = GATConv(in_channels, out_channels, heads=heads, dropout=dropout)

        self.linear_decoder = nn.Linear(out_channels, congestion_space_dimension)

    def forward(self, x, adjacency, targets):
        # probably it's inefficient because not vectorized, look best RNN solutions 

        # x = self.congestion_embeddings(self.congestions)

        # flattened_output = torch.stack([self.gat_conv(terminal, edge_dim=self.adjacency).view(-1) for terminal in x])
        # return flattened_output

        ##################
        # yes, vector-parallelism is possible here
        
        loss = 0
        batch_dim = x.size(0)

        x = x.view(-1, self.node_number, self.out_channels)
        print(x.shape, 'before graph decod')

        x = list(x)
        data_list = [Data(x=x_, edge_index=adjacency) for x_ in x] 
        batch_loader = Batch.from_data_list(data_list)
        x = self.gat_conv(batch_loader.x, edge_index=batch_loader.edge_index)
        x = x.view(-1, self.node_number, self.out_channels)
        print(x.shape, 'after graph decod')

        x = self.linear_decoder(x)
        print(x.shape, 'after linear decod')

        congestions = x.max(2, keepdims=True)

        if targets is not None:
            loss += functional.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1), ignore_index=-1)
            print(functional.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1), ignore_index=-1))
        else: loss = torch.inf

        return congestions, loss/batch_dim

class CongestionTransformerDecoder(nn.Module):
    def __init__(self,
                 terminal_number: int = DEFAULT_TERMINAL_NUMBER,
                 num_transformer_layers: int = ATTENTION_BLOCKS,
                 embedding_dim: int = CONGESTION_EMBEDDING_DIM,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 attn_dropout: float = 0,
                #  mlp_dropout: float = 0.1, ?
                 embedding_dropout: float = 0.1):
        super().__init__()
        self.terminal_number = terminal_number

        self.prediction_congestion_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim*terminal_number),
                                            requires_grad=True)
        print(self.terminal_number, 'lol')
        self.position_embedding = nn.Parameter(data=torch.randn(1, self.terminal_number + 1, embedding_dim*terminal_number),
                                               requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.transformer_encoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=embedding_dim*terminal_number,
                                                                              nhead=num_heads,
                                                                              dim_feedforward=mlp_size,
                                                                              dropout=attn_dropout,
                                                                              activation="gelu",
                                                                              batch_first=True,
                                                                              norm_first=True) for _ in
                                                   range(num_transformer_layers)])

    def forward(self, x):

        batch_size = x.shape[0]
        prediction_congestion_embedding = self.prediction_congestion_embedding.expand(batch_size, -1, -1)
        print(prediction_congestion_embedding.shape, 'shape')
        x = torch.cat((x, prediction_congestion_embedding), dim=1) #TODO cat ord? 
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)

        return x
    

class CongestionModel(nn.Module):
    def __init__(self,
                 congestion_embedding: CongestionNonLearnableEmbedding,
                 congestion_wrapper_encoder: CongestionWrapperEncoder1,
                 congestion_decoder: CongestionTransformerDecoder,
                 congestion_wrapper_decoder: CongestionWrapperDecoder,
                 num_nodes=DEFAULT_TERMINAL_NUMBER,
                 congestion_space_dimension=CONGESTION_SPACE_CARDINAL,
                 embedding_dim=CONGESTION_EMBEDDING_DIM,
                 in_channels_wrapper_encoder=CONGESTION_EMBEDDING_DIM,
                 out_channels_wrapper_encoder=CONGESTION_EMBEDDING_DIM,
                 num_heads_wrapper_encoder=ATTENTION_HEADS, 
                 dropout_wrapper_encoder=DROPOUT,
                 num_transformer_layers: int = ATTENTION_BLOCKS,
                 mlp_size: int = 3072,
                 num_heads: int = 1, #12
                 attn_dropout: float = 0,
                 ):
        super().__init__()

        self.congestion_wrapper_encoder = congestion_wrapper_encoder(congestion_embedding,
                                                                     num_nodes,
                                                                     congestion_space_dimension,
                                                                     in_channels_wrapper_encoder,
                                                                     out_channels_wrapper_encoder,
                                                                     num_heads_wrapper_encoder,
                                                                     dropout_wrapper_encoder)
        
        self.congestion_decoder = congestion_decoder(num_nodes,
                                                     num_transformer_layers,
                                                     embedding_dim,
                                                     mlp_size,
                                                     num_heads,
                                                     attn_dropout)
        
        self.congestion_wrapper_decoder = congestion_wrapper_decoder(num_nodes,
                                                                     congestion_space_dimension,
                                                                     in_channels_wrapper_encoder,
                                                                     out_channels_wrapper_encoder,
                                                                     num_heads_wrapper_encoder,
                                                                     dropout_wrapper_encoder)


    def forward(self, x, adjacency, targets=None):
        loss = 0

        x = self.congestion_wrapper_encoder(x, adjacency)
        x = self.congestion_decoder(x)[:,-1,:]
        x, loss = self.congestion_wrapper_decoder(x, adjacency, targets)

        return x, loss