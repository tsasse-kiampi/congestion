import torch
from torch import nn
from torch.nn import functional
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from parameters import DIM, ATTENTION_BLOCKS, EMBEDDING_DIM, MAX_CONGESTION, TERMINAL_DEFAULT_NUMBER


class CongestionLearnableEmbedding(nn.Module):
    r"""Tokenization of congestion
        
        Each indice of embeddings represents congestion in hours
    """
    def __init__(self, num_tokens=MAX_CONGESTION, embedding_dim=EMBEDDING_DIM):
        super().__init__()
        self.embeddings = nn.Embedding(num_tokens, embedding_dim)

    def forward(self, input_tokens):
        return self.embeddings(input_tokens)

class CongestionWrapperEncoder(nn.Module):

    def __init__(self, embeds, congestion_data):
        super().__init__()
        self.embeds = embeds()
    pass #TODO

class CongestionWrapperDecoder(nn.Module):

    def __init__(self,
                 embeds: CongestionLearnableEmbedding,):
        super().__init__()
        self.embeds = embeds()

class DecoderLinear(CongestionWrapperDecoder):
    
    def __init__(self, 
                 embeds: CongestionLearnableEmbedding,
                 embedding_dim=EMBEDDING_DIM,
                 terminal_number=TERMINAL_DEFAULT_NUMBER):
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

class CongestionTransformerDecoder(nn.Module):
    def __init__(self,
                 terminal_number: int = TERMINAL_DEFAULT_NUMBER,
                 num_transformer_layers: int = ATTENTION_BLOCKS,
                 embedding_dim: int = EMBEDDING_DIM,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 attn_dropout: float = 0,
                #  mlp_dropout: float = 0.1, ?
                 embedding_dropout: float = 0.1):
        super().__init__()
        self.terminal_number = terminal_number

        self.prediction_congestion_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        self.position_embedding = nn.Parameter(data=torch.randn(1, self.terminal_number + 1, embedding_dim),
                                               requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.transformer_encoder = nn.Sequential(*[nn.TransformerDecoderLayer(d_model=embedding_dim,
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
        x = torch.cat((prediction_congestion_embedding, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)

        return x
    


class CongestionModel(nn.Module):
    def __init__(self,
                 congestion_embedding: CongestionLearnableEmbedding,
                 congestion_wrapper_encoder: CongestionWrapperEncoder,
                 congestion_decoder: CongestionTransformerDecoder,
                 congestion_wrapper_decoder: CongestionWrapperDecoder,
                 ):
        super().__init__()

        self.congestion_wrapper_encoder = congestion_wrapper_encoder(congestion_embedding)
        self.congestion_wrapper_decoder = congestion_wrapper_decoder()
        self.congestion_decoder = congestion_decoder()

    def forward(self, x, targets=None):

        x = self.congestion_wrapper_encoder(x)
        x = self.congestion_decoder(x)[-1]
        x, loss = self.congestion_wrapper_decoder(x, targets)

        return x, loss