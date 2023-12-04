import torch
from torch import nn
from parameters import DIM, ATTENTION_BLOCKS, EMBEDDING_DIM


class CongestionLearnableEmbedding:
    """Tokenization of congestion"""
    pass


class CongestionTransformer(nn.Module):
    def __init__(self,
                 congestion_wrapper_encoder,
                 congestion_wrapper_decoder,
                 congestion_size: int = DIM,
                 num_transformer_layers: int = ATTENTION_BLOCKS,
                 embedding_dim: int = EMBEDDING_DIM,
                 mlp_size: int = 3072,
                 num_heads: int = 12,
                 attn_dropout: float = 0,
                 mlp_dropout: float = 0.1,
                 embedding_dropout: float = 0.1,
                 num_classes: int = 3):
        super().__init__()


        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embedding_dim),
                                            requires_grad=True)

        self.position_embedding = nn.Parameter(data=torch.randn(1, self.num_patches + 1, embedding_dim),
                                               requires_grad=True)

        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.patch_embedding = PatchEmbedding()

        self.transformer_encoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=embedding_dim,
                                                                              nhead=num_heads,
                                                                              dim_feedforward=mlp_size,
                                                                              dropout=attn_dropout,
                                                                              activation="gelu",
                                                                              batch_first=True,
                                                                              norm_first=True) for _ in
                                                   range(num_transformer_layers)])
        
        self.congestion_wrapper_encoder = congestion_wrapper_encoder()
        self.congestion_wrapper_decoder = congestion_wrapper_decoder()

    def forward(self, x):

        x = self.congestion_wrapper_encoder(x)
        batch_size = x.shape[0]
        class_token = self.class_embedding.expand(batch_size, -1, -1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token, x), dim=1)
        x = self.position_embedding + x
        x = self.embedding_dropout(x)
        x = self.transformer_encoder(x)
        x = self.congestion_wrapper_decoder(x)

        return x
    


class CongestionModel(nn.Module):
    def __init__(self,
                 congestion_wrapper_encoder,
                 congestion_decoder,
                 congestion_wrapper_decoder,
                 ):
        super().__init__()

        self.congestion_wrapper_encoder = congestion_wrapper_encoder()
        self.congestion_wrapper_decoder = congestion_wrapper_decoder()
        self.congestion_decoder = congestion_decoder()

    def forward(self, x):

        x = self.congestion_wrapper_encoder(x)
        x = self.congestion_decoder(x)[-1]
        x = self.congestion_wrapper_decoder(x)

        return x