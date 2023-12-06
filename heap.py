import torch.nn as nn
import torch

class LearnableTokens(nn.Module):
    def __init__(self, num_tokens, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(num_tokens, embedding_dim)


    def forward(self, input_tokens):
        return self.embeddings(input_tokens)

num_tokens = 30
embedding_dim = 3

learnable_tokens_model = LearnableTokens(num_tokens, embedding_dim)

example_tokens = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
embeddings = learnable_tokens_model(example_tokens)