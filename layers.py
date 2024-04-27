import math

from typing import Callable

import torch
import torch.nn as nn

from torch.nn.functional import log_softmax

from utils import clones


class LayerNorm(nn.Module):
    def __init__(self, d_input: int, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_input))
        self.beta = nn.Parameter(torch.zeros(d_input))
        self.eps = eps 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.linear1(x).relu())
        x = self.linear2(x)

        return x


class Embedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multiply to make the embeddings bigger
        # when compared to positional encodings (to retain meaning)
        return self.lut(x) * math.sqrt(self.d_model)


class ClassificationHead(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return log_softmax(self.linear(x), dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_encodings = torch.zeros(max_len, d_model)
        positions = torch.arange(0., max_len).unsqueeze(1)
        denoms = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000) / d_model)
        )
        pos_encodings[:, 0::2] = torch.sin(positions*denoms)
        pos_encodings[:, 1::2] = torch.cos(positions*denoms)

        pos_encodings = pos_encodings.unsqueeze(0)

        # Add non-trainable parameters to state_dict
        self.register_buffer('pos_encodings', pos_encodings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_encodings[:, : x.size(1)]
        return self.dropout(x)


class ResConnectionWithLayerNorm(nn.Module):
    def __init__(self, d_input: int, dropout: float):
        super(ResConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(d_input)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable) -> torch.Tensor:
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        
        # Assumption: dimensionality of queries, keys and values 
        # equals d_model // n_heads
        assert d_model % n_heads == 0  
        self.d_kqv = d_model // n_heads
        self.n_heads = n_heads

        # Implement correponding matrices in all heads just as one 
        # big matrix to speed up computation
        self.linears = clones(nn.Linear(d_model, d_model), 4) 
        self.dropout = nn.Dropout(p=dropout)
        
        self.scores_p = None  # Stored for visualization purposes

    def forward(
        self, to_query: torch.Tensor, to_key: torch.Tensor, to_value: torch.Tensor, 
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = to_query.size(0)

        # Project inputs to queries, keys and values
        query, key, value = [
            lin(x).view(batch_size, -1, self.n_heads, self.d_kqv).transpose(1, 2)
            for lin, x in zip(self.linears, (to_query, to_key, to_value))
        ]

        x, self.scores_p = self.attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # Concat multiple heads output
        # Resize (batch_size, n_heads, max_len, d_kqv) to (batch_size, max_len, d_model)
        x = x.transpose(1, 2).reshape(batch_size, -1, self.n_heads*self.d_kqv)

        return self.linears[-1](x)
    
    @staticmethod
    def attention(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
        mask: torch.Tensor = None, dropout: nn.Dropout = None
    ) -> tuple:
        # query/key/value have shape (batch_size, n_heads, max_len, d_kqv)
        d_kqv = query.size(-1)  

        # matmul automatically handles outer dimensions
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_kqv)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        scores_p = scores.softmax(dim=-1)
        if dropout is not None:
            scores_p = dropout(scores_p)
        
        return torch.matmul(scores_p, value), scores_p