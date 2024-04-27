import math

import torch
import torch.nn as nn

from utils import clones


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