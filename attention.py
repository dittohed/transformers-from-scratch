import math

import torch
import torch.nn as nn

from utils import clones


def attention(query, key, value, mask=None, dropout=None):
    # query/key/value have shape (n_batches, h, time steps, d_k)
    d_k = query.size(-1)  

    # matmul automatically handles outer dimensions
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores_p = scores.softmax(dim=-1)
    if dropout is not None:
        scores_p = dropout(scores_p)
    
    return torch.matmul(scores_p, value), scores_p


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        
        # The code uses a convention that dimensionality of queries,
        # keys and values equals d_model // n_heads
        assert d_model % n_heads == 0  
        self.d_k = d_model // n_heads  # = d_q = d_v
        self.n_heads = n_heads

        # Implement multiple heads' matrices just as one bigger matrix
        # to speed up computation
        self.linears = clones(nn.Linear(d_model, d_model), 4) 
        self.dropout = nn.Dropout(p=dropout)
        
        self.scores_p = None  # Stored for visualization purposes

    def forward(self, to_query, to_key, to_value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)  # Add dummy dimension for multiple heads
        batch_size = to_query.size(0)

        # Project inputs to queries, keys and values
        # linears[0] corresponds to W_q for all heads
        # linears[1] corresponds to W_k for all heads
        # linears[2] corresponds to W_v for all heads
        query, key, value = [
            lin(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (to_query, to_key, to_value))
        ]

        # Self-attention
        x, self.scores_p = attention(query, key, value, 
                                     mask=mask, dropout=self.dropout)

        # Concat multiple heads
        # Resize (batch_size, n_heads, t, d_k) to (batch_size, t, d_model)
        # Use contiguous to physically reorganize data (not changing just
        # metadata)
        x = (
                x.transpose(1, 2)
                .contiguous()
                .view(batch_size, -1, self.n_heads*self.d_k))

        # TODO: Why delete these variables?
        del query
        del key 
        del value 

        return self.linears[-1](x)