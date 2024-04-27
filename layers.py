import math

from typing import Callable

import torch
import torch.nn as nn

from torch.nn.functional import log_softmax


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
