import math

import torch
import torch.nn as nn

from torch.nn.functional import softmax, log_softmax


class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.eps = eps 

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(
                self.dropout(
                    self.linear1(x).relu()))


class Embedding(nn.Module):
    def __init__(self, d_input, d_model):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(d_input, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Multiply by sqrt(self.d_model) to make the embeddings bigger
        # when compared to positional encodings
        return self.lut(x) * math.sqrt(self.d_model)


class ClassificationHead(nn.Module):
    def __init__(self, d_model, vocab_size, cls_token_only=False):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)
        self.cls_token_only = cls_token_only

    def forward(self, x):
        if self.cls_token_only:
            preds = softmax(self.linear(x[:, 0, :]), dim=-1)
        else:
            preds = log_softmax(self.linear(x), dim=-1)

        return preds


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_encodings = torch.zeros(max_len, d_model)
        positions = torch.arange(0., max_len).unsqueeze(1)
        denoms = torch.exp(
                    torch.arange(0., d_model, 2) * -(math.log(10000) / d_model))
        pos_encodings[:, 0::2] = torch.sin(positions*denoms)
        pos_encodings[:, 1::2] = torch.cos(positions*denoms)

        # Add a dummy batch dimension
        pos_encodings = pos_encodings.unsqueeze(0)

        # Use register_buffer() to add nontrainable parameters to state_dict
        # - won't be returned by model.parameters()
        self.register_buffer('pos_encodings', pos_encodings)

    def forward(self, x):
        x = x + self.pos_encodings[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class ClsTokenPrepend(nn.Module):
    def __init__(self, d_model):
        super(ClsTokenPrepend, self).__init__()
        self.d_model = d_model
        self.cls_token = nn.Parameter(torch.randn((1, self.d_model)))

    def forward(self, tokens_batch):
        return torch.stack(
            [torch.vstack((self.cls_token, tokens)) for tokens in tokens_batch])


if __name__ == '__main__':
    # tokens = torch.randn((32, 16, 512))
    # cls_token_prep = ClsTokenPrepend(512)
    # proc = cls_token_prep(tokens)
    # print(proc.shape)
    # print(proc[0][0])
    # print(proc[0][1])

    head = ClassificationHead(512, 1000, cls_token_only=True)
    x = torch.randn((8, 64, 512))
    out = head(x)
    print(out.shape)