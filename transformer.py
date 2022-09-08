import copy

import torch
import torch.nn as nn

from layers import (
    LayerNorm, FeedForward, PositionalEncoding, 
    Embedding, ClassificationHead, ClsTokenPrepend)
from utils import clones
from attention import MultiHeadedAttention


class EncoderLayer(nn.Module):
    def __init__(self, size, mha, ff, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = mha 
        self.ff = ff
        self.res_conn = clones(ResConnectionWithLayerNorm(size, dropout), 2)
        self.size = size 

    def forward(self, x, mask):
        x = self.res_conn[0](x, lambda x: self.mha(x, x, x, mask))
        return self.res_conn[1](x, self.ff)


class Encoder(nn.Module):
    def __init__(self, layer, n_layers):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # TODO: Why the authors put norm here?


class DecoderLayer(nn.Module):
    def __init__(self, size, mha1, mha2, ff, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.mha1 = mha1
        self.mha2 = mha2 
        self.ff = ff 
        self.res_conn = clones(ResConnectionWithLayerNorm(size, dropout), 3)

    def forward(self, x, memory, src_mask, target_mask):
        x = self.res_conn[0](x, lambda x: self.mha1(x, x, x, target_mask))
        x = self.res_conn[1](x, lambda x: self.mha2(x, memory, memory, src_mask))
        return self.res_conn[2](x, self.ff)


class Decoder(nn.Module):
    def __init__(self, layer, n_layers):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x  = layer(x, memory, src_mask, target_mask)
        return self.norm(x)  # TODO: Why the authors put norm here?


class ResConnectionWithLayerNorm(nn.Module):
    def __init__(self, size, dropout):
        super(ResConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # TODO: Why the authors put norm inside?
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderDecoderModel(nn.Module):
    def __init__(self, encoder, decoder, src_preproc, target_preproc,
                 head):
        super(EncoderDecoderModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_preproc = src_preproc
        self.target_preproc = target_preproc
        self.head = head 

    def forward(self, src, target, src_mask, target_mask):
        return self.decode(
                self.encode(src, src_mask), src_mask, target, target_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_preproc(src), src_mask)

    def decode(self, memory, src_mask, target, target_mask):
        return self.decoder(self.target_preproc(target), memory, 
                            src_mask, target_mask)


class EncoderModel(nn.Module):
    def __init__(self, encoder, input_preproc, head):
        super(EncoderModel, self).__init__()

        self.encoder = encoder 
        self.input_preproc = input_preproc
        self.head = head

    def forward(self, tokens):
        tokens = torch.squeeze(tokens)  # TODO: find another fix
        return self.encoder(self.input_preproc(tokens), None)


def build_encoder_decoder_model(device, src_vocab_size, target_vocab_size, 
        n_layers=6, d_model=512, d_ff=2048, n_heads=8, dropout=0.1):
    c = copy.deepcopy
    mha = MultiHeadedAttention(n_heads, d_model)
    ff = FeedForward(d_model, d_ff, dropout)
    pos_encoding = PositionalEncoding(d_model, dropout)

    encoder = Encoder(EncoderLayer(d_model, c(mha), c(ff), dropout), 
                      n_layers)
    decoder = Decoder(DecoderLayer(d_model, c(mha), c(mha), c(ff), dropout), 
                      n_layers)
    src_preproc = nn.Sequential(Embedding(src_vocab_size, d_model), 
                                c(pos_encoding))
    target_preproc = nn.Sequential(Embedding(target_vocab_size, d_model), 
                                c(pos_encoding))
    head = ClassificationHead(d_model, target_vocab_size)

    model = EncoderDecoderModel(encoder, decoder,
                        src_preproc, target_preproc, head)

    for p in model.parameters():
        if p.dim() > 1:
            # TODO: Read more about it
            nn.init.xavier_uniform_(p)  

    model.to(device)
    return model


def build_encoder_model(
        device, d_input, n_classes, n_layers=6, d_model=512, d_ff=2048, 
        n_heads=8, dropout=0.1):
    embedding = nn.Linear(d_input, d_model)
    cls_token_prep = ClsTokenPrepend(d_model)
    pos_encoding = PositionalEncoding(d_model, dropout)

    mha = MultiHeadedAttention(n_heads, d_model)
    ff = FeedForward(d_model, d_ff, dropout)
    
    input_preproc = nn.Sequential(embedding, cls_token_prep, pos_encoding)
    encoder = Encoder(EncoderLayer(d_model, mha, ff, dropout), 
                      n_layers)
    head = ClassificationHead(d_model, n_classes, cls_token_only=True)

    model = EncoderModel(encoder, input_preproc, head)
    for p in model.parameters():
        if p.dim() > 1:
            # TODO: Read more about it
            nn.init.xavier_uniform_(p)

    model.to(device)
    return model