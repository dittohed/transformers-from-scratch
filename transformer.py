import copy

import torch
import torch.nn as nn

from layers import (
    LayerNorm, FeedForward, PositionalEncoding, Embedding, 
    ClassificationHead, ResConnectionWithLayerNorm, MultiHeadedAttention
)
from utils import clones


class EncoderLayer(nn.Module):
    """
    Implementation of a classical transformer encoder layer, consisting of:
    * a residual connection with multi-head self-attention;
    * a residual connection with feed-forward layers.
    """

    def __init__(
        self, d_model: int, mha: MultiHeadedAttention, ff: FeedForward, 
        dropout: float
    ):
        super(EncoderLayer, self).__init__()
        self.mha = mha 
        self.ff = ff
        self.res_conn = clones(ResConnectionWithLayerNorm(d_model, dropout), 2)
        self.size = d_model 

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.res_conn[0](x, lambda x: self.mha(x, x, x, mask))
        return self.res_conn[1](x, self.ff)


class Encoder(nn.Module):
    """
    Implementation of a classical transformer encoder, consisting of 
    multiple identical consecutive encoder layers.
    """

    def __init__(self, layer: EncoderLayer, n_layers: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    Implementation of a classical transformer decoder layer, consisting of:
    * a residual connection with masked multi-head self-attention;
    * a residual connection with multi-head cross-attention;
    * a residual connection with feed-forward layers.
    """

    def __init__(
        self, d_model: int, mha1: MultiHeadedAttention, mha2: MultiHeadedAttention, 
        ff: FeedForward, dropout: float
    ):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        self.mha1 = mha1
        self.mha2 = mha2 
        self.ff = ff 
        self.res_conn = clones(ResConnectionWithLayerNorm(d_model, dropout), 3)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, 
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        x = self.res_conn[0](x, lambda x: self.mha1(x, x, x, target_mask))
        x = self.res_conn[1](x, lambda x: self.mha2(x, memory, memory, src_mask))
        return self.res_conn[2](x, self.ff)


class Decoder(nn.Module):
    """
    Implementation of a classical transformer decoder, consisting of 
    multiple identical consecutive encoder layers.
    """

    def __init__(self, layer: DecoderLayer, n_layers: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n_layers)
        self.norm = LayerNorm(layer.size)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor, 
        target_mask: torch.Tensor
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, target_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Implementation of a classical transformer, consisting of:
    * an source preprocessing module;
    * an encoder;
    * a target preprocessing module;
    * a decoder;
    * a classification head (not part of `forward` to reduce memory during 
    inference by feeding only the most recent token into the head).
    """

    def __init__(
        self, encoder: Encoder, decoder: Decoder, 
        src_preproc: nn.Sequential, target_preproc: nn.Sequential,
        head: ClassificationHead
    ):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_preproc = src_preproc
        self.target_preproc = target_preproc
        self.head = head 

    def forward(
        self, src: torch.Tensor, target: torch.Tensor, 
        src_mask: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, target, target_mask)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.src_preproc(src), src_mask)

    def decode(
        self, memory: torch.Tensor, src_mask: torch.Tensor, 
        target: torch.Tensor, target_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.decoder(
            self.target_preproc(target), memory, src_mask, target_mask
        )

    @classmethod
    def from_hyperparams(
        cls, src_vocab_size: int, target_vocab_size: int, n_layers: int = 6, 
        d_model: int = 512, d_ff: int = 2048, n_heads: int = 8, 
        dropout: float = 0.1
    ) -> 'Transformer':
        c = copy.deepcopy
        mha = MultiHeadedAttention(n_heads, d_model)
        ff = FeedForward(d_model, d_ff, dropout)
        pos_encoding = PositionalEncoding(d_model, dropout)

        encoder = Encoder(
            EncoderLayer(d_model, c(mha), c(ff), dropout), 
            n_layers
        )
        decoder = Decoder(
            DecoderLayer(d_model, c(mha), c(mha), c(ff), dropout), 
            n_layers
        )
        src_preproc = nn.Sequential(
            Embedding(d_model, src_vocab_size), 
            c(pos_encoding)
        )
        target_preproc = nn.Sequential(
            Embedding(d_model, target_vocab_size), 
            c(pos_encoding)
        )
        head = ClassificationHead(d_model, target_vocab_size)

        model = cls(encoder, decoder, src_preproc, target_preproc, head)

        # TODO: is this really needed?
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)  

        return model