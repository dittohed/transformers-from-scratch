import copy

import torch
import torch.nn as nn


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce `n` identical layers.
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)]) 


class Batch:
    """
    Custom wrapper for inputs and outputs.

    Attributes:
        src (torch.Tensor): 
            Tensor of shape [B, MAX_LEN_IN] with integer representations 
            of input tokens.
        src_mask (torch.Tensor):
            Tensor of shape [B, 1, MAX_LEN_IN] with booleans, where
            False indicates input padding (has to be used to calculate self-attention
            only with respect to actual tokens).
        target (torch.Tensor):
            Tensor of shape [B, MAX_LEN_OUT-1] with integer representations 
            of GT output tokens. It's used during training with "teacher forcing"
            method (multiple GTs based on single sequence) as decoder input so 
            the last token is removed.
        target_y (torch.Tensor):
            Tensor of shape [B, MAX_LEN_OUT-1] with integer representations 
            of GT output tokens. It's used during training with "teacher forcing"
            method (multiple GTs based on single sequence) as decoder output so 
            the first token is removed.
        target_mask (torch.Tensor):
            Tensor of shape [B, MAX_LEN_OUT-1, MAX_LEN_OUT-1] with booleans, where
            False indicates GT output padding (has to be used to calculate attention
            only with respect to actual tokens) or masked tokens (to perform masked 
            attention).
        n_tokens (int):
            Total number of tokens that will be predicted.
    """

    def __init__(self, src: torch.Tensor, target: torch.Tensor = None, pad: int = 0):
        self.src = src 
        self.src_mask = (src != pad).unsqueeze(-2)

        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            self.target_mask = self.make_mask(self.target, pad)
            self.n_tokens = (self.target_y != pad).sum().item()

    @staticmethod
    def make_mask(target: torch.Tensor, pad: int) -> torch.Tensor:
        target_mask = (target != pad).unsqueeze(-2)
        target_mask = (
            target_mask 
            & subsequent_mask(target.size(-1)).type_as(target_mask.data)
        )

        return target_mask


def subsequent_mask(size: int) -> torch.Tensor:
    """
    Return a boolean square mask, where in `i`-th row, `i` first elements
    from the left are True.
    """

    scores_shape = (1, size, size)
    mask = torch.triu(torch.ones(scores_shape), diagonal=1).type(torch.uint8)

    return mask == 0