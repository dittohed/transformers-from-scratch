import copy

import torch
import torch.nn as nn

from torch.nn.functional import pad


def clones(module, n):
    """
    Produce n identical layers.
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)]) 


class Batch:
    def __init__(self, src, target=None, pad=2):
        self.src = src 
        self.src_mask = (src != pad).unsqueeze(-2)

        if target is not None:
            self.target = target[:, :-1]
            self.target_y = target[:, 1:]
            self.target_mask = self.make_mask(self.target, pad)
            self.n_tokens = (self.target_y != pad).data.sum()  # TODO: data?

    @staticmethod
    def make_mask(target, pad):
        target_mask = (target != pad).unsqueeze(-2)
        target_mask = (
            target_mask & subsequent_mask(target.size(-1))
                            .type_as(target_mask.data))

        return target_mask


def collate_batch(batch, src_pipeline, target_pipeline, 
        src_vocab, target_vocab, device, 
        max_padding=128, pad_id=2):

    start_id = torch.tensor([0], device=device)
    end_id = torch.tensor([1], device=device)

    src_list = []
    target_list = []

    for (src, target) in batch:
        src_processed = torch.cat(
                            [start_id,
                             torch.tensor(src_vocab(src_pipeline(src)),
                                          dtype=torch.int64,
                                          device=device),
                             end_id], 
                             0)

        target_processed = torch.cat(
                            [start_id,
                             torch.tensor(target_vocab(target_pipeline(target)),
                                          dtype=torch.int64,
                                          device=device),
                             end_id], 
                             0)

        src_list.append(
            pad(src_processed, (0, max_padding - len(src_processed)), 
                value=pad_id)
        )

        target_list.append(
            pad(target_processed, (0, max_padding - len(target_processed)), 
                value=pad_id)
        )

    src = torch.stack(src_list)
    target = torch.stack(target_list)

    return (src, target)


def decode_greedy(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src)

    for i in range(max_len - 1):
        out = model.decode(memory, src_mask, ys, 
                           subsequent_mask(ys.size(1)).type_as(src))
        prob = model.head(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src).fill_(next_word)], 
            dim=1)

    return ys 


def subsequent_mask(size):
    scores_shape = (1, size, size)
    mask = torch.triu(torch.ones(scores_shape), diagonal=1).type(torch.uint8)

    return mask == 0