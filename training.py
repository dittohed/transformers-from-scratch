from collections.abc import Iterator

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from layers import ClassificationHead
from utils import Batch
from transformer import Transformer


def run_epoch(
    data_gen: Iterator[Batch], model: Transformer, criterion: nn.Module,
    n_steps: int, optimizer: Optimizer, lr_scheduler: LRScheduler, 
    curr_epoch: int, n_epochs: int
) -> None:
    """
    Train `model` until `data_gen` is exhausted (for `n_steps`). 
    """

    model.train()

    tqdm_it = tqdm(data_gen, total=n_steps, leave=True)
    tqdm_it.set_description(f'Epoch [{curr_epoch+1}/{n_epochs}]')

    for batch in tqdm_it:
        out = model(batch.src, batch.target, batch.src_mask, batch.target_mask)
        loss = criterion(out, batch.target_y, batch.n_tokens)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()

        tqdm_it.set_postfix(loss=loss.item())


def rate(step: int, d_model: int, factor: float, warmup_iters: int) -> float:
    """
    Return learning rate at `step`.

    Corresponds to increasing the learning rate linearly for `warmup_iters`, 
    then decreasing it proportionally to the inverse square root of the step 
    number.
    """

    if step == 0:
        step += 1  # Start with 1 to not raise 0 to negative power

    return (
        factor 
        * d_model ** (-0.5) 
        * min(step ** (-0.5), step * warmup_iters ** (-1.5))
    )


class KLDivWithLabelSmoothing(nn.Module):
    """
    KL Divergence with custom label smoothing (assigns 1.0 - `smoothing` to GT class, 
    fills padding with 0 then and distributes `smoothing` to all other entries).
    """

    def __init__(self, n_classes: int, pad_idx: int, smoothing:float = 0.0):
        super(KLDivWithLabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(reduction='sum')
        self.pad_idx = pad_idx
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing
        self.n_classes = n_classes 
        self.true_dist = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Perform label smoothing on `target`, then return unormalized 
        (no average) KL divergence between `x` and `target`.
        """

        assert x.size(1) == self.n_classes  

        true_dist = x.clone()

        # Just fills all entries with eps / K
        true_dist.fill_(self.smoothing / (self.n_classes - 2))

        # Writes values from self.confidence to entries corresponding to GT 1s
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.pad_idx] = 0
        mask = torch.nonzero(target.data == self.pad_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
        # .detach() means detaching from computation graph 
        # (gradient won't be calculated for this tensor)


class Seq2SeqLoss:
    """
    Wrapper for calculating loss.
    """

    def __init__(self, head: ClassificationHead, criterion: nn.Module):
        self.head = head 
        self.criterion = criterion

    def __call__(
        self, x: torch.Tensor, y: torch.Tensor, n_pred_tokens: int
    ) -> torch.Tensor:
        y_pred = self.head(x)
        loss = self.criterion(
            y_pred.contiguous().view(-1, y_pred.size(-1)),
            y.contiguous().view(-1)
        )
        loss /= n_pred_tokens

        return loss