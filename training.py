import torch
import torch.nn as nn

from tqdm import tqdm


def run_epoch(
        device, model, dataloader, criterion,
        optimizer=None, lr_scheduler=None, n_batches=None,
        curr_epoch=None, n_epochs=None, mode='train', 
        cls_token_only=False):
    assert mode in ['train', 'eval']

    tqdm_it = tqdm(enumerate(dataloader), total=n_batches, leave=True)

    if mode == 'train':
        tqdm_it.set_description(f'Epoch [{curr_epoch+1}/{n_epochs}]')

    for i, batch in tqdm_it:
        if cls_token_only:
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            out = model.forward(x)
            loss = criterion(out, y, out.shape[0])
        else:
            out = model.forward(batch.src, batch.target, 
                                batch.src_mask, batch.target_mask)
            loss = criterion(out, batch.target_y, batch.n_tokens)

        if mode == 'train':
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

        if mode == 'eval':
            tqdm_it.set_description(f'Validation batch [{i+1}/{n_batches}]')

        tqdm_it.set_postfix(loss=loss.item())
        del loss 


# TODO: implement for multiple GPUs
def train_worker(
        device, model, train_dataloader, val_dataloader,
        criterion, optimizer, lr_scheduler, 
        n_batches_train, n_batches_val, n_epochs,
        cls_token_only=False):

    for epoch in range(n_epochs):
        model.train()
        run_epoch(device, model, train_dataloader, criterion, optimizer,
            lr_scheduler, n_batches_train, epoch, n_epochs, 'train',
            cls_token_only=cls_token_only)
            
        # TODO: checkpoint

        model.eval()
        run_epoch(device, model, val_dataloader, criterion, 
            n_batches=n_batches_val, mode='eval',
            cls_token_only=cls_token_only)


# Very important to use it!
def rate(step, d_model, factor, warmup_iters):
    if step == 0:
        step += 1  # Start with 1 to not raise 0 to negative power

    return (
        factor 
        * d_model ** (-0.5) 
        * min(step ** (-0.5), step * warmup_iters ** (-1.5))
    )


class LabelSmoothing(nn.Module):
    """
    Custom label smoothing from Harvard NLP.
    Assigns `1.0 - smoothing` to GT class, pads entry at padding_idx with 0
    then and distributes `smoothing` to all other entries.
    """

    def __init__(self, n_classes, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing
        self.n_classes = n_classes 
        self.true_dist = None 

    def forward(self, x, target):
        assert x.size(1) == self.n_classes  

        true_dist = x.clone()

        # Just fills all entries with eps / K
        true_dist.fill_(self.smoothing / (self.n_classes - 2))  # TODO: why -2?

        # Writes values from self.confidence to entries corresponding to GT 1s
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)

        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())
        # .detach() means detaching from computation graph 
        # (gradient won't be calculated for this tensor)


class LossWrapper:
    def __init__(self, head, criterion):
        self.head = head 
        self.criterion = criterion

    def __call__(self, x, y, norm):
        y_pred = self.head(x)
        loss = self.criterion(
                y_pred.contiguous().view(-1, y_pred.size(-1)),
                y.contiguous().view(-1))
        loss /= norm

        return loss