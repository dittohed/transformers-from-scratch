import numpy as np
import torch

from torch.optim.lr_scheduler import LambdaLR

from training import KLDivWithLabelSmoothing, Seq2SeqLoss, rate, run_epoch
from transformer import Transformer
from utils import Batch, subsequent_mask


def train_restore_input_task(
    vocab_size: int, min_input_len: int, max_input_len: int, batch_size: int, 
    n_steps: int, n_epochs: int
) -> None:
    """
    Train a small transformer for restoring short integer sequences.
    """

    model = Transformer.from_hyperparams(vocab_size, vocab_size, n_layers=2)

    criterion = KLDivWithLabelSmoothing(n_classes=vocab_size, pad_idx=0, smoothing=0)    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step, model.src_preproc[0].d_model, factor=1.0, 
            warmup_iters=400
        )
    )

    for epoch in range(n_epochs):
        data_gen_train = data_gen(
            vocab_size, min_input_len, max_input_len,
            batch_size, n_steps
        )
        run_epoch(
            data_gen_train,
            model,
            Seq2SeqLoss(model.head, criterion),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n_steps=n_steps,
            curr_epoch=epoch,
            n_epochs=n_epochs
        )

        print('=== Inference demo ===')
        src = torch.LongTensor([
            [1, 4, 3, 7, 8, 2]
        ])
        src_mask = torch.ones(1, 1, src.shape[1])
        print(f'Source: {src.tolist()[0]}')
        pred = decode_greedy(
            model, src, src_mask, max_len=src.shape[1], start_idx=1, pad_idx=0
        )
        print(f'Prediction: {pred.tolist()[0]}\n')


def data_gen(
    vocab_size: int, min_input_len: int, max_input_len: int, 
    batch_size: int, n_steps: int
):
    """
    Generate random data for the input restoration task
    (output = input).
    """

    for _ in range(n_steps):
        data = np.random.randint(2, vocab_size, size=(batch_size, max_input_len))

        # Insert padding randomly at the end
        lens = np.random.randint(min_input_len, max_input_len+1, size=batch_size)
        data = data * (np.arange(max_input_len) < lens[:, None])

        # Insert start symbol
        data[:, 0] = 1

        data = torch.tensor(data, dtype=torch.long)
        yield Batch(data, data, pad=0)


def decode_greedy(
    model: Transformer, src: torch.Tensor, src_mask: torch.Tensor, 
    max_len: int, start_idx: int, pad_idx: int
) -> torch.Tensor:
    """
    Perform inference using greedy decoding (always output token
    with highest score).
    """

    model.eval()
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_idx).type_as(src)

    for _ in range(max_len-1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src)
        )

        prob = model.head(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word[0]

        if next_word.item() == pad_idx:
            break

        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src).fill_(next_word)], 
            dim=1
        )

    return ys


if __name__ == '__main__':
    VOCAB_SIZE = 10  # Inputs will be random ints from [0, VOCAB_SIZE-1] interval
                     # where 0 denotes padding and 1 is a start symbol 
                     # (to make things easier, both src and target has start symbol)
    MIN_INPUT_LEN = 3
    MAX_INPUT_LEN = 10
    BATCH_SIZE = 64
    N_STEPS = 16  # Number of randomly generated batches per epoch
    N_EPOCHS = 8

    train_restore_input_task(
        vocab_size=VOCAB_SIZE,
        min_input_len=MIN_INPUT_LEN,
        max_input_len=MAX_INPUT_LEN,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        n_epochs=N_EPOCHS
    )