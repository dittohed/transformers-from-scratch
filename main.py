import numpy as np
import torch

from torch.optim.lr_scheduler import LambdaLR

from training import LabelSmoothing, Seq2SeqLoss, rate, run_epoch
from transformer import build_model
from utils import Batch, subsequent_mask, decode_greedy
from lang import load_tokenizers, load_vocab, train_worker


def label_smoothing_test():
    criterion = LabelSmoothing(5, 0, 0.1)
    preds = torch.Tensor([
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0]])
    
    criterion(x=preds, target=torch.LongTensor([1, 2]))
    print(criterion.true_dist)


def inference_test():
    test_model = build_model(11, 11, n_layers=2, n_heads=4)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)

    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.head(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)


def train_restore_input_task(
    vocab_size: int, min_input_len: int, max_input_len: int,
    batch_size: int, n_steps: int, n_epochs: int
):
    """
    Train a demo small transformer for restoring short input
    small integer sequences.
    """

    model = build_model(vocab_size, vocab_size, n_layers=2)

    criterion = LabelSmoothing(n_classes=vocab_size, padding_idx=0, smoothing=0)    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.5, betas=(0.9, 0.98), eps=1e-9)

    lr_scheduler = LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=lambda step: rate(
                        step, model.src_preproc[0].d_model, factor=1.0, 
                        warmup_iters=400))

    data_gen_val = data_gen(
        vocab_size, min_input_len, max_input_len,
        batch_size, n_steps
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
            n_batches=n_steps,
            curr_epoch=epoch,
            n_epochs=n_epochs,
            mode='train'
        )
        run_epoch(
            data_gen_val,
            model,
            Seq2SeqLoss(model.head, criterion),
            n_batches=n_steps,
            mode='eval'
        )

    print('=== Post-training test ===')
    model.eval()
    src = torch.LongTensor([list(range(10))])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(f'Source: {src}')
    pred = decode_greedy(model, src, src_mask, max_len=max_len, start_symbol=0)
    print(f'Prediction: {pred}')


def data_gen(
    vocab_size: int, min_input_len: int, max_input_len: int,
    batch_size: int, n_steps: int
):
    """
    Generate random data for the input restoration task
    (output = input).
    """

    for _ in range(n_steps):
        data = np.random.randint(1, vocab_size, size=(batch_size, max_input_len))

        # Insert padding randomly at the end
        lens = np.random.randint(min_input_len, max_input_len+1, size=batch_size)
        data = data * (np.arange(max_input_len) < lens[:, None])

        data = torch.tensor(data, dtype=torch.long)
        yield Batch(data, data, pad=0)


if __name__ == '__main__':
    VOCAB_SIZE = 10  # Inputs will be random ints from [1, VOCAB_SIZE-1] interval
                     # (0 for padding)
    MIN_INPUT_LEN = 5
    MAX_INPUT_LEN = 10
    BATCH_SIZE = 64
    N_STEPS = 32  # Number of randomly generated batches per epoch
    N_EPOCHS = 5

    train_restore_input_task(
        vocab_size=VOCAB_SIZE,
        min_input_len=MIN_INPUT_LEN,
        max_input_len=MAX_INPUT_LEN,
        batch_size=BATCH_SIZE,
        n_steps=N_STEPS,
        n_epochs=N_EPOCHS
    )