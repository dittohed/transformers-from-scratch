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


def train_copy_task():
    vocab_size = 11
    batch_size = 64
    n_batches = 32
    n_epochs = 1

    criterion = LabelSmoothing(n_classes=vocab_size, padding_idx=0, 
                               smoothing=0)

    model = build_model(vocab_size, vocab_size, n_layers=2)

    optimizer = torch.optim.Adam(
                    model.parameters(), lr=0.5, 
                    betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=lambda step: rate(
                        step, model.src_preproc[0].d_model, factor=1.0, 
                        warmup_iters=400))

    for epoch in range(n_epochs):
        model.train()  # Switch to training mode
        run_epoch(
            data_gen(vocab_size, batch_size, n_batches),
            model,
            Seq2SeqLoss(model.head, criterion),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n_batches=n_batches,
            curr_epoch=epoch,
            n_epochs=n_epochs,
            mode='train')

        model.eval()
        run_epoch(
            data_gen(vocab_size, batch_size, int(n_batches/4)),
            model,
            Seq2SeqLoss(model.head, criterion),
            n_batches=int(n_batches/4),
            mode='eval')

    print('=== Post-training test ===')
    model.eval()
    src = torch.LongTensor([list(range(10))])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(f'Source: {src}')
    pred = decode_greedy(model, src, src_mask, max_len=max_len, start_symbol=0)
    print(f'Prediction: {pred}')


def data_gen(vocab_size, batch_size, n_batches):
    """
    Generates random data for a source-to-target copy task.
    """

    for i in range(n_batches):
        # Each rows contains an example of length 10
        data = torch.randint(1, vocab_size, size=(batch_size, 10))
        data[:, 0] = 1

        # TODO: try without requires_grad_(False) since detach is used
        src = data.requires_grad_(False).clone().detach()
        target = data.requires_grad_(False).clone().detach()

        yield Batch(src, target, 0)


def train_de_to_en():
    config = {
        'batch_size': 8,
        'distributed': False,
        'n_epochs': 8,
        'base_lr': 1.0,
        'max_padding': 72,
        'warmup': 3000}

    spacy_de, spacy_en = load_tokenizers()
    src_vocab, target_vocab = load_vocab(spacy_de, spacy_en)

    if config['distributed']:
        pass 
    else:
        train_worker(
            None, 1, src_vocab, target_vocab, spacy_de, spacy_en, config, False)


if __name__ == '__main__':
    # inference_test()
    # label_smoothing_test()
    train_copy_task()
    # train_de_to_en()