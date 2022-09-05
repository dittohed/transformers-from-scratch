import torch

from torch.optim.lr_scheduler import LambdaLR

from training import LabelSmoothing, Seq2SeqLoss, rate, run_epoch
from transformer import build_encoder_decoder_model
from utils import Batch, decode_greedy
from lang import load_tokenizers, load_vocab, train_worker, create_dataloaders


def train_copy_task(device):
    # Hyperparameters
    vocab_size = 11
    batch_size = 64
    n_batches = 32
    n_epochs = 1

    # Data
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

    train_dataloader = data_gen(vocab_size, batch_size, n_batches)
    n_batches_train = n_batches
    val_dataloader = data_gen(vocab_size, batch_size, int(n_batches/4))
    n_batches_val = int(n_batches/4)

    # Model
    model = build_encoder_decoder_model(
                vocab_size, vocab_size, n_layers=2).to(device)

    # Training
    criterion = LabelSmoothing(n_classes=vocab_size, padding_idx=0, 
                               smoothing=0)
    optimizer = torch.optim.Adam(
                    model.parameters(), lr=0.5, 
                    betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=lambda step: rate(
                        step, model.src_preproc[0].d_model, factor=1.0, 
                        warmup_iters=400))

    train_worker(device, model, train_dataloader, val_dataloader,
        criterion, optimizer, lr_scheduler, 
        n_batches_train, n_batches_val, n_epochs)

    # Eval
    print('=== Post-training test ===')
    model.eval()
    src = torch.LongTensor([list(range(10))])
    max_len = src.shape[1]
    src_mask = torch.ones(1, 1, max_len)
    print(f'Source: {src}')
    pred = decode_greedy(model, src, src_mask, max_len=max_len, start_symbol=0)
    print(f'Prediction: {pred}')


def train_de_to_en(device):
    # Hyperparameters
    batch_size = 8
    n_epochs = 8
    base_lr = 1.0 
    max_padding = 72
    warmup = 3000
    d_model = 512
    n_layers = 6

    # Data
    spacy_de, spacy_en = load_tokenizers()
    src_vocab, target_vocab = load_vocab(spacy_de, spacy_en)
    pad_idx = target_vocab['<blank>']

    (train_dataloader, n_batches_train,
    val_dataloader, n_batches_val) = create_dataloaders(
                                        device,
                                        src_vocab,
                                        target_vocab,
                                        spacy_de,
                                        spacy_en,
                                        batch_size,
                                        max_padding
                                        )

    train_dataloader = (Batch(b[0], b[1], pad_idx) for b in train_dataloader)
    val_dataloader = (Batch(b[0], b[1], pad_idx) for b in val_dataloader)
    
    # Model
    model = build_encoder_decoder_model(len(src_vocab), len(target_vocab), 
                                        n_layers).to(device)

    # Training
    criterion = LabelSmoothing(n_classes=len(target_vocab), 
                               padding_idx=pad_idx,
                               smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr,
                    betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(optimizer=optimizer, 
                            lr_lambda=lambda step: rate(
                                step, d_model, factor=1, 
                                warmup_iters=warmup))

    train_worker(device, model, train_dataloader, val_dataloader, 
        criterion, optimizer, lr_scheduler, 
        n_batches_train, n_batches_val, n_epochs)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train_copy_task(device)
    train_de_to_en(device)