import os 

from os.path import exists

import torch
import torchtext.datasets as datasets
import torch.distributed as dist

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import spacy

from utils import Batch, collate_batch
from transformer import build_model
from training import LabelSmoothing, Seq2SeqLoss, rate, run_epoch


def load_tokenizers():
    try:
        spacy_de = spacy.load('de_core_news_sm')
    except IOError:
        os.system('python -m spacy download de_core_news_sm')
        spacy_de = spacy.load('de_core_news_sm')

    try:
        spacy_en = spacy.load('en_core_web_sm')
    except IOError:
        os.system('python -m spacy download en_core_web_sm')
        spacy_en = spacy.load('en_core_web_sm')

    return spacy_de, spacy_en


def tokenize(text, tokenizer):
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer, index):
    for x in data_iter:
        yield tokenizer(x[index])


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    # TODO: rzeczywiście potrzeba dwa razy brać datasets?
    print('Building german vocabulary...')
    train, val, test = datasets.Multi30k(language_pair=('de', 'en'))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train+val+test, tokenize_de, index=1),
        min_freq=2,
        specials=['<s>', '</s>', '<blank>', '<unk>']
    )

    print('Building english vocabulary...')
    train, val, test = datasets.Multi30k(language_pair=('de', 'en'))
    vocab_target = build_vocab_from_iterator(
        yield_tokens(train+val+test, tokenize_en, index=1),
        min_freq=2,
        specials=['<s>', '</s>', '<blank>', '<unk>']
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_target.set_default_index(vocab_target["<unk>"])

    return vocab_src, vocab_target


def load_vocab(spacy_de, spacy_en):
    if not exists('vocab.pt'):
        vocab_src, vocab_target = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_target), 'vocab.pt')
    else:
        vocab_src, vocab_target = torch.load('vocab.pt')

    return vocab_src, vocab_target


def create_dataloaders(
        device, src_vocab, target_vocab, spacy_de, spacy_en, 
        batch_size=12000, max_padding=128, distributed=True):

    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            src_vocab,
            target_vocab,
            device,
            max_padding=max_padding,
            pad_id=src_vocab.get_stoi()['<blank>']
        )

    train_iter, val_iter, test_iter = datasets.Multi30k(
                                            language_pair=('de', 'en'))
    train_iter_map = to_map_style_dataset(train_iter)
    val_iter_map = to_map_style_dataset(val_iter)

    train_sampler = DistributedSampler(train_iter_map) if distributed else None
    val_sampler = DistributedSampler(val_iter_map) if distributed else None

    train_dataloader = DataLoader(
        train_iter_map,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn)
    val_dataloader = DataLoader(
        val_iter_map,
        batch_size=batch_size,
        shuffle=(val_sampler is None),
        sampler=val_sampler,
        collate_fn=collate_fn)

    return (train_dataloader, len(train_iter_map), 
           val_dataloader, len(val_iter_map))


def train_worker(gpu, n_gpus_per_node, src_vocab, target_vocab, 
        spacy_de, spacy_en, config, distributed=False):

    # print(f'Train worker process using GPU: {gpu} for training', flush=True)
    # torch.cuda.set_device(gpu)

    pad_idx = target_vocab['<blank>']
    d_model = 512
    model = build_model(len(src_vocab), len(target_vocab), 
                        n_layers=2)  # TODO: to 6
    # model.cuda(gpu)
    module = model 
    is_main_process = True
    batch_size = config['batch_size'] // n_gpus_per_node

    if distributed:
        dist.init_process_group(
            'nccl', init_method='env://', rank=gpu, world_size=n_gpus_per_node)

        model = DDP(model, device_ids=[gpu])
        module = model.module 
        is_main_process = (gpu == 0)

    criterion = LabelSmoothing(n_classes=len(target_vocab), 
                               padding_idx=pad_idx,
                               smoothing=0.1)
    # criterion.cuda(gpu)

    (train_dataloader, n_batches_train,
    val_dataloader, n_batches_val) = create_dataloaders(
                                        gpu,
                                        src_vocab,
                                        target_vocab,
                                        spacy_de,
                                        spacy_en,
                                        batch_size,
                                        config['max_padding'],
                                        distributed)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'],
                    betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(optimizer=optimizer, 
                            lr_lambda=lambda step: rate(
                                step, d_model, factor=1, 
                                warmup_iters=config['warmup']))
    
    for epoch in range(config['n_epochs']):
        if distributed:
            train_dataloader.sampler.set_epoch(epoch)
            val_dataloader.sampler.set_epoch(epoch)

        model.train()  # Switch to training mode
        run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            Seq2SeqLoss(module.head, criterion),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            n_batches=n_batches_train,
            curr_epoch=epoch,
            n_epochs=config['n_epochs'],
            mode='train')

        if is_main_process:
            pass  # TODO: checkpoint

        model.eval()
        run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in val_dataloader),
            model,
            Seq2SeqLoss(module.head, criterion),
            n_batches=n_batches_val,
            mode='eval')
