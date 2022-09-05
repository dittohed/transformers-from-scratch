import os 

from os.path import exists

import spacy

import torch
import torchtext.datasets as datasets

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from utils import collate_batch


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
        batch_size=12000, max_padding=128, distributed=False):

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
