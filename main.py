import copy
import math

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax
from torch.optim.lr_scheduler import LambdaLR

from tqdm import tqdm


def clones(module, n):
    """
    Produces n identical layers.
    """

    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)]) 


class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.eps = eps 

    def forward(self, x):
        # TODO: dlaczego -1?
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class Encoder(nn.Module):
    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)  # TODO: po co tutaj norm na końcu?


class EncoderLayer(nn.Module):
    def __init__(self, size, mha, ff, dropout):
        super(EncoderLayer, self).__init__()
        self.mha = mha 
        self.ff = ff
        self.res_conn = clones(ResConnectionWithLayerNorm(size, dropout), 2)
        self.size = size 

    def forward(self, x, mask):
        x = self.res_conn[0](x, lambda x: self.mha(x, x, x, mask))  # TODO: po co tu maska?
        return self.res_conn[1](x, self.ff)


class ResConnectionWithLayerNorm(nn.Module):
    def __init__(self, size, dropout):
        super(ResConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # TODO: dlaczego autorzy dali norm w środku?
        return x + self.dropout(sublayer(self.norm(x)))


class Decoder(nn.Module):
    def __init__(self, layer, n):
        super(Decoder, self).__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, target_mask):
        for layer in self.layers:
            x  = layer(x, memory, src_mask, target_mask)
        return self.norm(x)  # TODO: po co tutaj norm na końcu?


class DecoderLayer(nn.Module):
    def __init__(self, size, mha1, mha2, ff, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.mha1 = mha1
        self.mha2 = mha2 
        self.ff = ff 
        self.res_conn = clones(ResConnectionWithLayerNorm(size, dropout), 3)

    def forward(self, x, memory, src_mask, target_mask):
        # TODO: po co te lambdy?
        # TODO: po co jakikolwiek src_mask?
        x = self.res_conn[0](x, lambda x: self.mha1(x, x, x, target_mask))
        x = self.res_conn[1](x, lambda x: self.mha2(x, memory, memory, src_mask))
        return self.res_conn[2](x, self.ff)


# TODO: po co kolejna maska?
def subsequent_mask(size):
    scores_shape = (1, size, size)
    mask = torch.triu(torch.ones(scores_shape), diagonal=1).type(torch.uint8)

    return mask == 0


def attention(query, key, value, mask=None, dropout=None):
    # query/key/value have shape (n_batches, h, time steps, d_k)
    d_k = query.size(-1)  

    # matmul automatically handles outer dimensions
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    scores_p = scores.softmax(dim=-1)
    if dropout is not None:
        scores_p = dropout(scores_p)
    
    return torch.matmul(scores_p, value), scores_p


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0  # Taka konwencja autorów, bo wtedy
        # niby koszt równoległych operacji jest taki jak jednego heada
        # dla wektorów embeddowanych w oryginalnej wymiarowości

        self.d_k = d_model // h  # = d_q = d_v
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4) 
        # Używamy 3 większe macierze zamiast 3*8 mniejszych macierzy
        # + do tego 4. macierz W_O

        self.scores_p = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # TODO: change query, key, value to x, x below to y
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # Linear projections to queries, keys and values
        query, key, value = [
            lin(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
            # linears[0] to W_q, linears[1] to W_k itd., linears[3] nie użyte
        ]

        # linears[i] przechowuje macierze projekcji (np. dla query) dla
        # wszystkich headów
        # Zmieniamy transposem na (batch_size, h, t, d_k), bo taki
        # shape przyjmuje attention

        # Self-attention
        x, self.scores_p = attention(query, key, value, 
                                     mask=mask, dropout=self.dropout)

        # Concat multiple heads
        # W tym momencie mamy (batch_size, h, t, d_k)
        # a potrzebujemy (batch_size, t, d_model)
        x = (
                x.transpose(1, 2)
                .contiguous()  # Zmienia organizację tensora w pamięci
                              # żeby dane rzeczywiście były odpowiednio
                              # ułożone, a nie tylko metadane były zmienione
                              # po transpose
                .view(batch_size, -1, self.h*self.d_k))

        # Free up memory
        # TODO: zrozumieć czemu akurat tutaj to robimy i nic się nie psuje
        del query
        del key 
        del value 

        return self.linears[-1](x)  # Zwraca (batch_size, t, d_model)


# TODO: czemu akurat tak?
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(
                self.dropout(
                    self.linear1(x).relu()))
    

class Embedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embedding, self).__init__()
        # Mogłoby być Linear, ale szybciej robić lookup
        self.lut = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)  # TODO: po co skalowanie?


class ClassificationHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(ClassificationHead, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return log_softmax(self.linear(x), dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_encodings = torch.zeros(max_len, d_model)
        positions = torch.arange(0., max_len).unsqueeze(1)  # Adds dummy dim
        denoms = torch.exp(
                    torch.arange(0., d_model, 2) * -(math.log(10000) / d_model))
        pos_encodings[:, 0::2] = torch.sin(positions*denoms)
        pos_encodings[:, 1::2] = torch.cos(positions*denoms)

        pos_encodings = pos_encodings.unsqueeze(0)
        # TODO: po co te squeeze'y?

        self.register_buffer('pos_encodings', pos_encodings)
        # Use register_buffer() to add nontrainable parameters to state_dict
        # - won't be returned by model.parameters()

    def forward(self, x):
        x = x + self.pos_encodings[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_preproc, target_preproc,
                 head):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_preproc = src_preproc
        self.target_preproc = target_preproc
        self.head = head 

    def forward(self, src, target, src_mask, target_mask):
        return self.decode(
                self.encode(src, src_mask), src_mask, target, target_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_preproc(src), src_mask)

    def decode(self, memory, src_mask, target, target_mask):
        return self.decoder(self.target_preproc(target), memory, 
                            src_mask, target_mask)


def build_model(src_vocab, target_vocab, n=6, d_model=512, d_ff=2048, 
                h=8, dropout=0.1):
    c = copy.deepcopy
    mha = MultiHeadedAttention(h, d_model)
    ff = FeedForward(d_model, d_ff, dropout)
    pos_encoding = PositionalEncoding(d_model, dropout)

    encoder = Encoder(EncoderLayer(d_model, c(mha), c(ff), dropout), n)
    decoder = Decoder(DecoderLayer(d_model, c(mha), c(mha), c(ff), dropout), n)
    src_preproc = nn.Sequential(Embedding(d_model, src_vocab), 
                                c(pos_encoding))
    target_preproc = nn.Sequential(Embedding(d_model, target_vocab), 
                                c(pos_encoding))
    head = ClassificationHead(d_model, target_vocab)

    model = EncoderDecoder(
                encoder,
                decoder,
                src_preproc,
                target_preproc,
                head)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)  # TODO: doczytać i zrozumieć

    return model


# TODO: zrozumieć
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


class TrainingState:
    def __init__(self):
        self.step = 0  # Steps in current epoch
        self.accum_step = 0  # Gradient accumulation steps
        self.samples_used = 0  # Samples used
        self.tokens = 0  # Tokens processed


def run_epoch(
        data_gen,
        model,
        criterion,
        optimizer=None,
        lr_scheduler=None,
        n_batches=None,
        curr_epoch=None,
        n_epochs=None,
        mode='train'):
    assert mode in ['train', 'eval']

    tqdm_it = tqdm(enumerate(data_gen), total=n_batches, leave=True)

    if mode == 'train':
        tqdm_it.set_description(f'Epoch [{curr_epoch+1}/{n_epochs}]')

    for i, batch in tqdm_it:
        out = model.forward(batch.src, batch.target, 
                            batch.src_mask, batch.target_mask)
        loss, loss_node = criterion(out, batch.target_y, batch.n_tokens)

        if mode == 'train':
            loss_node.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()

        if mode == 'eval':
            tqdm_it.set_description(f'Validation batch [{i+1}/{n_batches}]')

        tqdm_it.set_postfix(loss=loss.item())
        del loss 
        del loss_node


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


def label_smoothing_test():
    criterion = LabelSmoothing(5, 0, 0.1)
    preds = torch.Tensor([
        [0, 0.2, 0.7, 0.1, 0],
        [0, 0.2, 0.7, 0.1, 0]])
    
    criterion(x=preds, target=torch.LongTensor([1, 2]))
    print(criterion.true_dist)


def inference_test():
    test_model = build_model(11, 11, n=2, h=4)
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


class SourceTargetCopyLoss:
    def __init__(self, head, criterion):
        self.head = head 
        self.criterion = criterion

    def __call__(self, x, y, norm):
        y_pred = self.head(x)
        loss = self.criterion(
                y_pred.contiguous().view(-1, y_pred.size(-1)),
                y.contiguous().view(-1))
        loss /= norm

        return loss * norm, loss  # WTF?


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


def train_copy_task_test():
    vocab_size = 11
    batch_size = 64
    n_batches = 32
    n_epochs = 20

    criterion = LabelSmoothing(n_classes=vocab_size, padding_idx=0, 
                               smoothing=0)

    model = build_model(vocab_size, vocab_size, n=2)

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
            SourceTargetCopyLoss(model.head, criterion),
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
            SourceTargetCopyLoss(model.head, criterion),
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
        spacy_en = spacy.load('de_core_web_sm')

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




if __name__ == '__main__':
    # inference_test()
    # label_smoothing_test()
    train_copy_task_test()


    

    
        


    
        


