import argparse, logging, random, time, os
from io import open
from dataclasses import dataclass
from typing import List, Tuple

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
device = torch.device("cpu")  # flip to "cuda" if appropriate

SOS_token = "<SOS>"
EOS_token = "<EOS>"
PAD_token = "<PAD>"

SOS_index = 0
EOS_index = 1
PAD_index = 2
MAX_LENGTH = 15  # max decoding steps (not used for truncation)

class Vocab:
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token, PAD_index: PAD_token}
        self.n_words = 3

    def add_sentence(self, sentence: str):
        for w in sentence.strip().split(' '):
            if w:
                self._add_word(w)

    def _add_word(self, w: str):
        if w not in self.word2index:
            self.word2index[w] = self.n_words
            self.word2count[w] = 1
            self.index2word[self.n_words] = w
            self.n_words += 1
        else:
            self.word2count[w] += 1

def split_lines(input_file: str) -> List[Tuple[str,str]]:
    logging.info("Reading lines of %s...", input_file)
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    pairs = [tuple(l.split('|||')) for l in lines]
    return [(a.strip(), b.strip()) for a,b in pairs]

def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    src_vocab, tgt_vocab = Vocab(src_lang_code), Vocab(tgt_lang_code)
    for s,t in split_lines(train_file):
        src_vocab.add_sentence(s); tgt_vocab.add_sentence(t)
    logging.info('%s (src) vocab size: %d', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %d', tgt_vocab.lang_code, tgt_vocab.n_words)
    return src_vocab, tgt_vocab

def tensorize_sentence(vocab: Vocab, sentence: str) -> List[int]:
    idxs = []
    for w in sentence.split():
        if w in vocab.word2index:
            idxs.append(vocab.word2index[w])
        # silently drop OOVs to match original
    idxs.append(EOS_index)
    return idxs

@dataclass
class Example:
    src_ids: List[int]
    tgt_ids: List[int]

class ParallelDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str,str]], src_vocab: Vocab, tgt_vocab: Vocab):
        self.data = []
        for s,t in pairs:
            self.data.append(Example(
                tensorize_sentence(src_vocab, s),
                tensorize_sentence(tgt_vocab, t)
            ))
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]

def collate_examples(batch: List[Example]):
    # make [B, T] padded src and tgt, plus lengths for packing, and decoder inputs/targets
    src_lens = torch.tensor([len(x.src_ids) for x in batch], dtype=torch.long)
    tgt_lens = torch.tensor([len(x.tgt_ids) for x in batch], dtype=torch.long)

    max_src = src_lens.max().item()
    max_tgt = tgt_lens.max().item()

    B = len(batch)
    src = torch.full((B, max_src), PAD_index, dtype=torch.long)
    tgt = torch.full((B, max_tgt), PAD_index, dtype=torch.long)

    # Decoder input starts with SOS; target is the full sequence (incl EOS)
    dec_in = torch.full((B, max_tgt), PAD_index, dtype=torch.long)
    dec_in[:,0] = SOS_index

    for i, ex in enumerate(batch):
        src[i, :len(ex.src_ids)] = torch.tensor(ex.src_ids, dtype=torch.long)
        tgt[i, :len(ex.tgt_ids)] = torch.tensor(ex.tgt_ids, dtype=torch.long)
        # teacher forcing inputs: SOS + gold[:-1]
        if len(ex.tgt_ids) > 1:
            dec_in[i, 1:len(ex.tgt_ids)] = torch.tensor(ex.tgt_ids[:-1], dtype=torch.long)

    # mask for attention (True where real tokens exist)
    src_mask = (src != PAD_index)  # [B, Tsrc]
    return src.to(device), src_lens.to(device), src_mask.to(device), tgt.to(device), tgt_lens.to(device), dec_in.to(device)

class EncoderLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, bidir=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidir else 1
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=PAD_index)
        self.lstm = nn.LSTM(hidden_size, hidden_size//self.num_directions, batch_first=True, bidirectional=bidir)

    def forward(self, src_ids: torch.Tensor, src_lens: torch.Tensor):
        # src_ids: [B, T]
        emb = self.embedding(src_ids)  # [B, T, H]
        packed = pack_padded_sequence(emb, src_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.lstm(packed)
        enc_out, _ = pad_packed_sequence(packed_out, batch_first=True)  # [B, T, H*(dir)]
        if self.num_directions == 2:
            # concat top layer's forward/backward hidden states
            h = torch.cat([h[-2], h[-1]], dim=1).unsqueeze(0)  # [1, B, H]
            c = torch.cat([c[-2], c[-1]], dim=1).unsqueeze(0)
        return enc_out, (h, c)  # enc_out for attention

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Ua = nn.Linear(hidden_size, hidden_size, bias=False)
        self.va = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h_t: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor):
        """
        h_t: [B, H]; enc_out: [B, Tsrc, H]; src_mask: [B, Tsrc] bool
        returns: context [B,H], attn_weights [B,Tsrc]
        """
        # e_ij = v^T tanh(Wa h_t + Ua h_j)
        B, T, H = enc_out.size()
        h_exp = self.Wa(h_t).unsqueeze(1).expand(B, T, H)           # [B,T,H]
        e_ij = self.va(torch.tanh(h_exp + self.Ua(enc_out))).squeeze(-1)  # [B,T]
        e_ij = e_ij.masked_fill(~src_mask, float('-inf'))
        attn = F.softmax(e_ij, dim=1)                               # [B,T]
        context = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # [B,H]
        return context, attn

class LuongAttention(nn.Module):
    """
    Luong global attention (dot/general):
      score(h_t, h_s) = h_t^T h_s        # dot
      score(h_t, h_s) = h_t^T W_a h_s    # general
    """
    def __init__(self, hidden_size, method: str = "dot"):
        super().__init__()
        assert method in ("dot", "general")
        self.method = method
        if method == "general":
            self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, h_t: torch.Tensor, enc_out: torch.Tensor, src_mask: torch.Tensor):
        """
        h_t: [B, H]       (current decoder hidden)
        enc_out: [B, T, H]  (encoder outputs)
        src_mask: [B, T]  (True for real tokens)
        returns: context [B,H], attn_weights [B,T]
        """
        if self.method == "dot":
            # scores: [B, T] = enc_out @ h_t
            scores = torch.bmm(enc_out, h_t.unsqueeze(2)).squeeze(2)
        else:  # general
            # scores: [B, T] = (W_a enc_out) @ h_t
            proj = self.Wa(enc_out)                       # [B,T,H]
            scores = torch.bmm(proj, h_t.unsqueeze(2)).squeeze(2)

        scores = scores.masked_fill(~src_mask, float('-inf'))
        attn = F.softmax(scores, dim=1)                   # [B,T]
        context = torch.bmm(attn.unsqueeze(1), enc_out).squeeze(1)  # [B,H]
        return context, attn

class AttnDecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_index)
        self.dropout = nn.Dropout(dropout_p)
        # self.attn = BahdanauAttention(hidden_size)
        self.attn = LuongAttention(hidden_size)
        self.lstm_cell = nn.LSTMCell(hidden_size, hidden_size)  # input will be pre-projected [emb+ctx]->H
        self.in_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.readout = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward_step(self, y_prev: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor],
                     enc_out: torch.Tensor, src_mask: torch.Tensor):
        """
        One decoding step.
        y_prev: [B] next-input tokens; hidden: (h,c) each [B,H]
        """
        emb = self.dropout(self.embedding(y_prev))              # [B,H]
        ctx, attn = self.attn(hidden[0], enc_out, src_mask)     # [B,H], [B,T]
        dec_in = torch.tanh(self.in_proj(torch.cat([emb, ctx], dim=1)))  # [B,H]
        h_new, c_new = self.lstm_cell(dec_in, hidden)           # [B,H]
        readout = torch.tanh(self.readout(torch.cat([h_new, ctx], dim=1)))
        logits = self.out(readout)                               # [B,V]
        logp = F.log_softmax(logits, dim=1)
        return logp, (h_new, c_new), attn

    def forward(self, dec_in: torch.Tensor, init_state: Tuple[torch.Tensor, torch.Tensor],
                enc_out: torch.Tensor, src_mask: torch.Tensor, teacher_forcing_ratio: float = 0.5):
        """
        dec_in: [B, Tdec] (teacher-forcing inputs, starts with SOS)
        Returns logprobs: [B,Tdec,V]
        """
        B, Tdec = dec_in.size()
        h0, c0 = init_state
        h0, c0 = h0.squeeze(0), c0.squeeze(0)  # [B,H]
        outputs = []
        y_t = dec_in[:,0]  # SOS
        hidden = (h0, c0)

        for t in range(1, Tdec+1):  # produce up to gold length (including EOS)
            logp, hidden, _ = self.forward_step(y_t, hidden, enc_out, src_mask)
            outputs.append(logp.unsqueeze(1))
            if t < Tdec:
                if random.random() < teacher_forcing_ratio:
                    y_t = dec_in[:, t]
                else:
                    y_t = logp.argmax(dim=1)
        return torch.cat(outputs, dim=1)  # [B,Tdec,V]

def clean(strx: str) -> str:
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())

@torch.no_grad()
def translate_batch(encoder, decoder, sentences: List[str], src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    # simple greedy decode for a list of raw strings
    batch = [Example(tensorize_sentence(src_vocab, s), [EOS_index]) for s in sentences]
    src, src_lens, src_mask, _, _, _ = collate_examples(batch)
    enc_out, (h,c) = encoder(src, src_lens)
    B = src.size(0)
    y_t = torch.full((B,), SOS_index, dtype=torch.long, device=device)
    hidden = (h.squeeze(0), c.squeeze(0))
    decoded = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_length):
        logp, hidden, _ = decoder.forward_step(y_t, hidden, enc_out, src_mask)
        y_t = logp.argmax(dim=1)
        for i in range(B):
            if not finished[i]:
                if y_t[i].item() == EOS_index:
                    finished[i] = True
                else:
                    decoded[i].append(tgt_vocab.index2word.get(y_t[i].item(), '<unk>'))
        if finished.all(): break
    return [' '.join(seq) for seq in decoded]

def bleu_on_pairs(encoder, decoder, pairs, src_vocab, tgt_vocab, max_eval=200):
    sample = pairs[:max_eval]
    cands = translate_batch(encoder, decoder, [s for s,_ in sample], src_vocab, tgt_vocab)
    refs = [[clean(t).split()] for _,t in sample]
    cands_clean = [clean(c).split() for c in cands]
    return corpus_bleu(refs, cands_clean)

def show_attention(*args, **kwargs):
    # (kept for API compatibility—plotting attention per-batch would need extra plumbing.)
    pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=1024, type=int)
    ap.add_argument('--n_iters', default=160010, type=int)
    ap.add_argument('--print_every', default=5000, type=int)
    ap.add_argument('--checkpoint_every', default=10000, type=int)
    ap.add_argument('--initial_learning_rate', default=0.0005, type=float)
    ap.add_argument('--src_lang', default='fr')
    ap.add_argument('--tgt_lang', default='en')
    ap.add_argument('--train_file', default='data/fren.train.bpe')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe')
    ap.add_argument('--test_file', default='data/fren.test.bpe')
    ap.add_argument('--out_file', default='out_lstm.txt')
    ap.add_argument('--batch_size', default=32, type=int)
    ap.add_argument('--bidir', action='store_true')
    ap.add_argument('--teacher_forcing', default=0.5, type=float)
    ap.add_argument('--load_checkpoint', nargs=1)
    args = ap.parse_args()

    # Checkpoint or build vocabs
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0], map_location=device)
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']; tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang, args.tgt_lang, args.train_file)

    # Models
    encoder = EncoderLSTM(src_vocab.n_words, args.hidden_size, bidir=args.bidir).to(device)
    decoder = AttnDecoderLSTM(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    train_ds = ParallelDataset(train_pairs, src_vocab, tgt_vocab)
    # Shuffle every epoch via DataLoader; num_workers=0 on cluster CPUs
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_examples)

    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss(ignore_index=PAD_index, reduction='sum')

    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    print_loss_total = 0.0
    sentences_count = 0
    window_start_time = time.time()

    # simple epoch cycling until n_iters examples are consumed
    while iter_num < args.n_iters:
        for src, src_lens, src_mask, tgt, tgt_lens, dec_in in loader:
            if iter_num >= args.n_iters: break
            iter_num += src.size(0)  # count sentences
            encoder.train(); decoder.train()
            optimizer.zero_grad()

            enc_out, (h, c) = encoder(src, src_lens)
            logprobs = decoder(dec_in, (h, c), enc_out, src_mask, teacher_forcing_ratio=args.teacher_forcing)  # [B,T,V]

            # Align loss over gold lengths; build gold index tensor [B,T]
            B, T, V = logprobs.shape
            # We trained to predict positions 1..Tgold; so compare against full tgt
            loss = criterion(logprobs.view(B*T, V), tgt.view(-1))
            denom = (tgt != PAD_index).sum().item()
            batch_loss = loss / max(1, denom)

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()

            print_loss_total += batch_loss.item()
            sentences_count += src.size(0)

            if (iter_num // args.print_every) != ((iter_num - src.size(0)) // args.print_every):
                elapsed = time.time() - window_start_time
                sps = sentences_count / max(1e-9, elapsed)
                avg_loss = print_loss_total
                logging.info('iter_sents:%d/%d  loss_avg:%.4f  throughput: %.1f sentences/sec  (bs=%d)',
                             iter_num, args.n_iters, avg_loss, sps, args.batch_size)
                # quick dev BLEU sample (fast)
                encoder.eval(); decoder.eval()
                try:
                    dev_bleu = bleu_on_pairs(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, max_eval=100)
                    logging.info('Dev BLEU (sampled 100): %.2f', dev_bleu)
                except Exception as e:
                    logging.warning('BLEU eval skipped: %s', e)

                print_loss_total = 0.0
                sentences_count = 0
                window_start_time = time.time()

            if (iter_num // args.checkpoint_every) != ((iter_num - src.size(0)) // args.checkpoint_every):
                state = {'iter_num': iter_num,
                         'enc_state': encoder.state_dict(),
                         'dec_state': decoder.state_dict(),
                         'opt_state': optimizer.state_dict(),
                         'src_vocab': src_vocab,
                         'tgt_vocab': tgt_vocab}
                fname = f'hidden_size_{args.hidden_size}_state_{iter_num:010d}.pt'
                torch.save(state, fname)
                logging.debug('wrote checkpoint to %s', fname)

    # Translate test set (greedy) and write out
    encoder.eval(); decoder.eval()
    out_lines = []
    # do small batches for I/O efficiency
    batch = []
    for s,_ in split_lines(args.test_file):
        batch.append(s)
        if len(batch) == 64:
            outs = translate_batch(encoder, decoder, batch, src_vocab, tgt_vocab)
            out_lines.extend([clean(x) for x in outs]); batch = []
    if batch:
        outs = translate_batch(encoder, decoder, batch, src_vocab, tgt_vocab)
        out_lines.extend([clean(x) for x in outs])

    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for line in out_lines:
            outf.write(line + '\n')

if __name__ == '__main__':
    main()