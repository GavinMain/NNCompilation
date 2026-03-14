import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import json
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        sequence_length,
        n_block,
        n_head,
        dropout=0.2,
    ):
        super().__init__()

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.position_encoding = nn.Embedding(sequence_length, embedding_dim)

        #The decoder layer has built in cross attention, so the encoder layer is used instead
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=n_head,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_block,
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)

        #Precomputes positional encodings for efficiency
        self.register_buffer(
            "pos_ids",
            torch.arange(sequence_length),
            persistent=False,
        )
        
        self.register_buffer(
            "triangular_mask",
            torch.triu(
                torch.full((sequence_length, sequence_length), float("-inf")),
                diagonal=1,
            ),
            persistent=False,
        )


        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        #embedding + positional encoding
        tok_emb = self.embedding_layer(idx)
        pos_emb = self.position_encoding(self.pos_ids[:T])
        x = tok_emb + pos_emb

        #runs encoding layer with triangular mask, making it masked attention
        x = self.transformer(
            x,
            mask=self.triangular_mask[:T, :T],
        )

        x = self.norm(x)
        logits = self.linear(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
        return logits, loss


    @torch.no_grad()
    def generate(self, idx, max_new_tokens, sequence_length, stopping_id=None):
        for _ in range(max_new_tokens):
            #only keep the last seq length tokens
            idx_cond = idx[:, -sequence_length:]
            
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 

            probs = F.softmax(logits, dim=-1) 
            prediction = torch.multinomial(probs, num_samples=1) 
            
            if stopping_id and stopping_id == prediction:
                break
    
            #concatenate latest prediction to next input
            idx = torch.cat((idx, prediction), dim=1)
            
        return idx
    
def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            logits, loss = model(x, y)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def train_step(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

class BytePairTokenizer:
    def __init__(self, vocab_file=None):
        self.token2id = {}
        self.id2token = {}
        self.merges = []
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.token2id = data['token2id']
                self.id2token = {int(v): k for k, v in self.token2id.items()}
                self.merges = [tuple(m) for m in data.get('merges', [])]

    def build_vocab(self, text, special_tokens=None, max_vocab_size=None, min_frequency=1):
        self.token2id = {}
        idx = 0

        if special_tokens:
            for token in special_tokens:
                self.token2id[token] = idx
                idx += 1

        for ch in sorted(set(text)):
            if ch not in self.token2id:
                self.token2id[ch] = idx
                idx += 1

        words = self._split_into_words(text)
        corpus = [list(w) for w in words]

        self.merges = []
        while True:
            if max_vocab_size is not None and len(self.token2id) >= max_vocab_size:
                break

            pair_counts = {}
            for word_tokens in corpus:
                for i in range(len(word_tokens) - 1):
                    pair = (word_tokens[i], word_tokens[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] < min_frequency:
                break

            merged_token = best_pair[0] + best_pair[1]
            if merged_token not in self.token2id:
                self.token2id[merged_token] = idx
                idx += 1
            self.merges.append(best_pair)

            corpus = [self._merge_pair(word_tokens, best_pair, merged_token)
                      for word_tokens in corpus]

        self.id2token = {v: k for k, v in self.token2id.items()}

    def _split_into_words(self, text):
        if not text:
            return []
        words = []
        current = text[0]
        for ch in text[1:]:
            if ch == ' ' or ch == '\n' or ch == '\t':
                words.append(current)
                current = ch
            elif current[-1] in (' ', '\n', '\t'):
                current += ch
            else:
                current += ch
        if current:
            words.append(current)
        return words

    @staticmethod
    def _merge_pair(tokens, pair, merged):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def get_vocab_size(self):
        return len(self.token2id)

    def save_vocab(self, file_name):
        if not file_name.endswith('.json'):
            file_name += '.json'

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump({
                'token2id': self.token2id,
                'merges': [list(m) for m in self.merges],
            }, f, ensure_ascii=False, indent=2)

    def load_vocab(self, file_name):
        if not file_name.endswith('.json'):
            file_name += '.json'

        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Vocabulary file '{file_name}' not found.")

        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.token2id = data['token2id']
            self.id2token = {int(v): k for k, v in self.token2id.items()}
            self.merges = [tuple(m) for m in data.get('merges', [])]

    def encode(self, text):
        ids = []
        unk_id = self.token2id.get("<unk>", None)
        words = self._split_into_words(text)
        for word in words:
            tokens = list(word)
            for pair in self.merges:
                tokens = self._merge_pair(tokens, pair, pair[0] + pair[1])
            for tok in tokens:
                if tok in self.token2id:
                    ids.append(self.token2id[tok])
                elif unk_id is not None:
                    ids.append(unk_id)
                else:
                    raise ValueError(f"Unknown token '{tok}' encountered but <unk> token not defined")
        return ids

    def decode(self, ids):
        tokens = []
        for i in ids:
            tokens.append(self.id2token.get(i, "<unk>"))
        return ''.join(tokens)

class PretrainDataset(Dataset):
    def __init__(self, token_ids, sequence_length):
        self.ids = token_ids
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, len(self.ids) - self.sequence_length)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx:idx + self.sequence_length], dtype=torch.long)
        y = torch.tensor(self.ids[idx+1:idx+self.sequence_length+1], dtype=torch.long)
        return x, y
    
class SFTDataset(Dataset):
    def __init__(self, samples, tokenizer, sequence_length):
        self.samples = samples
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.ignore_index = -100
        self.pad_id = tokenizer.token2id.get("<pad>", 0)
        self.answer_token_ids = tokenizer.encode("A:")

    def __len__(self):
        return len(self.samples)

    def _build_sft_labels(self, labels):
        T = labels.size(0)
        a = torch.tensor(self.answer_token_ids)

        start = None
        for i in range(T - len(a)):
            if torch.equal(labels[i : i + len(a)], a):
                start = i + len(a)
                break
        if start is not None:
            labels[:start] = self.ignore_index

        return labels

    def __getitem__(self, idx):
        text = self.samples[idx]

        input_ids = self.tokenizer.encode(text)
        input_ids = input_ids[: self.sequence_length + 1]

        if len(input_ids) < self.sequence_length + 1:
            input_ids += [self.pad_id] * (self.sequence_length + 1 - len(input_ids))

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        x = input_ids[:-1]
        y = input_ids[1:].clone()

        y = self._build_sft_labels(y)
        y[y == self.pad_id] = self.ignore_index

        return x, y