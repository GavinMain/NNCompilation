import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset
import json
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Self Attention Head
class SelfAttentionHead(nn.Module):
    def __init__(self, head_size, embedding_dim, sequence_length, dropout=0.2):
        super().__init__()
        #k,q,v
        self.key = nn.Linear(embedding_dim, head_size, bias=False)
        self.query = nn.Linear(embedding_dim, head_size, bias=False)
        self.value = nn.Linear(embedding_dim, head_size, bias=False)
        
        #upper triangular mask
        self.register_buffer('tril', torch.tril(torch.ones(sequence_length, sequence_length)))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape 
        
        k = self.key(x)  
        q = self.query(x) 
        v = self.value(x)
        
        s = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 
        s_masked = s.masked_fill(self.tril[:T, :T] == 0, float('-inf')) 
        
        a = F.softmax(s_masked, dim=-1) 
        a = self.dropout(a)
        
         
        z = a @ v
        
        return z
    
    
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, embedding_dim, sequence_length, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttentionHead(head_size, embedding_dim, sequence_length, dropout) for _ in range(num_heads)])
        self.linear = nn.Linear(head_size * num_heads, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #Concatenate
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        
        #Mix
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class FeedFoward(nn.Module):
    def __init__(self, embedding_dim, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, 4 * embedding_dim)
        self.linear2 = nn.Linear(4 * embedding_dim, embedding_dim)
        self.Dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.Dropout(self.linear2(x))
        return x
    
class AttentionBlock(nn.Module):
    def __init__(self, embedding_dim, n_head, sequence_length, dropout=0.2):
        super().__init__()
        head_size = embedding_dim // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size, embedding_dim, sequence_length, dropout)
        self.feed_forward = FeedFoward(embedding_dim, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.self_attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x
    
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, sequence_length, n_block, n_head, dropout=0.2):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.positional_encoding = nn.Embedding(sequence_length, embedding_dim)
        
        self.attention_blocks = nn.Sequential(*[AttentionBlock(embedding_dim=embedding_dim, n_head=n_head, sequence_length=sequence_length, dropout=dropout) for _ in range(n_block)])
        
        self.norm = nn.LayerNorm(embedding_dim) 
        self.linear = nn.Linear(embedding_dim, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        #embedding
        tok_emb = self.embedding_layer(idx) 
        
        #positional encoding
        pos_emb = self.positional_encoding(torch.arange(T, device=device)) 
        x = tok_emb + pos_emb 
        
        #multiheaded attention (several blocks)
        x = self.attention_blocks(x) 
        x = self.norm(x) 
        out = self.linear(x) 
        
        if targets is None:
            loss = None
        else:
            B, T, C = out.shape
            out = out.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(out, targets)
        return out, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, sequence_length):
        for _ in range(max_new_tokens):
            #only keep the last seq length tokens
            idx_cond = idx[:, -sequence_length:]
            
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :] 

            probs = F.softmax(logits, dim=-1) 
            prediction = torch.multinomial(probs, num_samples=1) 
    
            #concatenate latest prediction to next input
            idx = torch.cat((idx, prediction), dim=1)
            
        return idx

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

class CharTokenizer:
    def __init__(self, vocab_file=None):
        self.char2id = {}
        self.id2char = {}
        if vocab_file and os.path.exists(vocab_file):
            with open(vocab_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.char2id = data['char2id']
                self.id2char = {int(v): k for k, v in self.char2id.items()}

    def build_vocab(self, text, special_tokens=None):
        unique_chars = sorted(list(set(text)))
        self.char2id = {}
        idx = 0

        if special_tokens:
            for token in special_tokens:
                self.char2id[token] = idx
                idx += 1

        for ch in unique_chars:
            if ch not in self.char2id:
                self.char2id[ch] = idx
                idx += 1

        self.id2char = {v: k for k, v in self.char2id.items()}
        
    def get_vocab_size(self):
        return len(self.char2id)

    def save_vocab(self, file_name):
        if not file_name.endswith('.json'):
            file_name += '.json'

        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump({'char2id': self.char2id}, f, ensure_ascii=False, indent=2)

    def load_vocab(self, file_name):
        if not file_name.endswith('.json'):
            file_name += '.json'

        if not os.path.exists(file_name):
            raise FileNotFoundError(f"Vocabulary file '{file_name}' not found.")

        with open(file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.char2id = data['char2id']
            self.id2char = {int(v): k for k, v in self.char2id.items()}

    def encode(self, text):
        ids = []
        unk_id = self.char2id.get("<unk>", None)
        for ch in text:
            if ch in self.char2id:
                ids.append(self.char2id[ch])
            elif unk_id is not None:
                ids.append(unk_id)
            else:
                raise ValueError(f"Unknown character '{ch}' encountered but <unk> token not defined")
        return ids

    def decode(self, ids):
        chars = []
        for i in ids:
            chars.append(self.id2char.get(i, "<unk>"))
        return ''.join(chars)
    
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