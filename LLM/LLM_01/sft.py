import time
import os
import random
from _model import LanguageModel, train_step, evaluate, BytePairTokenizer, SFTDataset, device
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

#Hyper Parameters
batch_size = 128
sequence_length = 256
max_epochs = 10
save_interval = 5
learning_rate = .00001
embedding_dim = 240
num_attention_heads = 6
num_attention_blocks = 6
train_test_split = 0.8

#Other Parameters
input_file = "sft_data.txt"
vocab_file = "vocab.json"
pretrain_model_path = "pretrain.pth"
model_path = "sft.pth"
log_file = "sft_log.txt"

special_tokens = ["<eos>", "<pad>", "<unk>"]

max_tokens = 128

def load_samples(paths, tokenizer, sequence_length):
    samples = []
    for path in paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        if len(tokenizer.encode(line)) >= sequence_length:
                            continue
                        samples.append(line)
    return samples

def main():
    tokenizer = BytePairTokenizer()

    if not os.path.exists(vocab_file):
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer.build_vocab(text, special_tokens=special_tokens)
        tokenizer.save_vocab(vocab_file)
    else:
        tokenizer.load_vocab(vocab_file)

    vocab_size = tokenizer.get_vocab_size()

    model = LanguageModel(vocab_size, embedding_dim, sequence_length, num_attention_blocks, num_attention_heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    initial_epoch = 0

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_epoch = checkpoint["epoch"]
    elif os.path.exists(pretrain_model_path):
        checkpoint = torch.load(pretrain_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_epoch = checkpoint["epoch"]

    samples = load_samples([input_file], tokenizer, sequence_length)
    random.shuffle(samples)

    n = int(len(samples) * train_test_split)
    train_samples = samples[:n]
    val_samples = samples[n:]

    train_dataset = SFTDataset(train_samples, tokenizer, sequence_length)
    val_dataset = SFTDataset(val_samples, tokenizer, sequence_length)

    num_workers = 4 if device == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=(device=='cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(device=='cuda'))

    print("LLM_01 SFT: PyTorch Layers with Byte Pair Tokenizer:")
    print("Device: ", device)
    print("Epoch: ", initial_epoch)
    print("Vocab Size: ", vocab_size)
    print("Train Samples: ", len(train_samples))
    print("Val Samples: ", len(val_samples))

    for epoch in range(1, max_epochs + 1):
        time_start = time.time()

        train_loss = train_step(model, train_loader, optimizer)

        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")

        if epoch % save_interval == 0:
            val_loss = evaluate(model, val_loader)
            print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            checkpoint = {
                "vocab_size": vocab_size,
                "embedding_dim": embedding_dim,
                "sequence_length": sequence_length,
                "attention_blocks": num_attention_blocks,
                "attention_heads": num_attention_heads,
                "epoch": epoch+initial_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }

            torch.save(checkpoint, model_path)

            with open(log_file, "a") as f:
                f.write(f"Epoch {initial_epoch+epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Time: {time.time() - time_start:.2f}s\n")


if __name__ == "__main__":
    main()
