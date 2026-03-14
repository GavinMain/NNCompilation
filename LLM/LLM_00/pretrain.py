import time
import os
from _model import LanguageModel, train_step, CharTokenizer, PretrainDataset, device
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

#Hyper Parameters
batch_size = 128 
sequence_length = 256 
max_epochs = 1
save_interval = 1
learning_rate = .001
embedding_dim = 240
num_attention_heads = 6
num_attention_blocks = 6

#Other Parameters
input_file = "input.txt"
vocab_file = "vocab.json"
model_path = "checkpoint.pth"
token_ids_file = "token_ids.pt"
log_file = "log.txt"

max_tokens = 128

def main():
    with open(input_file, "r", encoding="utf-8") as f:
        text = f.read()
        
    tokenizer = CharTokenizer()
        
    if not os.path.exists(vocab_file):
        tokenizer.build_vocab(text)
        tokenizer.save_vocab(vocab_file)
    else:
        tokenizer.load_vocab(vocab_file)
    
    vocab_size = tokenizer.get_vocab_size()
        
    if os.path.exists(token_ids_file):
        data_ids = torch.load(token_ids_file)
    else:
        data_ids = tokenizer.encode(text)
        torch.save(data_ids, token_ids_file)
        
    model = LanguageModel(vocab_size, embedding_dim, sequence_length, num_attention_blocks, num_attention_heads).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    initial_epoch = 0
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        initial_epoch = checkpoint["epoch"]
    
    train_ids = data_ids
    
    train_dataset = PretrainDataset(train_ids, sequence_length)
    
    num_workers = 4 if device == 'cuda' else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=(device=='cuda'))

    print("LLM_00: Basic Implementation with Char Tokenizer:")
    print("Device: ", device)
    print("Epoch: ", initial_epoch)
    print("Vocab Size: ", vocab_size)
    print("Data Length: ", len(train_dataset))
    
    for epoch in range(1, max_epochs + 1):
        time_start = time.time()
        
        train_loss = train_step(model, train_loader, optimizer)
        
        print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
        
        if epoch % save_interval == 0:
            
            idx = model.generate(torch.tensor(tokenizer.encode(" "), dtype=torch.long, device=device).unsqueeze(0), max_tokens, sequence_length)
            generated_text = tokenizer.decode(idx[0].cpu().tolist())

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
                f.write(f"Epoch {initial_epoch+epoch}, Train Loss: {train_loss:.4f}, Time: {time.time() - time_start:.2f}s, Sample Text: {generated_text}\n")
            

if __name__ == "__main__":
    main()