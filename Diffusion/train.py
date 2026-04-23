import os
import time
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from _model import DiffusionModel, TextEncoder, ImageDataset, train_step, device

#Hyper Parameters
batch_size = 128
num_epochs = 4000
eval_interval = 100
learning_rate = 1e-4 
cfg_prob = 0.1

#Model Parameters
image_size = (16, 16)
in_channels = 3
timesteps = 200
beta_start = 0.0001
beta_end = 0.02
base_channels = 160
channel_mults = (1, 1, 2, 2)
num_res_blocks = 2
time_emb_dim = 640
text_emb_dim = 128
groups = 32
dropout = 0.1

#Other Parameters
data_path = "../image_set"
text_id_map_path = "text_id_map.json"
model_path = "checkpoint.pth"

def collate_fn(batch):
    images, tokens = zip(*batch)
    images = torch.stack(images)
    tokens = [torch.tensor(t) for t in tokens]
    tokens = pad_sequence(tokens, batch_first=True, padding_value=0)
    return images, tokens

def main():
    images = [f for f in os.listdir(data_path)]
    text = [img.split('.')[0].replace('_', ' ') for img in images]

    texts = []
    for t in text:
        words = t.split()
        for w in words:
            texts.append(w)

    text_encoder = TextEncoder(vocab_size=1, text_emb_dim=text_emb_dim, time_emb_dim=time_emb_dim)

    if os.path.exists(text_id_map_path):
        text_encoder.load_vocab(text_id_map_path)
    else:
        text_encoder.build_vocab(texts, save_path=text_id_map_path)

    vocab_size = text_encoder.get_vocab_size()

    text_tokens = [torch.tensor(text_encoder.encode(t)) for t in text]

    train_data = ImageDataset(
        image_paths=[os.path.join(data_path, img) for img in images],
        text_tokens=text_tokens
    )

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = DiffusionModel(
        vocab_size=vocab_size,
        image_size=image_size,
        in_channels=in_channels,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        time_emb_dim=time_emb_dim,
        text_emb_dim=text_emb_dim,
        groups=groups,
        dropout=dropout,
        cfg_prob=cfg_prob,
    ).to(device)

    model.text_encoder.token2id = text_encoder.token2id

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    current_epoch = 0

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.text_encoder.token2id = checkpoint["text_id_map"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        current_epoch = checkpoint["epoch"]

    print("Diffusion Training:")
    print("Device: ", device)
    print("Epoch: ", current_epoch)
    print("Vocab Size: ", vocab_size)
    print("Data Length: ", len(train_data))

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss = train_step(model, data_loader, optimizer)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        if epoch % eval_interval == 0:
            checkpoint = {
                "vocab_size": vocab_size,
                "image_size": image_size,
                "in_channels": in_channels,
                "timesteps": timesteps,
                "beta_start": beta_start,
                "beta_end": beta_end,
                "base_channels": base_channels,
                "channel_mults": channel_mults,
                "num_res_blocks": num_res_blocks,
                "time_emb_dim": time_emb_dim,
                "text_emb_dim": text_emb_dim,
                "dropout": dropout,
                "groups": groups,
                "epoch": epoch + current_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "text_id_map": model.text_encoder.token2id,
            }
            torch.save(checkpoint, model_path)

if __name__ == "__main__":
    main()
