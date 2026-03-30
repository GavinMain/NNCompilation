import torch
from _model import DiffusionModel, show_images, device

model_path = "checkpoint.pth"
num_samples = 16

def main():
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    model = DiffusionModel(
        vocab_size=checkpoint["vocab_size"],
        in_channels=checkpoint["in_channels"],
        timesteps=checkpoint["timesteps"],
        beta_start=checkpoint["beta_start"],
        beta_end=checkpoint["beta_end"],
        base_channels=checkpoint["base_channels"],
        channel_mults=checkpoint["channel_mults"],
        num_res_blocks=checkpoint["num_res_blocks"],
        time_emb_dim=checkpoint["time_emb_dim"],
        text_emb_dim=checkpoint["text_emb_dim"],
        dropout=checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.text_encoder.token2id = checkpoint["text_id_map"]
    model.eval()

    prompt = input("Prompt: ")
    prompt_ids = model.text_encoder.encode(prompt)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.sample(num_samples, idx)
        show_images(out)

if __name__ == "__main__":
    main()
