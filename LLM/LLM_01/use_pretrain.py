import torch
from _model import LanguageModel, BytePairTokenizer, device

model_path = "pretrain.pth"
vocab_file = "vocab.json"
max_tokens = 128

def main():
    tokenizer = BytePairTokenizer()
    tokenizer.load_vocab(vocab_file)

    checkpoint = torch.load(model_path, map_location=device)
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    model = LanguageModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embedding_dim"],
        sequence_length=checkpoint["sequence_length"],
        n_block=checkpoint["attention_blocks"],
        n_head=checkpoint["attention_heads"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    user_input = input("Prompt: ")

    prompt_ids = tokenizer.encode(user_input)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out = model.generate(idx, max_tokens, checkpoint["sequence_length"])

    generated_text = tokenizer.decode(out[0].tolist())
    print("\nGenerated text:")
    print(generated_text)

if __name__ == "__main__":
    main()
