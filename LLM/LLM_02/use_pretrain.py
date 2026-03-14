import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#Other Parameters
base_model_path = "./gpt2-pretrain"
max_new_tokens = 200

def main():
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.eval()

    user_input = input("Prompt: ")

    inputs = tokenizer(user_input, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\nGenerated text:")
    print(generated_text)

if __name__ == "__main__":
    main()
