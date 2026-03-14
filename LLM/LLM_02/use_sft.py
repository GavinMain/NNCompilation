import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

#Other Parameters
base_model_path = "./gpt2-pretrain"
lora_path = "./gpt2-sft"
max_new_tokens = 200

def main():
    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    base_model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()

    user_input = input("Prompt: ")

    prompt = f"Q: {user_input} A: "
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = generated_text.split("A:")[-1].strip()
    print("\nGenerated text:")
    print(answer)

if __name__ == "__main__":
    main()
