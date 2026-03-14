import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

#Hyper Parameters
max_new_tokens = 200
temperature = 0.9
top_p = 0.9

#Other Parameters
base_model_path = "./gpt2-pretrain"
lora_path = "./gpt2-sft"
save_file = "rlhf_dataset.json"

def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = decoded.split("A:")[-1].strip()
    return answer

def save_preference(prompt, chosen, rejected):
    try:
        with open(save_file, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    with open(save_file, "w") as f:
        json.dump(data, f, indent=2)

    print("Preference saved.\n")

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

    print("RLHF Preference Collection Started.")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Prompt: ")

        if user_input.lower() == "exit":
            break

        prompt = f"Q: {user_input} A: "

        print("\nGenerating responses...\n")

        response1 = generate_response(model, tokenizer, prompt)
        response2 = generate_response(model, tokenizer, prompt)

        print("---------- OPTION 1 ----------")
        print(response1)
        print("\n---------- OPTION 2 ----------")
        print(response2)
        print("\nChoose better response:")
        print("1 = Option 1")
        print("2 = Option 2")
        print("0 = Skip")

        choice = input("Your choice: ")

        if choice == "1":
            save_preference(prompt, response1, response2)
        elif choice == "2":
            save_preference(prompt, response2, response1)
        elif choice == "0":
            print("Skipped.\n")
        else:
            break

    print("Session ended.")

if __name__ == "__main__":
    main()
