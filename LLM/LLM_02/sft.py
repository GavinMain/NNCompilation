import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

#Hyper Parameters
batch_size = 1
max_epochs = 1
learning_rate = 2e-4
block_size = 1024
gradient_accumulation_steps = 8

#Other Parameters
base_model_path = "./gpt2-pretrain"
dataset_path = "./qa_data.jsonl"
output_dir = "./gpt2-sft"

def main():
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    print(f"Tokenizer loaded. Pad token: {tokenizer.pad_token}, EOS token: {tokenizer.eos_token}")

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    print("Dataset size:", len(dataset))
    print("Sample:", dataset[0])

    def tokenize_function(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            max_length=block_size,
            padding="max_length",
        )

        labels = [
            token if token != tokenizer.pad_token_id else -100
            for token in tokens["input_ids"]
        ]

        tokens["labels"] = labels
        return tokens

    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=dataset.column_names,
        num_proc=os.cpu_count(),
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=max_epochs,
        learning_rate=learning_rate,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
