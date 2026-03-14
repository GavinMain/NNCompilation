import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig

#Hyper Parameters
batch_size = 1
max_epochs = 3
learning_rate = 2e-4
gradient_accumulation_steps = 8
beta = 0.1
max_length = 1024

#Other Parameters
base_model_path = "./gpt2-pretrain"
sft_lora_path = "./gpt2-sft"
rlhf_data_path = "rlhf_dataset.json"
output_dir = "./gpt2-rlhf"

def main():
    dataset = load_dataset("json", data_files=rlhf_data_path, split="train")
    print("Dataset size:", len(dataset))
    print("Sample:", dataset[0])

    tokenizer = AutoTokenizer.from_pretrained(sft_lora_path)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    base_model.resize_token_embeddings(len(tokenizer))
    base_model = prepare_model_for_kbit_training(base_model)

    model = PeftModel.from_pretrained(base_model, sft_lora_path, is_trainable=True)
    model.print_trainable_parameters()
    model.train()

    ref_base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
    )

    ref_base.resize_token_embeddings(len(tokenizer))
    ref_model = PeftModel.from_pretrained(ref_base, sft_lora_path)
    ref_model.eval()

    for param in ref_model.parameters():
        param.requires_grad = False

    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=max_epochs,
        logging_steps=10,
        save_steps=500,
        beta=beta,
        max_length=max_length,
        report_to="none",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("DPO training complete.")

if __name__ == "__main__":
    main()
