from transformers import AutoTokenizer, AutoModelForCausalLM

base_save_path = "./gpt2-pretrain"

model_name = "openai-community/gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.save_pretrained(base_save_path)
model.save_pretrained(base_save_path)
