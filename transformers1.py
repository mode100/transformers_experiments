from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import os

tokenizer = AutoTokenizer.from_pretrained("rinna/japanese-gpt2-small",use_fast=False)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small").to(device)

from datasets import load_dataset
from random import randint as rnd

directory = os.path.dirname(__file__)

dataset = load_dataset(
    path="text",
    data_files={
        "train": rf"{directory}\datas\princess_data1.txt",
        "test": rf"{directory}\datas\princess_data2.txt",
    }
)

tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["text"])
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args  = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    load_best_model_at_end=True,
    evaluation_strategy="steps"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator
)

trainer.train()

# input_prompt()
finetuned_model = trainer.model.to("cpu")

prompt = ""
prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids

outputs = finetuned_model.generate(
    prompt_ids,
    do_sample = True,
    max_length = 30,
    num_return_sequences = 100,
    repetition_penalty = 1.1,
)

decoded = tokenizer.batch_decode(outputs, skip_special_tokens= True)
print(decoded)
