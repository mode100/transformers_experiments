from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments, DataCollatorForTokenClassification,AutoModelForSequenceClassification
import transformers
import torch
import os

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",use_fast=False)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=6).to(device)

from datasets import load_dataset
from random import randint as rnd

directory = os.path.dirname(__file__)

dataset = load_dataset(
    path="csv",
    data_files={
        "train": rf"{directory}\datas\dog_cat_data1.txt",
        "test": rf"{directory}\datas\dog_cat_data2.txt",
    },
    delimiter="\t",
    column_names=["label","text"]
)

tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["text"])
)

data_collator = transformers.DataCollatorWithPadding(
    tokenizer=tokenizer
)

training_args  = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    load_best_model_at_end=True,
    evaluation_strategy="steps"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)

trainer.train()

def make_prompt():
    data = [rnd(0,1) for i in range(10)]
    data_str = "".join([["dog ","cat "][data[i]] for i in range(10)])
    return data_str

def get_answer(data_str):
    arr = data_str.split("dog")
    count = 0
    for a in arr:
        if "cat" in a: count += 1
    return count

def eval_model():
    num_all = 1000
    num_pass = 0
    num_fail = 0
    for i in range(num_all):
        prompt = make_prompt()
        inputs = tokenizer(prompt, return_tensors="pt").to('cuda:0')
        outputs = finetuned_model(**inputs)
        predictions = outputs.logits.argmax(dim=-1).item()
        answer = get_answer(prompt)
        isCorrect = predictions == answer
        if isCorrect: 
            num_pass += 1
        else:
            num_fail += 1
            print(f"prompt:{prompt}, prediction:{predictions}, answer:{answer}")
    print(f"accuracy: {num_pass}/{num_all}, rate: {num_pass/num_all*100}%")


finetuned_model = trainer.model
finetuned_model.eval()

eval_model()

