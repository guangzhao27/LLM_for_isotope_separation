import os
os.environ['HF_HOME'] = '/pscratch/sd/g/gzhao27/huggingface'
import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import statistics

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig

from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import transformers
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from seqeval.metrics import f1_score, accuracy_score

model_name = "allenai/scibert_scivocab_cased"
fine_tune_save_path = "fine_tuned_mlm"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(torch.cuda.get_device_name(0))

num_cores = os.cpu_count()
print("Number of CPU cores:", num_cores)

import os

def get_all_file_paths(directory):
    file_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    
    return file_paths

directory_path = "./sample_articles/"
file_paths_list = get_all_file_paths(directory_path)

from datasets import load_dataset
ssl_dataset = load_dataset("text", split='train', data_files=file_paths_list)

#filter empty rows
ssl_dataset = ssl_dataset.filter(lambda example: example['text'])

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_name)
def preprocess_function(examples):
    return tokenizer(examples["text"])

ssl_tokenized = ssl_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=128,
    remove_columns=ssl_dataset.column_names,
)

block_size = 128


def group_texts(examples):
    # Concatenate all texts.
    
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    
    #print(total_length)
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    return result

ssl_lm = ssl_tokenized.map(group_texts, batched=True, num_proc=1)
from transformers import DataCollatorForLanguageModeling

tokenizer.add_special_tokens({'pad_token': '[PAD]'})
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer
model = AutoModelForMaskedLM.from_pretrained(model_name)
ssl_grouped = ssl_lm.train_test_split(test_size=0.2)
training_args = TrainingArguments(
    output_dir="fine_tuned_mlm_model",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=15,
    weight_decay=0.01,
    push_to_hub=False,
    report_to="tensorboard",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ssl_grouped['train'],
    eval_dataset=ssl_grouped['test'],
    data_collator=data_collator,
)

trainer.train()
model.save_pretrained(fine_tune_save_path)