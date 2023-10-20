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

from sklearn.model_selection import train_test_split

import transformers
from transformers import BertForTokenClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)
# **Preprocessing data**
# Reading data
import json


#open the file, and format correctly
f =open('all.jsonl', 'r')
json_object = json.dumps(f.readlines(), indent=4)
f.close()

#save better format into file
p = open('sample.txt', 'w')
for i in json_object:
    p.write(i)
p.close()

#open new file, and save each 
j = open('sample.txt', 'r')
text = json.loads(j.read())
j.close()

#compile all json dicts into a list
info = []
for i in text:
    info.append(json.loads(str(i)))
    
def degreekify(char):
    #char will be a character
    greek = {'α': '[alpha]', 'β':'[beta]', 'γ': '[gamma]', 'δ':'[delta]', 'ε': '[epsilon]', 'ζ':'[zeta]', 'η': '[eta]',
            'θ':'[theta]', 'ι': '[iota]', 'κ':'[kappa]', 'λ':'[lambda]', 'μ': '[mu]', 'ν':'[nu]', 'ξ':'[xi]', 'ο':'[omicron]', 'π':'[pi]', 'ρ':'[rho]',
            'σ': '[sigma]', 'τ': '[tau]', 'υ':'[upsilon]', 'φ':'[phi]', 'χ':'[chi]', 'ψ':'[psi]', 'ω':'[omega]' }
    if char in greek:
        return greek[char]
    else:
        return char
def pre_process(text, annotations):
    #text will be the straight sentence, info[i]['text']
    #annotations will be the list of labels, must be info[i]['annotations']
    
    text_dict = []
    
    for i in range(len(text)):
        text_dict.append(degreekify(text[i]))
    
    
    
    ann_indices = []
    def fun(x):
        return x['start_offset']
    annotations.sort(key=fun)
    
    
    
    if len(annotations)==0:
        ann_indices.append([[0, len(text)],0])
    else:
        ann_indices.append([[0, annotations[0]['start_offset']], 0])
        for i in range(len(annotations)-1):
            ann_indices.append([[annotations[i]['start_offset'], annotations[i]['end_offset']], data_tags.index(annotations[i]['label'])])
            ann_indices.append([[annotations[i]['end_offset'], annotations[i+1]['start_offset']], 0])
            
        ann_indices.append([[annotations[-1]['start_offset'], annotations[-1]['end_offset']], data_tags.index(annotations[-1]['label'])])
        ann_indices.append([[annotations[-1]['end_offset'], len(text)], 0])
         
    
    labels = []
    sentences = []
    for a in ann_indices:
        
        if a[0][1]-a[0][0] !=0:
            together = ''
            for i in range(a[0][0], a[0][1]):
                together += text_dict[i]
                
            toke = together.split()
            sentences.extend(toke)
            t = len(toke)
            if t != 0:
                temp = [data_tags[a[1]+1]] * t
                if a[1] != 0:
                    temp[0] = data_tags[a[1]]
                labels.extend(temp)
 
    return labels, sentences
def reduce(sent, label, slist, llist):
    lens = len(sent)
    if lens < 256:
        slist.append(sent)
        llist.append(label)
    else:
        t = lens//2
        return reduce(sent[:t], label[:t], slist, llist), reduce(sent[t:], label[t:], slist, llist)
#create labels
data_tags = ['ahhhhhhhhhhhhhhhhhhhh','0','Metal', 'M-cont' , 'Element', 'E-cont', 'Acid', 'A-cont', 'Yield' , 'Y-cont', 'Separation Method' , 'S-cont', 'Resin', 'R-cont', 'Method of Analysis', 'T-cont', 'pH', 'P-cont', 'Chemical Compound', 'H-cont', 'Organic solvent', 'O-cont', 'Element Group', 'G-cont', 'Inorganic Solvent', 'I-cont', 'Flowrate', 'F-cont', 'Acid Concentration', 'C-cont', 'Reagent', 'X-cont']

sent_test, label_test = [], []
sentences, labels = [], []

for i in range(len(info)):
    l, s = pre_process(info[i]['text'], info[i]['entities'])
  
    if i % 5 == 0:
        reduce(s,l,sent_test, label_test)

    else:
        reduce(s,l,sentences, labels)


data_tags = data_tags[1:]


# Determine the list of tags
tag_values = data_tags
print(tag_values)

tag_values.append("PAD")
print(tag_values)

tag2idx = {t: i for i, t in enumerate(tag_values)}
print(tag2idx)
    
    
idx2tag = {value: key for key, value in tag2idx.items()}
bio_labels = [
    'O',
    'B-Metal',
    'I-Metal',
    'B-Element',
    'I-Element',
    'B-Acid',
    'I-Acid',
    'B-Yield',
    'I-Yield',
    'B-SeparationMethod',
    'I-SeparationMethod',
    'B-Resin',
    'I-Resin',
    'B-MethodOfAnalysis',
    'I-MethodOfAnalysis',
    'B-pH',
    'I-pH',
    'B-ChemicalCompound',
    'I-ChemicalCompound',
    'B-OrganicSolvent',
    'I-OrganicSolvent',
    'B-ElementGroup',
    'I-ElementGroup',
    'B-InorganicSolvent',
    'I-InorganicSolvent',
    'B-Flowrate',
    'I-Flowrate',
    'B-AcidConcentration',
    'I-AcidConcentration',
    'B-Reagent',
    'I-Reagent',
    'O'
]

label_list = data_tags[1:-1:2]
label_list

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from huggingface_hub import login
login(token='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')
model_name_or_path = "meta-llama/Llama-2-13b-hf"

model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main", 
                                             cache_dir='/pscratch/sd/g/gzhao27/huggingface')

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, cache_dir='/pscratch/sd/g/gzhao27/huggingface')

def prompt_generate(info_item):
    s = ""
    for e in info_item['entities']:
        temp_dict = dict()
        temp_dict['word'] = info_item['text'][e['start_offset']:e['end_offset']]
        # temp_dict['start_offset'] = e['start_offset']
        # temp_dict['end_offset'] = e['end_offset']
        temp_dict['label'] = e['label']

        
        s += f"{temp_dict}\n"

    return s

temp_info = info[2]

temp_info['text'] = temp_info['text'][:500]
temp_info['entities'] = [e for e in temp_info['entities'] if e['end_offset']<500]
prompt = f"""Context: Please perform Named Entity Recognition (NER) on the following text with label entities [{"] [".join(label_list)}]: 
Q: {info[0]['text']}
A: {prompt_generate(info[0])}
Q: {temp_info['text']}
A: {prompt_generate(temp_info)}
Q: {info[4]['text'][:500]}
A: 
    """
print(prompt)

print("\n\n*** Generate:")

input_ids = tokenizer(prompt, return_tensors='pt').input_ids.cuda()
output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
print(tokenizer.decode(output[0]))