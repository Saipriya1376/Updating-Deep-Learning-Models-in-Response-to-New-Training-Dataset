import torch
from tqdm import tqdm
import time
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
import os
import gc
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
import numpy as np

MAX_LEN = 512

def tokenize_sent(sentence, tokenizer):

    tokenized_sentence = []
    sentence = str(sentence).strip()

    for word in sentence.split():
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)
        tokenized_sentence.extend(tokenized_word)

    return tokenized_sentence

class qqp_dataset(Dataset):
    def __init__(self, sentence1, sentence2, label, tokenizer, max_len):
        self.len = len(sentence1)
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label
        self.max_len = max_len
        self.tokenizer = tokenizer
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sent1 = self.sentence1[idx]
        sent2 = self.sentence2[idx]
        label = self.label[idx]
        target = []
        target.append(label)
        
        token_type_ids = []
        token_type_ids.append(0)
        sent1 = tokenize_sent(sent1,self.tokenizer)
        sent2 = tokenize_sent(sent2,self.tokenizer)
        for i in enumerate(sent1):
            token_type_ids.append(0)
        token_type_ids.append(1)
        for i in enumerate(sent2):
            token_type_ids.append(1)
        token_type_ids.append(1)
        
        
        input_sent = ['[CLS]'] + sent1 + ['[SEP]'] + sent2 + ['[SEP]']
        # print(input_sent)
        input_sent = input_sent + ['[PAD]' for _ in range(self.max_len - len(input_sent))]
        token_type_ids = token_type_ids + [0 for _ in range(self.max_len - len(token_type_ids))]
        # print(input_sent)
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in input_sent]
        ids = self.tokenizer.convert_tokens_to_ids(input_sent)
        # print(len(ids))
        return {
            'index' : idx,
            'ids' : torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

def load_qqp(file_path,tokenizer):
    sentence1_list = []
    sentence2_list = []
    target_label_list = []

    dataframe = pd.read_csv(file_path)
    sentence1_list = dataframe['question1'].tolist()
    sentence2_list = dataframe['question2'].tolist()
    target_label_list = dataframe['is_duplicate'].tolist()

    path = file_path.split('.')
    np.savetxt('./QQP/qqp_groundtruth.txt', target_label_list, '%s')

    print(sentence1_list[0])
    print(len(sentence1_list))
    print(sentence2_list[0])
    print(len(sentence2_list))
    data = qqp_dataset(sentence1_list, sentence2_list, target_label_list, tokenizer, MAX_LEN)
    return data

class paws_dataset(Dataset):
    def __init__(self, sentence1, sentence2, label, tokenizer, max_len):
        self.len = len(sentence1)
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.label = label
        self.max_len = max_len
        self.tokenizer = tokenizer
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        sent1 = self.sentence1[idx]
        sent2 = self.sentence2[idx]
        label = self.label[idx]
        # print(sent1)
        target = []
        target.append(label)
        
        token_type_ids = []
        token_type_ids.append(0)
        sent1 = tokenize_sent(sent1,self.tokenizer)
        sent2 = tokenize_sent(sent2,self.tokenizer)
        for i in enumerate(sent1):
            token_type_ids.append(0)
        token_type_ids.append(1)
        for i in enumerate(sent2):
            token_type_ids.append(1)
        token_type_ids.append(1)
        
        
        input_sent = ['[CLS]'] + sent1 + ['[SEP]'] + sent2 + ['[SEP]']
        # print(input_sent)
        input_sent = input_sent + ['[PAD]' for _ in range(self.max_len - len(input_sent))]
        token_type_ids = token_type_ids + [0 for _ in range(self.max_len - len(token_type_ids))]
        # print(input_sent)
        attn_mask = [1 if tok != '[PAD]' else 0 for tok in input_sent]
        ids = self.tokenizer.convert_tokens_to_ids(input_sent)
        # print(len(ids))
        return {
            'index' : idx,
            'ids' : torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(attn_mask, dtype=torch.long),
            'token_type_ids' : torch.tensor(token_type_ids, dtype = torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }

def load_paws(file_path,tokenizer):
    sentence1_list = []
    sentence2_list = []
    target_label_list = []

    dataframe = pd.read_csv(file_path, delimiter='\t')
    sentence1_list = dataframe['sentence1'].tolist()
    sentence2_list = dataframe['sentence2'].tolist()
    target_label_list = dataframe['label'].tolist()

    path = file_path.split('.')
    np.savetxt('.' + path[-2] + '_groundtruth.txt', target_label_list, '%s')
    print(sentence1_list[0])
    print(len(sentence1_list))
    print(sentence2_list[0])
    print(len(sentence2_list))
    data = paws_dataset(sentence1_list, sentence2_list, target_label_list, tokenizer, MAX_LEN)
    # print(data[0])
    return data