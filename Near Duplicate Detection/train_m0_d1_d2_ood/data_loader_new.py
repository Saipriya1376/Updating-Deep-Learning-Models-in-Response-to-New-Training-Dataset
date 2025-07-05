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
import csv
import gensim

def read_dataset(data_path):
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data
    
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

def load_qqp(file_path,tokenizer, id_ood=None):
    sentence1_list = []
    sentence2_list = []
    target_label_list = []

    dataframe = pd.read_csv(file_path)
    sentence1_list = dataframe['question1'].tolist()
    sentence2_list = dataframe['question2'].tolist()
    target_label_list = dataframe['is_duplicate'].tolist()

    sentence1_list_id = []
    sentence1_list_ood = []
    sentence2_list_id = []
    sentence2_list_ood = []
    target_label_list_id = []
    target_label_list_ood = []

    """
    print(sentence1_list[0])
    print(len(sentence1_list))
    print(sentence2_list[0])
    print(len(sentence2_list))
    """
    if id_ood == None: 
     
    	path = file_path.split('.')
    	#np.savetxt('./QQP/qqp_groundtruth.txt', target_label_list, '%s')
    	data = qqp_dataset(sentence1_list, sentence2_list, target_label_list, tokenizer, MAX_LEN)
    	return data
    else:
        assert(len(id_ood) == len(sentence1_list))
        for i,label in enumerate(id_ood):
            if label == 0:
                sentence1_list_id.append(sentence1_list[i])
                sentence2_list_id.append(sentence2_list[i])
                target_label_list_id.append(target_label_list[i])
            else:
                sentence1_list_ood.append(sentence1_list[i])
                sentence2_list_ood.append(sentence2_list[i])
                target_label_list_ood.append(target_label_list[i])
                
        data_id = qqp_dataset(sentence1_list_id, sentence2_list_id, target_label_list_id, tokenizer, MAX_LEN)
        data_ood = qqp_dataset(sentence1_list_ood, sentence2_list_ood, target_label_list_ood, tokenizer, MAX_LEN)
        return data_id, data_ood

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

def load_paws(file_path,tokenizer, id_ood=None):
    sentence1_list = []
    sentence2_list = []
    target_label_list = []

    dataframe = pd.read_csv(file_path, delimiter='\t')
    sentence1_list = dataframe['sentence1'].tolist()
    sentence2_list = dataframe['sentence2'].tolist()
    target_label_list = dataframe['label'].tolist()
    
    sentence1_list_id = []
    sentence1_list_ood = []
    sentence2_list_id = []
    sentence2_list_ood = []
    target_label_list_id = []
    target_label_list_ood = []

    if id_ood == None:
        path = file_path.split('.')
        #np.savetxt('.' + path[-2] + '_groundtruth.txt', target_label_list, '%s')
        data = paws_dataset(sentence1_list, sentence2_list, target_label_list, tokenizer, MAX_LEN)
        # print(data[0])
        return data
    else: 
        assert(len(id_ood) == len(sentence1_list))
        for i,label in enumerate(id_ood):
            if label == 0:
                sentence1_list_id.append(sentence1_list[i])
                sentence2_list_id.append(sentence2_list[i])
                target_label_list_id.append(target_label_list[i])
            else:
                sentence1_list_ood.append(sentence1_list[i])
                sentence2_list_ood.append(sentence2_list[i])
                target_label_list_ood.append(target_label_list[i])
        data_id = paws_dataset(sentence1_list_id, sentence2_list_id, target_label_list_id, tokenizer, MAX_LEN)
        data_ood = paws_dataset(sentence1_list_ood, sentence2_list_ood, target_label_list_ood, tokenizer, MAX_LEN)
        return data_id, data_ood
        
def load_mrpc(file_path,tokenizer, id_ood=None):
    sentence1_list = []
    sentence2_list = []
    target_label_list = []

    Train_Data_File, Test_Data_File = file_path


    df1 = pd.read_csv(Train_Data_File, sep='\t', quoting=csv.QUOTE_NONE)
    df2 = pd.read_csv(Test_Data_File, sep='\t', quoting=csv.QUOTE_NONE)
    cols = ['isSimilar', 'SentenceID1', 'SentenceID2', 'Sentence1', 'Sentence2']
    df1.columns = cols
    df2.columns = cols

    for dataset in [df1, df2]:
        for i, row in dataset.iterrows():
            sentence1_list.append(row['Sentence1'])
            sentence2_list.append(row['Sentence2'])
            target_label_list.append(row['isSimilar'])
    
    sentence1_list_id = []
    sentence1_list_ood = []
    sentence2_list_id = []
    sentence2_list_ood = []
    target_label_list_id = []
    target_label_list_ood = []

    if id_ood == None:
        #path = file_path.split('.')
        #np.savetxt('.' + path[-2] + '_groundtruth.txt', target_label_list, '%s')
        data = paws_dataset(sentence1_list, sentence2_list, target_label_list, tokenizer, MAX_LEN)
        # print(data[0])
        return data
    else: 
        assert(len(id_ood) == len(sentence1_list))
        for i,label in enumerate(id_ood):
            if label == 0:
                sentence1_list_id.append(sentence1_list[i])
                sentence2_list_id.append(sentence2_list[i])
                target_label_list_id.append(target_label_list[i])
            else:
                sentence1_list_ood.append(sentence1_list[i])
                sentence2_list_ood.append(sentence2_list[i])
                target_label_list_ood.append(target_label_list[i])
        data_id = paws_dataset(sentence1_list_id, sentence2_list_id, target_label_list_id, tokenizer, MAX_LEN)
        data_ood = paws_dataset(sentence1_list_ood, sentence2_list_ood, target_label_list_ood, tokenizer, MAX_LEN)
        return data_id, data_ood
