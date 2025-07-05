import torch
from tqdm import tqdm
import time
import pickle
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
import os
import gc
import argparse
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig, AutoModelForTokenClassification, AutoModel, BertPreTrainedModel
from torch import cuda
from torch.utils.data import ConcatDataset
from data_loader_new import load_qqp, load_paws
from test import inference
from torch.utils.data import random_split

MAX_LEN = 512
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
input_path = './'
num_labels = 2


def read_dataset(data_path):
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data
    
def read_file(filename):
    lines = []
    with open(filename, 'r') as file:
        for line in file:
            key = line.rstrip()
            lines.append(int(key))
    return lines
    
class LossFunction(nn.Module):
    def forward(self, probability):
        loss = torch.log(probability)
        loss = -1 * loss
        # print(loss)
        loss = loss.mean()
        # print(loss)
        return loss



class MainModel(BertPreTrainedModel):
    def __init__(self, config, loss_fn = None):
        super(MainModel,self).__init__(config)
        self.num_labels = 2
        self.loss_fn = loss_fn
        config.output_hidden_states = True
        self.bert = AutoModel.from_pretrained("QQP_MODEL",config = config)
        self.classifier = nn.Linear(768, self.num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids, labels,device):
              
        output = self.bert(input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
        output = output.last_hidden_state
        output = output[:,0,:]
        classifier_out = self.classifier(output)
        main_prob = F.softmax(classifier_out, dim = 1)
        main_gold_prob = torch.gather(main_prob, 1, labels)
        loss_main = self.loss_fn.forward(main_gold_prob)
        return loss_main,main_prob
        
def train(model, dataloader, optimizer, device):
    tr_loss, tr_accuracy = 0, 0
    bias_loss = 0
    # tr_preds, tr_labels = [], []
    nb_tr_steps = 0
    #put model in training mode
    model.train()
    
    for idx, batch in enumerate(dataloader):
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)

        loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets,device = device)

        # print(f'\tLoss Main : {loss_main}')
        tr_loss += loss_main.item()
        nb_tr_steps += 1
        #compute training accuracy
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_tr_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
        if idx % 100 == 0:
            print(f'\tModel loss at {idx} steps: {tr_loss}')
            if idx != 0:
                print(f'\tModel Accuracy : {tr_accuracy/nb_tr_steps}')
            with open('live.txt', 'a') as fh:
                fh.write(f'\tModel Loss at {idx} steps : {tr_loss}\n')
                if idx != 0:
                    fh.write(f'\tModel Accuracy : {tr_accuracy/nb_tr_steps}')
        optimizer.zero_grad()
        loss_main.backward()
        optimizer.step()

    print(f'\tModel loss for the epoch: {tr_loss}')
    print(f'\tTraining accuracy for epoch: {tr_accuracy/nb_tr_steps}')
    with open('live.txt', 'a') as fh:
        fh.write(f'\tTraining Accuracy : {tr_accuracy/nb_tr_steps}\n')


def valid(model, dataloader, device):
    eval_loss = 0
    bias_loss = 0
    eval_accuracy = 0
    model.eval()
    nb_eval_steps = 0
    for batch in dataloader:
        indexes = batch['index']
        input_ids = batch['ids'].to(device, dtype=torch.long)
        mask = batch['mask'].to(device, dtype=torch.long)
        targets = batch['target'].to(device, dtype=torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)

        loss_main, main_prob = model(input_ids=input_ids, attention_mask=mask, token_type_ids = token_type_ids, labels=targets,device = device)
        eval_loss += loss_main.item()
        nb_eval_steps += 1
        #compute training accuracy
        predicted_labels = torch.argmax(main_prob, dim=1)
        # print(predicted_labels.shape)
        targets = targets.view(-1)
        # print(targets.shape)
        tmp_eval_accuracy = accuracy_score(targets.cpu().numpy(), predicted_labels.cpu().numpy())
        eval_accuracy += tmp_eval_accuracy
        
    
    return eval_loss, eval_accuracy/nb_eval_steps 

def save_dataset(data, data_path):
    with open(data_path, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
    return
    
def main():
    gc.collect()
    
    torch.cuda.empty_cache()
    print("Training model :")
    start = time.time()
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--m1_model_path', type=str, required=True)
    parser.add_argument('--output_model_directory', type=str, required=True)
    parser.add_argument('--output_tokenizer_directory', type=str, required=True)
    parser.add_argument('--train_sample_percent', type=int, required=True)
    parser.add_argument('--id_ood', type=str, required=True)
    
    args = parser.parse_args()

    output_model_path = os.path.join(input_path, args.output_model_directory)
    output_tokenizer_path = os.path.join(input_path, args.output_tokenizer_directory)
 
    best_output_model_path = output_model_path + '/BestModel'    
    if not os.path.exists(output_model_path):
        os.makedirs(output_model_path)
    if not os.path.exists(best_output_model_path):
        os.makedirs(best_output_model_path)

    #train_file_path = os.path.join(input_path, args.dataset_name,'questions.csv')
    #dev_file_path= os.path.join(input_path,'PAWS/dev.tsv')

    
    id_ood = read_file(args.id_ood)
    tokenizer = AutoTokenizer.from_pretrained("QQP_MODEL")
    data1 = read_dataset('./QQP/train.pkl')
    
    print(f'QQP train length : {len(data1)}')
    
    data1_sample_size = int((args.train_sample_percent/100)*len(data1))
    data1,_ = random_split(data1, [data1_sample_size, len(data1) - data1_sample_size])
    
    data2_id, data2_ood = load_paws(file_path='./PAWS/train.tsv', tokenizer=tokenizer, id_ood=id_ood)
    
    
    print(f'PAWS ID train length : {data2_id.len}')
    print(f'PAWS OOD train length : {data2_ood.len}')
    
    data2_id_sample_size = int((args.train_sample_percent/100)*data2_id.len)
    
    data2_id,_ = random_split(data2_id, [data2_id_sample_size, data2_id.len - data2_id_sample_size])
    train_data = ConcatDataset([data1,data2_id, data2_ood])
    print(f'Train dataset length : {len(train_data)}')
    save_dataset(train_data, './train.pkl')
    
    
    
    dev_data1 = load_paws(file_path='./PAWS/dev.tsv', tokenizer=tokenizer)
    dev_data2 = read_dataset('./QQP/val.pkl')
    dev_data = ConcatDataset([dev_data1, dev_data2])
    
    print(len(train_data))
    print(len(dev_data))
    model = MainModel.from_pretrained("QQP_MODEL", num_labels = 2, loss_fn = LossFunction())
    device = 'cuda' if cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    #val_size = int(0.1 * len(data))
    #train_size = len(data) - val_size
    #data, dev_data = random_split(data, [train_size, val_size])

    #save_dataset(data, './QQP/train.pkl')
    #save_dataset(dev_data, './QQP/val.pkl')
    #save_dataset(dev_data, './QQP/val.pkl')
    ## eval_data = ConcatDataset([dev_data1, dev_data2])
    ## print(len(eval_data))
    #paws_data = load_paws(file_path='./PAWS/test.tsv', tokenizer=tokenizer)
    #paws_dataloader = DataLoader(paws_data, shuffle = True, batch_size=BATCH_SIZE)
    train_dataloader = DataLoader(train_data, shuffle = True, batch_size=BATCH_SIZE)
    eval_dataloader = DataLoader(dev_data, shuffle=True, batch_size=BATCH_SIZE)

    
    num_epochs = 20
    max_acc = 0.0
    patience = 0
    max_hans_acc = 0.0
    max_mnli_mm_acc = 0.0
    best_model = model
    best_tokenizer = tokenizer
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}:')
        with open('live.txt', 'a') as fh:
            fh.write(f'Epoch : {epoch+1}\n')
        train(model, train_dataloader, optimizer, device)
        validation_loss, eval_acc = valid(model, eval_dataloader, device)
        #paws_acc,_,_ = inference(model, paws_dataloader, tokenizer, device)
        print(f'\tValidation loss: {validation_loss}')
        print(f'\tValidation accuracy for epoch: {eval_acc}')
        #print(f'\tPAWS accuracy: {paws_acc}')
        with open('live.txt', 'a') as fh:
            fh.write(f'\tValidation Loss : {validation_loss}\n')
            fh.write(f'\tValidation accuracy for epoch: {eval_acc}\n')
            #fh.write(f'\tPAWS accuracy: {paws_acc}\n')
        
        if eval_acc > max_acc:
            max_acc = eval_acc
            # model.save_pretrained(best_hans_model_path)
            # tokenizer.save_pretrained(best_hans_model_path)
            patience = 0
            best_model = model
            best_tokenizer = tokenizer
            best_model.save_pretrained(best_output_model_path)
            best_tokenizer.save_pretrained(best_output_model_path)
        else:
            patience += 1
            if patience > 3:
                print("Early stopping at epoch : ",epoch)
                best_model.save_pretrained(best_output_model_path)
                best_tokenizer.save_pretrained(best_output_model_path)
                patience = 0
                break
            
        model.save_pretrained(output_model_path)
        tokenizer.save_pretrained(output_tokenizer_path)
    
    model.save_pretrained(output_model_path)
    tokenizer.save_pretrained(output_tokenizer_path)

    best_model.save_pretrained(best_output_model_path)
    best_tokenizer.save_pretrained(best_output_model_path)

    end = time.time()
    total_time = end - start
    with open('live.txt', 'a') as fh:
        fh.write(f'Total training time : {total_time}\n')

    print(f"Total training time : {total_time}")
if __name__ == '__main__':
    main()
