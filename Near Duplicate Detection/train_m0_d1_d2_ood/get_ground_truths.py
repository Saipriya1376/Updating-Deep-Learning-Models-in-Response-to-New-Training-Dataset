import pickle
from torch.utils.data import random_split
import numpy as np


def read_dataset(data_path):
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data
    
def save_dataset(data, data_path):
    with open(data_path, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
    return
    
dataset = read_dataset('./QQP/train.pkl')
labels = []
for idx in range(len(dataset)):
    labels.append(dataset[idx]['target'].item())
    
np.savetxt('./QQP/qqp_train_groundtruth.txt', labels, '%s')
