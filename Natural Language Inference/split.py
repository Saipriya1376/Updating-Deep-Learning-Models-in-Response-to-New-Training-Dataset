import pickle
from torch.utils.data import random_split


def read_dataset(data_path):
    with open(data_path, 'rb') as inp:
        data = pickle.load(inp)
    return data
    
def save_dataset(data, data_path):
    with open(data_path, 'wb') as outp:
        pickle.dump(data, outp, pickle.HIGHEST_PROTOCOL)
    return
    
old_train_data = read_dataset('./QQP/train.pkl')
val_size = int(0.1 * len(old_train_data))
train_size = len(old_train_data) - val_size
train_data, test_data = random_split(old_train_data, [train_size, val_size])
save_dataset(train_data, './QQP/new_train.pkl')
save_dataset(test_data, './QQP/new_test.pkl')
