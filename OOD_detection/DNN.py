import numpy as np
import argparse
import torch
import faiss
import os
# from util import metrics

def read_data_func(dataset_name):
    tensor = np.loadtxt(dataset_name)
    return tensor

def normalize_embedding_func(tensor):
    # Calculate the sum of each row
    row_sq = np.square(tensor)
    row_sums = np.sum(row_sq, axis=1, keepdims=True)
    row_sums_sqrt = np.sqrt(row_sums)

    # Divide each element by its row's sum
    result_tensor = tensor / row_sums_sqrt

    # Saving the normalized embedding for future use
    # np.savetxt(f"{args.input_file[0:-4]}normalized.txt", result_tensor)
    return result_tensor

def find_distance_func(tensor1, tensor2):
    # getting length of the tensor
    n = len(tensor1)
    m = len(tensor2)
    distance_matrix = []
    
    # for every row we calculate the distance to every row
    for i in range(n):
        distance_row = []
        for j in range(m):
            distance_tensor = np.subtract(tensor1[i], tensor2[j])
            distance_tensor_sq = np.square(distance_tensor)
            dist_square = np.sum(distance_tensor_sq, axis=0)
            distance = np.sqrt(dist_square)
            distance_row.append(distance)
        distance_matrix.append(distance_row)
    
    distance_matrix = np.array(distance_matrix)
    return distance_matrix

def sort_distance_matrix_func(distance_matrix):
    distance_matrix.sort(axis=1)
    return distance_matrix

def pick_kth_value_func(distance_matrix, k):
    
    # picks the kth column from 2-d matrix
    tensor = distance_matrix[:, k]

    return tensor

def select_threshold_func(tensor, percent):
    tensor = np.sort(tensor)
    percent = percent/100
    n = len(tensor)
    percent = n*percent
    percent = int(percent)
    threshold = tensor[percent]
    return threshold

def get_ood_count_func(tensor, th, val):
    # Create a boolean mask for elements greater than 'k'
    mask = tensor > th

    # Count the number of elements greater than 'k' in each row
    count_per_row = np.sum(mask, axis=1)
    count_per_row = count_per_row/val

    return count_per_row


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file_train', type=str, required=True)
    parser.add_argument('--input_file_val', type=str, required=True)
    parser.add_argument('--input_file2', type=str, required=True)
    parser.add_argument('--test_dataset_name', type=str, required=True)
    parser.add_argument('--percent', type=float, required=True)
     


    args = parser.parse_args()

    # loading the embeddings 
    tensor1_train = read_data_func(args.input_file_train)
    tensor1_val = read_data_func(args.input_file_val)
    tensor2 = read_data_func(args.input_file2)
    print("tensor1")
    print(tensor1_train.shape)
    print()
    print("tensor2")
    print(tensor2.shape)
    print()
    
    # normalize the embeddings
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    tensor1_train = normalize_embedding_func(tensor1_train)
    tensor1_val = normalize_embedding_func(tensor1_val)
    tensor2 = normalize_embedding_func(tensor2)
    # tensor1 = np.concatenate(tensor1_train,tensor1_val, axis = 0)
    print("tensor1")
    print(tensor1_train)
    print()
    print("tensor2")
    print(tensor2)
    print()
    k_values=[10,20,50,100,200,500,800,1000]

    all_score_ood = []
    all_results = []
    print(f'tensor1 shape {tensor1_train.shape}')
    print(f'tensor1 shape {tensor1_train.shape[1]}')
    index = faiss.IndexFlatL2(tensor1_train.shape[1])
    index.add(tensor1_train)
    distance, _ = index.search(tensor1_val, 1000)
    test_distance, _ = index.search(tensor2, 1000)

    for k in k_values:
        result_dir = './results_DNN/' + '/' + args.test_dataset_name
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        print(f'k = {k}')
        kth_values = distance[:,k-1]
        print(f'Validation kth values shape: {kth_values.shape}')
        kth_values = list(kth_values)
        
        print(f'Max cosine similarity value : {max(kth_values)}')
        print(f'Min cosine similarity value : {min(kth_values)}')
        threshold = select_threshold_func(kth_values, args.percent)
        print(f"threshold {threshold}")
        kth_values_test = test_distance[:,k-1]
        print(f'Test kth values shape: {kth_values_test.shape}')
        kth_values_test = list(kth_values_test)
        
        cnt = 0
        true_ind = []
        for i in range(len(kth_values_test)):
            if kth_values_test[i] > threshold:
                true_ind.append(1)
                cnt += 1
            else:
                true_ind.append(0)



        print(f'Number of OOD data points: {cnt}')
        
        # Save the array with Pickle
        filename = result_dir + '/k_' + str(k) + '.txt'
        np.savetxt(filename, true_ind, fmt='%i')


if __name__ == '__main__':
    main()
