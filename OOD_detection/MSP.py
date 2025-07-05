import argparse
import numpy as np


def select_threshold_func(tensor, percent):
    tensor = np.sort(tensor)[::-1]
    print(tensor)
    percent = percent/100
    n = len(tensor)
    percent = n*percent
    percent = int(percent)
    threshold = tensor[percent]
    return threshold



def main():


    parser = argparse.ArgumentParser(description="Separate softmax scores of correct and incorrect data points and find the average maximum softmax score for each partition.")

    parser.add_argument("--val_softmax_scores", type=str, help="Train dataset softmax score")
    parser.add_argument("--test_softmax_scores", type=str, help="File containing test dataset softmax scores")
    parser.add_argument('--pred_dir', type=str, required=True)
    parser.add_argument('--percent', type=float, required=True)
    
    
    args = parser.parse_args()
    

    val_softmax_scores = np.loadtxt(args.val_softmax_scores, dtype=float)
    test_softmax_scores = np.loadtxt(args.test_softmax_scores, dtype=float)

    test_score = np.max(test_softmax_scores, axis=1)

    val_score = np.max(val_softmax_scores, axis = 1)
    threshold = select_threshold_func(val_score, args.percent)
    print(f'Threshold : {threshold}')

    true_ind = []
    cnt = 0
    for i in range(len(test_score)):
        if test_score[i] < threshold:
            true_ind.append(1)
            cnt += 1
        else:
            true_ind.append(0)


    print(f'Total number of data points : {len(test_score)}')
    print(f'Number of OOD data points: {cnt}')
    
    # Save the array with Pickle
    filename = args.pred_dir + '/ood.txt'
    np.savetxt(filename, true_ind, fmt='%i')

if __name__ == "__main__":
    main()
