#!/usr/bin/env python
# coding: utf-8


import numpy as np
import csv
import math
import sys


class NaiveBayes:
    def __init__(self, train_input, test_input, num_voxels):
        
        self.num_voxels = num_voxels
        
        self.x_train, self.y_train = read_csv(train_input)
        self.x_test, self.y_test = read_csv(test_input)
        self.classes =  np.unique(self.y_train)
        self.pdf = lambda x, mean, var:(np.exp(-np.power(x - mean, 2.) / (2 * np.power(var, 2.)))) \
                        / (var * np.power(2 * math.pi, 1/2))

        self.mean = {}
        self.var = {}
        self.std = {}
        self.class_freq = {}

    def slice_data(self, slice_idx):
        self.x_train = self.x_train[:slice_idx]
        self.y_train = self.y_train[:slice_idx]


    def train(self):

        for c in np.unique(self.y_train):
            
            idxs = np.where(self.y_train == c)[0]
            self.class_freq[c] = len(idxs)/self.y_train.shape[0]
            subset_x = self.x_train[idxs, :]
            
            self.mean[c], self.var[c], self.std[c]  = np.mean(subset_x, axis = 0), np.var(subset_x,  axis = 0), np.std(subset_x, axis = 0)

        self.top_idxs = self.select_k_features(self.y_train, self.num_voxels)

    def calculate_probability(self, x, mean, stdev):
        exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
        return ((1 / (math.sqrt(2 * math.pi) * stdev)) * exponent)

    def select_k_features(self, y_train, k):
        class_prob = {cls:np.log(self.class_freq[cls]) for cls in self.classes}
        diff = [0] * len(self.mean['tool'])
        for i in range(len(self.mean['tool'])):
            diff[i] = np.abs(self.mean[self.classes[0]][i] - self.mean[self.classes[1]][i])
        
        diff = np.array(diff)
        top_idxs = np.argsort(np.array(diff))
        top_idxs = top_idxs[-int(k):]
        return top_idxs


    def predict(self, x):
        
        class_prob = {cls:np.log(self.class_freq[cls]) for cls in self.classes}
        for cls in self.classes:        
            for i in self.top_idxs:
                class_prob[cls] += np.log(self.calculate_probability(x[i], self.mean[cls][i], self.std[cls][i]))
            class_prob[cls] = class_prob[cls]/len(self.top_idxs)
        class_prob = {cls:np.exp(class_prob[cls]) for cls in class_prob}
        return class_prob


    def predict_class(self, X):
        pred = []
        for x in X:
            pred_class = None
            max_prob = -float('inf')
            for cls, prob in self.predict(x).items():
                if prob>max_prob:
                    max_prob = prob
                    pred_class = cls
            pred.append(pred_class)
        return pred



def read_csv(filename: str) -> np.ndarray:

    with open(filename) as csv_file:

        csv_reader = csv.reader(csv_file, delimiter=',')

        x = []
        y = []
        for line_count, row in enumerate(csv_reader):
            if line_count == 0:
                features = len(row) - 1
            else:
                x.append(row[:features])
                y.append(row[-1])        

        # print(f'Processed {line_count} lines.')
        return np.array(x).astype(np.float), np.array(y)

def output_labels(arr, filename):
    with open(filename, 'w') as f:
        for a in arr:
            print(a, file = f)

def calculate_error(y_preds, y_gt):
    return len(np.where(np.array(y_preds != y_gt))[0])/len(y_preds)


def write_metrics(train_err, test_err, metrics_out):
    with open(metrics_out, 'w') as f:
        test_err = '{:.6f}'.format(test_err)
        train_err = '{:.6f}'.format(train_err)
        print(f'error(train) : {train_err}', file = f)
        print(f'error(test) : {test_err}', file = f)

if __name__ == '__main__':

    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_voxels = sys.argv[6]

    # nb = NaiveBayes(train_input, test_input, num_voxels)
    # nb.train()

    # y_preds = nb.predict_class(nb.x_train)
    # output_labels(y_preds, train_out)
    # train_err = calculate_error(y_preds, nb.y_train)
    
    # y_preds = nb.predict_class(nb.x_test)
    # output_labels(y_preds, test_out)
    # test_err = calculate_error(y_preds, nb.y_test)
    # write_metrics(train_err, test_err, metrics_out)
    
    test_errs = []
    for i in range(10, 42,1): 
        nb = NaiveBayes(train_input, test_input, num_voxels)
        nb.slice_data(i)
        nb.train()
        y_preds = nb.predict_class(nb.x_test)
        test_err = calculate_error(y_preds, nb.y_test)
        test_errs.append(test_err)
    
    k_test_errs = []
    for i in range(50, 21650, 50): 
        nb = NaiveBayes(train_input, test_input, i)
        nb.train()
        y_preds = nb.predict_class(nb.x_test)
        test_err = calculate_error(y_preds, nb.y_test)
        k_test_errs.append(test_err)
    
    np.savez('k_test_errors.npz',  k_test_errs = np.array(k_test_errs), test_errs = test_errs)
        