#!/usr/bin/env python
# coding: utf-8
from ml_toolbox.utils.feature import BagOfWords, BagOfWordsTrim
import numpy as np
import sys
from collections import defaultdict
# import jax.numpy as jnp


class LogisticRegression():
    def __init__(self,  train_formatted_output, 
                        test_formatted_output, 
                        valid_formatted_output, 
                        input_dict_filename,
                        train_file, 
                        test_file, 
                        metrics_out,
                        lr = 0.1,
                        epochs = 30):


        self.epochs = epochs
        self.lr = lr
        self.input_dict = self.read_dict(input_dict_filename)
        self.inv_dict = {v: k for k, v in self.input_dict.items()}
        self.train_features, self.train_labels = self.read_input(train_formatted_output)
        self.valid_features, self.valid_labels = self.read_input(valid_formatted_output)
        self.test_features, self.test_labels = self.read_input(test_formatted_output)
        self.train_out = train_file
        self.test_out = test_file
        self.metrics_out = metrics_out        
        self.sigmoid = lambda x: np.exp(x)/(1 + np.exp(x)) 

    def return_feature_idxs(self, features, vocab):
        idxs = []
        for f in features:
            idxs.append(vocab[f]) 
        features = np.zeros(len(vocab))
        features[idxs] = 1.0
        return features, idxs

    def read_input(self, filename):
        features = []
        labels = []
        with open(filename) as f:
            for l in f:
                data = defaultdict(list)
                labels.append(l[0])
                for val in l[1:].split('\t'):
                    if val: 
                        word = self.inv_dict[int(val.split(':')[0])]
                        data[word] = labels[-1]
                features.append(data)
        return features, labels
    
            
    def read_dict(self, filename):
        dict = {}
        with open(filename, 'r') as f:
            for line in f:
                split = line.split(' ')
                dict[split[0]] = int(split[1].split('\n')[0])
        return dict
            

    def train(self, X_train, Y_train, vocab, epochs, lr):
        
        weights = np.zeros(len(vocab))
        bias = 0
        for _ in range(epochs):
            for x_train, y_train in zip(X_train, Y_train):
                y_dot = self.sparse_product(weights, x_train) + bias
                features, _ = self.return_feature_idxs(x_train, vocab)
                gradient = (float(y_train) - float(self.sigmoid(y_dot))) / len(X_train)
                gradient_w = gradient * features 
                weights = weights + lr * gradient_w
                bias = bias + lr * gradient
        return weights, bias

    def sparse_product(self, weights, features):
        res = 0.0
        for f in features:
            index = self.input_dict[f]
            res += weights[index]
        return res

    def return_feature_idxs(self, features, vocab):
        idxs = []
        for f in features:
            idxs.append(vocab[f]) 
        features = np.zeros(len(vocab))
        features[idxs] = 1.0
        return features, idxs

    def predict(self, X_test, weights, bias):
        y_preds = []
        for x_test in X_test:
            y_dot = self.sparse_product(weights, x_test) + bias
            y_pred = 1 if self.sigmoid(y_dot) >= 0.5 else 0
            y_preds.append(str(y_pred))
        return y_preds

    def evaluate(self, X_test, Y_test, weights, bias):
        Y_preds = np.array(self.predict(X_test, weights, bias))
        err = len(np.where(np.array(Y_preds == Y_test) == False)[0])/len(Y_test)
        return err
    
    def output_label_file(self, y_preds, out_filename):
        with open(out_filename, 'w') as f:
            for y_pred in y_preds:
                print(y_pred, file = f)
    
    def run_survey(self):

        weights, bias = self.train(self.train_features, 
                            self.train_labels, 
                            self.input_dict, 
                            self.epochs, self.lr)

        self.test_err = self.evaluate(self.test_features, 
                                self.test_labels, 
                                weights, bias)
        
        self.train_err = self.evaluate(self.train_features, 
                                self.train_labels, 
                                weights, bias)

        self.valid_err =self.evaluate(self.valid_features, 
                                self.valid_labels, 
                                weights, bias)
        
        y_preds = self.predict(self.train_features, weights, bias)
        self.output_label_file(y_preds, self.train_out)

        y_preds = self.predict(self.test_features, weights, bias)
        self.output_label_file(y_preds, self.test_out)
           
        with open(self.metrics_out, 'w') as f:
            print(f'error(train): {self.train_err}', file = f)
            print(f'error(test): {self.test_err}', file = f)

    def calculate_log_likelihood(self, x, y, weights, bias):
        loss = 0.0
        for feature, label in zip(x, y):
            product = self.sparse_product(weights, feature)
            loss += (-float(label) * product + np.log(1 + np.exp(product)))
        return loss/len(x)


    def plot_loss(self):
        train_loss = []
        valid_loss = []
        weights = np.zeros(len(self.input_dict.keys()))
        bias = 0
        for _ in range(epochs):
            for x_train, y_train in zip(self.train_features, self.train_labels):
                y_dot = self.sparse_product(weights, x_train) + bias
                features, _ = self.return_feature_idxs(x_train, self.input_dict)
                gradient = (float(y_train) - float(self.sigmoid(y_dot))) / len(self.train_labels)
                gradient_w = gradient * features 
                weights = weights + self.lr * gradient_w
                bias = bias + self.lr * gradient
            train_loss.append(self.calculate_log_likelihood(self.train_features, self.train_labels, weights, bias))
            valid_loss.append(self.calculate_log_likelihood(self.valid_features, self.valid_labels, weights, bias))
        x = range(0,epochs)
        plt.plot(x, train_loss, "r", linewidth = 1.0, label = "Training loss")
        plt.plot(x, valid_loss, "g", linewidth = 1.0, label = "Validation loss")
        plt.title("Evaluation of losses")
        plt.xlabel('epochs')
        plt.ylabel('log likelihood')
        plt.legend()
        plt.show()
    
    def plot_loss_lrs(self):
        for lr in [0.1, 0.0001, 0.5]:
            train_loss = []
            valid_loss = []
            weights = np.zeros(len(self.input_dict.keys()))
            bias = 0
            losses = []
            for _ in range(epochs):
                for x_train, y_train in zip(self.train_features, self.train_labels):
                    y_dot = self.sparse_product(weights, x_train) + bias
                    features, _ = self.return_feature_idxs(x_train, self.input_dict)
                    gradient = (float(y_train) - float(self.sigmoid(y_dot))) / len(self.train_labels)
                    gradient_w = gradient * features 
                    weights = weights + lr * gradient_w
                    bias = bias + lr * gradient
                train_loss.append(self.calculate_log_likelihood(self.train_features, self.train_labels, weights, bias))
                
            x = range(0,epochs)
            plt.plot(x, train_loss, linewidth = 1.0, label = str(lr))
        plt.title("Evaluation on different learning rates")
        plt.xlabel('epochs')
        plt.ylabel('log likelihood')
        plt.legend()
        plt.show()


if __name__ == '__main__':

    formatted_train_out = 'formatted_train.tsv'
    formatted_validation_out = 'formatted_valid.tsv'
    formatted_test_out = 'formatted_test.tsv'
    input_dict_filename = 'handout/dict.txt'
    train_out = 'train_out.labels'
    test_out = 'test_out.labels'
    metrics_out = 'metrics_out.txt'
    epochs = 30
    
    logistic_regression = LogisticRegression(formatted_train_out, 
                                            formatted_test_out, 
                                            formatted_validation_out, 
                                            input_dict_filename,
                                            train_out, 
                                            test_out, 
                                            metrics_out,
                                            lr = 0.1, epochs = epochs)

    
    logistic_regression.run_survey()
    #logistic_regression.plot_loss()
    #logistic_regression.plot_loss_lrs()


