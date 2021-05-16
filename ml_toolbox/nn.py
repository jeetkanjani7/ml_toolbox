
import numpy as np
import sys
from collections import defaultdict
from dataclasses import dataclass
from ml_toolbox.utils.data import read_csv_data
import csv

class NeuralNetwork:

    def __init__(self, X_train: np.ndarray, 
                        Y_train: np.ndarray, 
                        learning_rate: float,
                        num_classes: int,
                        hidden_units: int,
                        epochs: int,
                        flag: int = 1):  
        
        
        self.X_train = self.append_bias(X_train)
        self.Y_train = Y_train
        
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.epochs = epochs
        self.hidden_units = hidden_units
        self.num_features = self.X_train.shape[1]
        
        self.num_samples = self.X_train.shape[0]
        self.initialize_weights(flag)
        self.flag = flag
        self.sigmoid = lambda x: 1/(1 + np.exp(-x))
        
    def append_bias(self, X):
        biases = np.expand_dims(np.ones(X.shape[0]), axis = -1)
        X = np.hstack( (biases, X))
        return X

    def initialize_weights(self, flag):        

        if flag == 1:
            self.alpha = np.random.uniform(-0.1, 0.1, 
                            (self.hidden_units, self.num_features))
            
            self.alpha[:, 0] = 1
            self.beta = np.random.uniform(-0.1, 0.1, \
                                (self.num_classes, self.hidden_units + 1 ))
            self.beta[:, 0] = 1

        elif flag == 2:
            self.alpha = np.zeros((self.hidden_units, self.num_features))
            self.beta = np.zeros((self.num_classes, self.hidden_units + 1 ))
            
        else:
            raise Exception("input valid initialization flag")
        
        
    def forward_pass_single(self, idx):
        # z1 has dimension of [number of hidden x num of samples]
        self.z1 = self.sigmoid(np.dot(self.alpha, self.X_train[idx, :].T))

        # z1_bias has dimension [(number of hidden + 1) x num of samples]
        self.z1_bias = np.insert(self.z1, 0, 1)
        
        #self.z1_bias = np.vstack((np.expand_dims(np.ones((1,1)),\
        #                         axis = 0), np.expand_dims(self.z1, axis = 0)))
        
        # z2 has dimension [number of classes x num of samples]
        self.z2 = np.dot(self.beta, self.z1_bias)

        # probs has dimension [number of classes x num of samples]
        self.probs = np.exp(self.z2)/np.exp(self.z2).sum(axis = 0)
        # print(self.probs)

        #one hot encoding has dimension [number of samples x num of classes]
        self.one_hot_encoding = return_one_hot(self.Y_train,\
                                            self.num_samples, \
                                            self.num_classes)
        
        self.one_hot_encoding = self.one_hot_encoding[idx, :]

        cross_entropy_global = 0
        for y_hat, y in zip(self.probs.T, self.one_hot_encoding):
            cross_entropy_local = - np.sum(np.dot(y, np.log(y_hat)))
            cross_entropy_global += cross_entropy_local
        
        self.loss = cross_entropy_global/self.X_train.shape[0]

        return self.loss



    def backward_pass_single(self, idx):
        
        # print(self.probs.shape, self.one_hot_encoding.shape)
        # of dimension [number of classes x 1]
        dl_dz2 = np.expand_dims(self.probs - \
                            self.one_hot_encoding, axis = 1)
        
        # of dimension dl_dbeta will [num_classes x num_hidden + 1]
        dl_dbeta = np.dot(dl_dz2,  np.expand_dims(\
                            self.z1_bias, axis = 0)) 

        # dl_dz1 has dims [Dx1]
        dl_dz1 = np.dot(self.beta[:, 1:].T, dl_dz2)

        # dl_dalpha_b has dims [D x 1]
        dl_dalpha_b = np.multiply(dl_dz1, \
                    np.expand_dims(np.multiply(self.z1,\
                     1 - self.z1), axis = -1 ))
        
        
        # dl_dalpha has dims []
        dl_dalpha = np.dot(dl_dalpha_b, \
                        np.expand_dims(self.X_train[idx,:], axis = 0))
        
        
        # print(dl_dalpha.shape, self.alpha.shape, self.X_train.shape)
        
        self.alpha -= self.learning_rate * dl_dalpha
        self.beta -= self.learning_rate * dl_dbeta
        
        

    def forward_pass(self, X_input):
        # z1 has dimension of [number of hidden x num of samples]
        self.z1 = self.sigmoid(np.dot(self.alpha, X_input.T))
        
        # z1_bias has dimension [(number of hidden + 1) x num of samples]
        self.z1_bias = np.vstack((np.expand_dims(np.ones(self.num_samples),\
                                 axis = 0), self.z1))
        
        # z2 has dimension [number of classes x num of samples]
        self.z2 = np.dot(self.beta, self.z1_bias)

        # probs has dimension [number of classes x num of samples]
        self.probs = np.exp(self.z2)/np.exp(self.z2).sum(axis = 0)

        #one hot encoding has dimension [number of samples x num of classes]
        self.one_hot_encoding = return_one_hot(self.Y_train,\
                                            self.num_samples, \
                                            self.num_classes)
        
        

        self.loss = - np.sum(np.sum(np.multiply(self.one_hot_encoding.T, 
                                 np.log(self.probs)),  axis = 0))/self.num_samples

        return self.loss


    def calc_error(self, ground_truth, prediction):
        return len(np.where(np.array(ground_truth) != np.array(prediction))[0])/len(ground_truth)
        


    def predict(self):
        self.forward_pass(self.X_train)
        self.y_preds = np.argmax(self.probs, axis = 0)
        return self.alpha, self.beta, self.loss


    def train(self, X_valid, Y_valid, metrics_out):
        
        train_losses = []
        valid_losses = []
        with open(metrics_out, 'w') as f:
            for epoch in range(self.epochs):
            
                for idx in range(self.X_train.shape[0]):
                    self.forward_pass_single(idx)
                    self.backward_pass_single(idx)
                self.predict()
            
                print(f'epoch={epoch} crossentropy(validation):  {self.loss}', file =f)
                
                train_losses.append(self.loss)
                
                
                nn_valid = NeuralNetwork(X_valid, Y_valid, self.learning_rate,\
                                self.num_classes, self.hidden_units,\
                                self.epochs, self.flag)

                _, _, valid_loss = nn_valid.predict()
                valid_losses.append(valid_loss)
                alpha, beta, loss = self.alpha, self.beta, self.loss
                nn_valid = NeuralNetwork(X_valid, Y_valid, self.learning_rate, \
                            self.num_classes, self.hidden_units,\
                            self.epochs, self.flag)
                
                nn_valid.alpha = alpha
                nn_valid.beta = beta
                _, _, valid_loss = nn_valid.predict()
            
                print(f'epoch={epoch} crossentropy(validation):  {valid_loss}', file=f)      
            valid_losses.append(valid_loss)
    
            self.predict()
            self.train_err = self.calc_error(self.Y_train, self.y_preds)
            self.validation_err = self.calc_error(Y_valid, nn_valid.y_preds)
            print(f'error(train): {self.train_err}', file = f)
            print(f'error(test): {self.validation_err}', file = f)
            
        



def return_one_hot(labels, num_examples, num_classes):
    one_hot_matrix = np.zeros((num_examples, num_classes))
    one_hot_matrix[np.arange(num_examples), labels] = 1
    return one_hot_matrix

if __name__ == '__main__':
    train_filename = sys.argv[1]
    valid_filename = sys.argv[2]
    train_ouput = sys.argv[3]
    valid_output = sys.argv[4]
    metrics_out = sys.argv[5]
    epochs = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    initialize_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])


    train_labels, train_features = read_csv_data(train_filename)
    valid_labels, valid_features = read_csv_data(valid_filename)
    nn = NeuralNetwork(train_features, train_labels,  
                        learning_rate, 10, hidden_units, 
                        epochs, initialize_flag)
    

    nn.train(valid_features, valid_labels, metrics_out)
    nn.forward_pass()
    