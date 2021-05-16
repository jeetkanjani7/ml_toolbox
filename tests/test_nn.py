from ml_toolbox.nn import NeuralNetwork
from ml_toolbox.utils.data import read_csv_data

def test_nn_performace():

    train_filename = '../data/smallTrain.csv'
    valid_filename = '../data/smallValidation.csv'

    train_labels, train_features = read_csv_data(train_filename)
    valid_labels, valid_features = read_csv_data(valid_filename)

    nn = NeuralNetwork(train_features, 
                        train_labels, 
                        learning_rate=0.1, 
                        num_classes=10, 
                        hidden_units=4, 
                        epochs=2,
                        flag = 2)
    
    nn.train(train_features, train_labels, 'logs.out')
    
    assert nn.train_err == 0.77 
