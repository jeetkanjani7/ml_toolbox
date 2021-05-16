from ml_toolbox.lr import LogisticRegression
from ml_toolbox.utils.data import read_csv_data

def test_lr_performace():
    
    formatted_train_out = '../data/formatted_train.tsv'
    formatted_validation_out = '../data/formatted_valid.tsv'
    formatted_test_out = '../data/formatted_test.tsv'
    input_dict_filename = '../data/dict.txt'
    train_out = 'train_out.labels'
    test_out = 'test_out.labels'
    metrics_out = 'metrics_out.txt'
    epochs = 60
    logistic_regression = LogisticRegression(formatted_train_out, 
                                            formatted_test_out, 
                                            formatted_validation_out, 
                                            input_dict_filename,
                                            train_out, 
                                            test_out, 
                                            metrics_out,
                                            lr = 0.1, epochs = epochs)

    
    logistic_regression.run_survey()
    assert logistic_regression.test_err == 0.2
    