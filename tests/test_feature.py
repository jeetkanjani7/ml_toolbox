from ml_toolbox.utils.feature import BagOfWords, BagOfWordsTrim
from ml_toolbox.utils.data import read_csv_data, read_dict
from collections import defaultdict

def read_input(filename, inv_dict):
    features = []
    labels = []
    with open(filename) as f:
        for l in f:
            data = defaultdict(list)
            labels.append(l[0])
            for val in l[1:].split('\t'):
                if val: 
                    word = inv_dict[int(val.split(':')[0])]
                    data[word] = labels[-1]
            features.append(data)
    return features, labels

def test_feature_performace():
    input_dict_filename = '../data/dict.txt'
    input_dict = read_dict(input_dict_filename)
    inv_dict = {v: k for k, v in input_dict.items()}
    train_file = '../data/smalldata/train_data.tsv'
    validation_file = '../data/smalldata/valid_data.tsv'
    test_file = '../data/smalldata/test_data.tsv'
    dict_input = '../data/dict.txt'
    formatted_train_out = './tmp/formatted_train.tsv'
    gt_formatted_train_out = '../data/smalloutput/model1_formatted_train.tsv'
    formatted_validation_out = './tmp/formatted_valid.tsv'
    formatted_test_out = './tmp/formatted_test.tsv'
    feature_flag = 1
    

    if feature_flag == "1":
        model = BagOfWords(train_file, test_file, validation_file, 
                            formatted_train_out, formatted_test_out, 
                            formatted_validation_out, dict_input)

    else:
        model = BagOfWordsTrim(train_file, test_file, validation_file, 
                            formatted_train_out, formatted_test_out,
                            formatted_validation_out, dict_input)
    
        
    pred_train_features, pred_train_labels = read_input(formatted_train_out, inv_dict)
    train_features, train_labels = read_input(gt_formatted_train_out, inv_dict)
    assert train_labels == pred_train_labels