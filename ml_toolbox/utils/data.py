import csv
import numpy as np

def read_csv_data(csv_filename: str): 
    with open(csv_filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        labels = []
        features = []
        for row in csv_reader:
        
            labels.append(int(row[0]))
            features.append(row[1:])
    return np.array(labels), np.array(features).astype(np.float64)



def read_dict(filename):
    dict = {}
    with open(filename, 'r') as f:
        for line in f:
            split = line.split(' ')
            dict[split[0]] = int(split[1].split('\n')[0])
    return dict

