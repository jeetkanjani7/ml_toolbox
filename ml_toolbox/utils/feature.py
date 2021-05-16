import numpy as np
from abc import ABC, abstractmethod
from collections import defaultdict
import sys
import csv

class FeatureExtractor(ABC):
    def __init__(self, train_file, 
                    test_file, 
                    valid_file,
                    train_formatted_output, 
                    test_formatted_output, 
                    valid_formatted_output, 
                    input_dict_filename):

        self.train_data = self.read_tsv(train_file)
        self.test_data = self.read_tsv(test_file)   
        self.valid_data = self.read_tsv(valid_file)
        self.train_formatted_output = train_formatted_output
        self.test_formatted_output = test_formatted_output
        self.valid_formatted_output = valid_formatted_output
        self.train_labels = np.array(list(self.train_data.values()))
        self.test_labels = np.array(list(self.test_data.values()))
        self.valid_labels = np.array(list(self.valid_data.values()))
        self.input_dict = self.read_dict(input_dict_filename)

    def read_tsv(self, file_path: str) -> dict:
        """Read a tab - delimited file .

        Args:
            file_path (str): input file path

        Returns:
            dict: dictionary containing the data
        """
        data = defaultdict(list)
        with open(file_path) as f:
            for l in f:
                data[l.split('\t')[1]] = l.split('\t')[0]
        return data
    
    @abstractmethod
    def extract_features(self):
        pass

    def formatted_output(self, features, labels, filename):
        with open(filename, 'w') as f:
            tsv_writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            for idx, feature in enumerate(features):
                tmp = [str(labels[idx])]
                for f in feature:
                    tmp.append(f'{self.input_dict[f]}:1')
                tsv_writer.writerow(tmp)
    

    def dump_formatted_output(self):
        self.formatted_output(self.train_features, self.train_labels, self.train_formatted_output)
        self.formatted_output(self.test_features, self.test_labels, self.test_formatted_output)
        self.formatted_output(self.valid_features, self.valid_labels, self.valid_formatted_output)
        
    def read_dict(self, filename):
        dict = {}
        with open(filename, 'r') as f:
            for line in f:
                split = line.split(' ')
                dict[split[0]] = int(split[1].split('\n')[0])
        return dict

    

class BagOfWords(FeatureExtractor):
    def __init__(self, train_file, 
                        test_file, 
                        valid_file,
                        train_formatted_output, 
                        test_formatted_output, 
                        valid_formatted_output, 
                        input_dict_filename):

        super().__init__(train_file, 
                        test_file, 
                        valid_file,
                        train_formatted_output, 
                        test_formatted_output, 
                        valid_formatted_output, 
                        input_dict_filename)

        self.train_features = self.extract_features(self.train_data)
        self.test_features = self.extract_features(self.test_data)
        self.valid_features = self.extract_features(self.valid_data)
        self.dump_formatted_output()
    
    def extract_features(self, reviews):
        features = []
        for review in reviews:
            f = {}
            for idx, word in enumerate(review.split(' ')):
                if word in self.input_dict and word not in f:
                    f[word] = 1
            features.append(f)
        return features
    

class BagOfWordsTrim(FeatureExtractor):
    def __init__(self, train_file, 
                        test_file, 
                        valid_file,
                        train_formatted_output, 
                        test_formatted_output, 
                        valid_formatted_output, 
                        input_dict_filename):

        super().__init__(train_file, 
                        test_file, 
                        valid_file,
                        train_formatted_output, 
                        test_formatted_output, 
                        valid_formatted_output, 
                        input_dict_filename)

        self.train_features = self.extract_features(self.train_data)
        self.test_features = self.extract_features(self.test_data)
        self.valid_features = self.extract_features(self.valid_data)
        self.dump_formatted_output()

    def extract_features(self, reviews):
        features = []
        for review in reviews:    
            f = defaultdict(lambda : 0)
            for idx, word in enumerate(review.split(' ')):
                if word in self.input_dict:
                    f[word] += 1
            features.append({word : 1 for word in f if f[word] < 4})
        return features