import argparse
import numpy as np
import sys
from collections import defaultdict
import copy
import pdb
import csv
import math
import pdb

class Node:
    def __init__(self, data_idxs):
        
        self.data = data_idxs
        self.split_attr = None
        self.split_map = None
        self.inv_map = None
        self.majority = None
        self.minority = None
        self.entropy = None
        self.left = None
        self.right = None
        self.is_leaf = False
        


class DecisionTree:
    
    def __init__(self, x_train, max_depth):
        self.x_train = x_train
        
        self.features = list(x_train.keys())[:-1]
        self.target = list(x_train.keys())[-1]
        self.labels = np.unique(self.x_train[self.target])
        
        self.root = Node(np.arange(len(x_train[self.target])))
        self.y_train = self.calculate_target_labels(self.root.data)
        self.max_depth = max_depth

    
    def calculate_split_index(self, node : Node, remaining_features:list, depth: int) -> None:
        """Splits the node based on the attribute which maximizes the mutual information gain

        Args:
            node (Node): instance of Node class 
            depth (int): depth of the node passed as input (relative to the root)
        """
            
        target_labels = self.calculate_target_labels(node.data)
        
        
        if node.entropy == 0 or depth >= self.max_depth:
            self.create_leaf_node(node, target_labels)
            return
            
        parent_entropy = node.entropy
        min_entropy = parent_entropy

        for feature in remaining_features:
            input_features = [self.x_train[feature][i] for i in node.data]
            unique, counts = np.unique(input_features, return_counts=True)
            split_idxs = []
            children_entropy = 0
            for idx, u in enumerate(unique):
                split_idxs.append(self.return_sub_indexes(feature, node.data ,u))   
                tmp_node = Node(split_idxs[-1])
                children_entropy += len(split_idxs[-1])/len(node.data) * \
                                    self.calculate_entropy(tmp_node)
            
            
            if children_entropy < min_entropy:
                min_entropy = children_entropy
                node.split_attr = feature
                

        if node.split_attr and min_entropy < parent_entropy:
            input_features = [self.x_train[node.split_attr][i] for i in node.data]
            unique, counts = np.unique(input_features, return_counts=True)        
            
            split_idxs_0 = self.return_sub_indexes(str(node.split_attr), node.data ,unique[0])
            sub_unique, sub_counts = np.unique(np.array(self.y_train)[split_idxs_0], return_counts=True)
            
            majority = sub_unique[np.argmax(sub_counts)]
            node.majority = majority
            node.left = Node(split_idxs_0)
            
            if len(unique) > 1:
                split_idxs_1 = self.return_sub_indexes(str(node.split_attr), node.data ,unique[1])                    
                for label in self.labels:
                    if label != str(majority):
                        minority = label
                        break
                node.right = Node(split_idxs_1)
                node.minority = minority
            
        
            node.split_map = {unique[0] : majority, unique[1] : minority}
            node.inv_map = {majority: unique[0], minority: unique[1]}
            node.is_leaf = False
            node.left.entropy = self.calculate_entropy(node.left)
            node.right.entropy = self.calculate_entropy(node.right)

            fts = copy.deepcopy(remaining_features)
            fts.remove(node.split_attr)
            self.calculate_split_index(node.left, fts, depth + 1)
            self.calculate_split_index(node.right, fts, depth + 1)
        else:
            self.create_leaf_node(node, target_labels)


    def predict(self, x_test: list) -> list:
        """Predict by traversing the decision tree

        Args:
            x_test (list): list of input features

        Returns:
            list: list of results (same size as x_test)
        """
        
        first_key = list(x_test.keys())[0]
        res = [0] * len(x_test[first_key])
        for i in range(len(x_test[first_key])):
            node = copy.deepcopy(self.root)
            while node.is_leaf == False:
                
                if str(node.split_map[x_test[node.split_attr][i]]) == node.majority:
                    node = node.left
                else:
                    node = node.right
            
            res[i] = node.majority
        return res


    def create_leaf_node(self, node: Node, target_labels: list):
        """Created the incoming node leaf node and assigns the majority attribute

        Args:
            node (Node): input node to make a leaf node
            target_labels (list): target labels corresponding to the node's data
        """

        unique, counts = np.unique(target_labels, return_counts=True)            
        
        if len(counts) == 2 and counts[0] == counts[1]:
            if less(unique[0], unique[1]):
                node.majority = unique[1]
            else:
                node.majority = unique[0]
        else:
            node.majority = unique[np.argmax(counts)]
        node.is_leaf = True
        return


    def return_sub_indexes(self, feature: str, idxs: list, target : str) -> list:
        """ returns sub indexes where the given features equals the target

        Args:
            feature (str): feature (column) in the data of interest
            idxs (list): indexes to create the subsection of 
            target (str): target label of interest

        Returns:
            list: sub indexes where the feature is equal to the target
        """
        sub_idxs = []
        for i in idxs:
            if np.array(self.x_train[feature])[i] == target:
                sub_idxs.append(i)
        return sub_idxs
    
        
    def calculate_target_labels(self, idxs:list) -> list:
        """return the target labels corresponding to the input indexes

        Args:
            idxs ([list]): input indexes

        Returns:
            list : list of target labels
        """
        return [self.x_train[self.target][idx] for idx in idxs]


    def calculate_entropy(self, node: Node) -> float:
        """Calculates total entropy of a given Node instance

        Args:
            node (Node): input Node instance

        Returns:
            float: total entropy of the labels corresponding to Node's data
        """
        entropy = lambda p : - p * math.log2(p)
        
        target_labels = self.calculate_target_labels(node.data)
        unique, counts = np.unique(np.array(target_labels), return_counts=True)
        total_entropy = 0
        for  count, u in zip(counts, unique):
            total_entropy += entropy(count/len(target_labels))
            
        #print(f'total_entropy: {total_entropy} {counts}')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
        node.entropy = total_entropy
        return total_entropy
    
    def target_distribution(self, idxs: list) -> dict:
        """returns target label distribution of the input indexes

        Args:
            idxs (list): indexes of the data of interest

        Returns:
            dict: dictionary of the label frequency
        """
        distribution = {label : 0 for label in self.labels}
        for idx in idxs:
            distribution[self.y_train[idx]] += 1
        return distribution
        
    def print_tree(self, node: Node, depth: int):
        """Pretty prints the tree
        """
        
        def return_distribution_string(distribution, depth = 0, feature = '', target = ''):
            str_ = []
            for _ in range(depth):
                str_.append('| ')
                

            if feature:
                str_.append(feature)
                str_.append(' = ')
                str_.append(str(target))
                str_.append(' : ')

            str_.append('[')
            for key, value in distribution.items():
                str_.append(str(value) + ' ' + str(key))
                str_.append('/')
                
            str_[-1] = ']'
            return ''.join(str_)
        
        target_distribution = self.target_distribution(node.data)
        if node == None or node.split_attr is None:
            return
        
        if depth == 1:
            print(return_distribution_string(target_distribution))
            
        if node.split_attr is not None:
            target_distribution = self.target_distribution(node.left.data)
            
            print(return_distribution_string(target_distribution, depth, node.split_attr, list(node.inv_map.values())[0]))
            if not node.is_leaf:
                self.print_tree(node.left, depth + 1)
            
            target_distribution = self.target_distribution(node.right.data)
            print(return_distribution_string(target_distribution, depth, node.split_attr, list(node.inv_map.values())[1]))
            
            if not node.is_leaf:
                self.print_tree(node.right, depth + 1)




def generate_output_labels(train_input : str, 
                    test_input: str,
                    max_depth: int,
                    train_out: str,
                    test_out: str,
                    metrics_out: str)->None:
    """Generate output labels .

    Args:
        train_input (str): Path to the training input .tsv file'
        test_input (str): path to the test input .tsv file
        split (int):  the index of feature at which we split the dataset. 
        train_out (str): path of train output .labels file 
        test_out (str): path of test output .labels file 
        metrics_out (str): path of the output .txt file to which metrics such as train and test
    """
    train_data = read_tsv(train_input)
    test_data = read_tsv(test_input)
    
    d = DecisionTree(train_data, max_depth)
    d.root.entropy = d.calculate_entropy(d.root)
    d.calculate_split_index(d.root, d.features, 0)


    test_preds = d.predict(test_data)
    test_error = calculate_error(test_preds, test_data[d.target])

    train_preds = d.predict(train_data)
    train_error = calculate_error(train_preds, d.y_train)

    #print(f'train_error : {train_error} test_error : {test_error} ')
    #d.print_tree(d.root, 1)
    
    write_metrics_file(train_error, test_error, metrics_out)
    write_label_file(train_preds, train_out)
    write_label_file(test_preds, test_out)

def write_label_file(arr: list, out_file: str):
    """Write array to out_file .

    Args:
        arr (list): output array to dump
        out_file (str): file path of the output to dump output to
    """

    with open(out_file, 'w') as f:
        for item in arr:
            print(item, file = f)
    
def read_tsv(file_path: str) -> dict:
    """Read a tab - delimited file .

    Args:
        file_path (str): input file path

    Returns:
        dict: dictionary containing the data
    """
    data = defaultdict(list)
    for record in csv.DictReader(open(file_path), delimiter="\t", quotechar='"'):
        for key, val in record.items():   
            data[key].append(val)
    return data

def calculate_error( predictions: list, ground_truth: list) -> float:
    """Calculate the error based on predictions and ground_truth .

    Args:
        predictions (list): [description]
        ground_truth (list): [description]

    Returns:
        float: [description]
    """
    return len(np.where(np.array(predictions) != np.array(ground_truth))[0])/len(ground_truth)


def write_metrics_file(train_error:str, test_error: str, out_file: str):
    """Write metrics file .

    Args:
        train_error (str): training error
        test_error (str): testing error
        out_file (str): file path to dump metrics to.
    """
    with open(out_file, 'w') as f:
        print(f'error(train): {train_error}', file = f)
        print(f'error(test): {test_error}', file = f)
        


def less(string1, string2):
    # code snippet from stack overflow question https://stackoverflow.com/questions/4806911/string-comparison-technique-used-by-python
    for idx in range(min(len(string1), len(string2))):
        ordinal1, ordinal2 = ord(string1[idx]), ord(string2[idx])
        if ordinal1 == ordinal2:
            continue
        else:
            return ordinal1 < ordinal2
    return len(string1) < len(string2)


if __name__ == '__main__':

    
    train_input  = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

    generate_output_labels(train_input,
                            test_input,
                            max_depth,
                            train_out,
                            test_out,
                            metrics_out)
                            
    
