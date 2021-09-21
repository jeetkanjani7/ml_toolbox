# ML Toolbox
Python Package containing ml algorithms written in numpy.

## Installation
To install this project's package run:

```
pip install /path/to/ml_toolbox
```

To install the package in editable mode, use:

```
pip install -e /path/to/ml_toolbox
```

## Example Usage
The package can be imported by another python module with:

### Neural Network
```
from ml_toolbox.nn import NeuralNetwork

# define custom input
x_train = np.array([[0., 0.], [1., 1.]])
y_train = np.array([0, 1])


# instantiate NN
nn = NeuralNetwork(learning_rate = 0.1, num_classes = 10, hidden_units = 4, epochs = 2)

nn.train(x_train, y_train, 'logs.out')
nn.predict(x_test)

```
### Logistic Regression
```
from ml_toolbox.lr import LogisticRegression

# define custom input
x_train = np.array([[0., 0.], [1., 1.]])
y_train = np.array([0, 1])


# instantiate LR
lr = LogisticRegression(learning_rate = 0.1, epochs = 2)

lr.train(x_train, y_train, 'logs.out')
lr.predict(x_test)
```

### Decision Tree
```
from ml_toolbox.dtree import DecisionTree

# define custom input
x_train = np.array([[0., 0.], [1., 1.]])
y_train = np.array([0, 1])


# instantiate Decision Tree
max_depth = 2
dtree = DecisionTree(x_train, max_depth)

dtree.predict(x_test)
print_tree(dtree.root)
```

### Gaussian Naive Bayes
```
from ml_toolbox.gnb import NaiveBayes

# define custom input
x_train = np.array([[0., 0.], [1., 1.]])
y_train = np.array([0, 1])


# instantiate NB
nb = NaiveBayes(x_train, y_train)

nb.predict(x_test)
```

## Testing

To run the tests from the root of the project:
```
python -m pytest tests
```

## Notes
Major part of the code was written for the coursework of 10-601 at CMU. This repository is under development. 
More vectorization, reusability in the code will be added. Some modules will be written in jax.

To contact the author, feel free to write to jkanjani@andrew.cmu.edu
