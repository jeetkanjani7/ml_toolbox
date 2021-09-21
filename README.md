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

```
from ml_toolbox.nn import NeuralNetwork

x = np.array([[0., 0.], [1., 1.]])
y = np.array([0, 1])
nn = NeuralNetwork(x, y, learning_rate = 0.1, num_classes = 10, hidden_units = 4, epochs = 2)
nn.train(x, y, 'logs.out')
nn.predict()
print(nn.y_preds)
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
