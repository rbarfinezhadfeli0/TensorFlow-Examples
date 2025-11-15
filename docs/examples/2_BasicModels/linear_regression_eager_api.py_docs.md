# Documentation: examples/2_BasicModels/linear_regression_eager_api.py

## File Metadata

- **File Path**: `examples/2_BasicModels/linear_regression_eager_api.py`
- **File Size**: 1964 bytes
- **File Type**: .py
- **Purpose**: Basic machine learning model implementation

## Original Source

```python
''' Linear Regression with Eager API.

A linear regression learning algorithm example using TensorFlow's Eager API.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Set Eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager

# Training Data
train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]
train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]
n_samples = len(train_X)

# Parameters
learning_rate = 0.01
display_step = 100
num_steps = 1000

# Weight and Bias
W = tfe.Variable(np.random.randn())
b = tfe.Variable(np.random.randn())


# Linear regression (Wx + b)
def linear_regression(inputs):
    return inputs * W + b


# Mean square error
def mean_square_fn(model_fn, inputs, labels):
    return tf.reduce_sum(tf.pow(model_fn(inputs) - labels, 2)) / (2 * n_samples)


# SGD Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Compute gradients
grad = tfe.implicit_gradients(mean_square_fn)

# Initial cost, before optimizing
print("Initial cost= {:.9f}".format(
    mean_square_fn(linear_regression, train_X, train_Y)),
    "W=", W.numpy(), "b=", b.numpy())

# Training
for step in range(num_steps):

    optimizer.apply_gradients(grad(linear_regression, train_X, train_Y))

    if (step + 1) % display_step == 0 or step == 0:
        print("Epoch:", '%04d' % (step + 1), "cost=",
              "{:.9f}".format(mean_square_fn(linear_regression, train_X, train_Y)),
              "W=", W.numpy(), "b=", b.numpy())

# Graphic display
plt.plot(train_X, train_Y, 'ro', label='Original data')
plt.plot(train_X, np.array(W * train_X + b), label='Fitted line')
plt.legend()
plt.show()

```

## High-Level Overview

This file is located at `examples/2_BasicModels/linear_regression_eager_api.py` within the TensorFlow Examples repository.

### Purpose

This file defines functions and methods as part of the TensorFlow Examples repository.

### Context

Located in the `examples/` directory, specifically within `examples/2_BasicModels/`, this file is part of the main examples collection (defaulting to latest TensorFlow).



## Detailed Walkthrough

### Imports and Dependencies

This file imports the following modules:

- `from __future__ import absolute_import, division, print_function`
- `import matplotlib.pyplot as plt`
- `import numpy as np`
- `import tensorflow as tf`

### Functions and Methods

#### `linear_regression(inputs)`

This function handles linear regression functionality.

#### `mean_square_fn(model_fn, inputs, labels)`

This function handles mean square fn functionality.



## Inline Code Examples

### Example Usage

The code in this file demonstrates TensorFlow patterns and can be used as a reference for implementing similar functionality.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It demonstrates basic machine learning models.

### Design Patterns

- Uses TensorFlow framework for machine learning operations
- Organizes functionality into modular functions


## Performance & Complexity

### Performance Characteristics

- **Batch Processing**: Uses batched operations for efficient data processing
- **Training Optimization**: Implements iterative training with multiple epochs

For optimal performance, ensure appropriate hardware resources and TensorFlow GPU support if applicable.



## Security & Safety Considerations

### Security Considerations

As educational example code:

- **Input Validation**: Production use should add input validation and sanitization
- **Data Privacy**: Be cautious when training on sensitive data
- **Model Security**: Trained models can potentially leak information about training data



## Alternatives & Variants

### Alternative Approaches

- **Graph Mode**: This uses eager execution; graph mode (TF 1.x style) is an alternative
- **Model Architectures**: Various network architectures can solve similar problems
- **Training Strategies**: Different optimizers, learning rates, and regularization techniques are possible



## Testing & Usage Notes

### Testing Recommendations

To test this module:

```bash
python examples/2_BasicModels/linear_regression_eager_api.py
```

Verify that:
- TensorFlow is properly installed
- Required datasets are accessible
- Output matches expected results

### Usage Notes

- Ensure TensorFlow is installed: `pip install tensorflow`
- Some examples may require additional dependencies
- GPU support is optional but recommended for large models



## Related Files

### Related Files in Repository

Files in the same directory:

- [gradient_boosted_decision_tree.py](gradient_boosted_decision_tree.py_docs.md)
- [kmeans.py](kmeans.py_docs.md)
- [linear_regression.py](linear_regression.py_docs.md)
- [logistic_regression.py](logistic_regression.py_docs.md)
- [logistic_regression_eager_api.py](logistic_regression_eager_api.py_docs.md)
- [nearest_neighbor.py](nearest_neighbor.py_docs.md)
- [random_forest.py](random_forest.py_docs.md)
- [word2vec.py](word2vec.py_docs.md)


See the [folder index](./index.md) for a complete list of related files.



## Keywords

absolute_import, __future__, training, linear_regression, regression, gradient, tensorflow, mean_square_fn, tensorflow, matplotlib.pyplot, optimizer, numpy

---

*This documentation was automatically generated for comprehensive repository understanding.*
