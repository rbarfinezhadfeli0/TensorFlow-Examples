# Documentation: examples/2_BasicModels/logistic_regression_eager_api.py

## File Metadata

- **File Path**: `examples/2_BasicModels/logistic_regression_eager_api.py`
- **File Size**: 3088 bytes
- **File Type**: .py
- **Purpose**: Basic machine learning model implementation

## Original Source

```python
''' Logistic Regression with Eager API.

A logistic regression learning algorithm example using TensorFlow's Eager API.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''
from __future__ import absolute_import, division, print_function

import tensorflow as tf

# Set Eager API
tf.enable_eager_execution()
tfe = tf.contrib.eager

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)

# Parameters
learning_rate = 0.1
batch_size = 128
num_steps = 1000
display_step = 100

dataset = tf.data.Dataset.from_tensor_slices(
    (mnist.train.images, mnist.train.labels))
dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
dataset_iter = tfe.Iterator(dataset)

# Variables
W = tfe.Variable(tf.zeros([784, 10]), name='weights')
b = tfe.Variable(tf.zeros([10]), name='bias')


# Logistic regression (Wx + b)
def logistic_regression(inputs):
    return tf.matmul(inputs, W) + b


# Cross-Entropy loss function
def loss_fn(inference_fn, inputs, labels):
    # Using sparse_softmax cross entropy
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=inference_fn(inputs), labels=labels))


# Calculate accuracy
def accuracy_fn(inference_fn, inputs, labels):
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# SGD Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# Compute gradients
grad = tfe.implicit_gradients(loss_fn)

# Training
average_loss = 0.
average_acc = 0.
for step in range(num_steps):

    # Iterate through the dataset
    d = dataset_iter.next()

    # Images
    x_batch = d[0]
    # Labels
    y_batch = tf.cast(d[1], dtype=tf.int64)

    # Compute the batch loss
    batch_loss = loss_fn(logistic_regression, x_batch, y_batch)
    average_loss += batch_loss
    # Compute the batch accuracy
    batch_accuracy = accuracy_fn(logistic_regression, x_batch, y_batch)
    average_acc += batch_accuracy

    if step == 0:
        # Display the initial cost, before optimizing
        print("Initial loss= {:.9f}".format(average_loss))

    # Update the variables following gradients info
    optimizer.apply_gradients(grad(logistic_regression, x_batch, y_batch))

    # Display info
    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            average_loss /= display_step
            average_acc /= display_step
        print("Step:", '%04d' % (step + 1), " loss=",
              "{:.9f}".format(average_loss), " accuracy=",
              "{:.4f}".format(average_acc))
        average_loss = 0.
        average_acc = 0.

# Evaluate model on the test image set
testX = mnist.test.images
testY = mnist.test.labels

test_acc = accuracy_fn(logistic_regression, testX, testY)
print("Testset Accuracy: {:.4f}".format(test_acc))

```

## High-Level Overview

This file is located at `examples/2_BasicModels/logistic_regression_eager_api.py` within the TensorFlow Examples repository.

### Purpose

This file defines functions and methods as part of the TensorFlow Examples repository.

### Context

Located in the `examples/` directory, specifically within `examples/2_BasicModels/`, this file is part of the main examples collection (defaulting to latest TensorFlow).



## Detailed Walkthrough

### Imports and Dependencies

This file imports the following modules:

- `from __future__ import absolute_import, division, print_function`
- `import tensorflow as tf`
- `from tensorflow.examples.tutorials.mnist import input_data`

### Functions and Methods

#### `logistic_regression(inputs)`

This function handles logistic regression functionality.

#### `loss_fn(inference_fn, inputs, labels)`

This function handles loss fn functionality.

#### `accuracy_fn(inference_fn, inputs, labels)`

This function handles accuracy fn functionality.



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
python examples/2_BasicModels/logistic_regression_eager_api.py
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
- [linear_regression_eager_api.py](linear_regression_eager_api.py_docs.md)
- [logistic_regression.py](logistic_regression.py_docs.md)
- [nearest_neighbor.py](nearest_neighbor.py_docs.md)
- [random_forest.py](random_forest.py_docs.md)
- [word2vec.py](word2vec.py_docs.md)


See the [folder index](./index.md) for a complete list of related files.



## Keywords

logistic_regression, absolute_import, __future__, training, dataset, regression, loss_fn, gradient, accuracy, loss, mnist, tensorflow, input_data, tensorflow.examples.tutorials.mnist, tensorflow, optimizer, softmax, accuracy_fn

---

*This documentation was automatically generated for comprehensive repository understanding.*
