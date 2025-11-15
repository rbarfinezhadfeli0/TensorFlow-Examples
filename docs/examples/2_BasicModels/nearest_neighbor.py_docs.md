# Documentation: examples/2_BasicModels/nearest_neighbor.py

## File Metadata

- **File Path**: `examples/2_BasicModels/nearest_neighbor.py`
- **File Size**: 1739 bytes
- **File Type**: .py
- **Purpose**: Basic machine learning model implementation

## Original Source

```python
'''
A nearest neighbor learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# In this example, we limit mnist data
Xtr, Ytr = mnist.train.next_batch(5000) #5000 for training (nn candidates)
Xte, Yte = mnist.test.next_batch(200) #200 for testing

# tf Graph Input
xtr = tf.placeholder("float", [None, 784])
xte = tf.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance
# Calculate L1 Distance
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)
# Prediction: Get min distance index (Nearest neighbor)
pred = tf.arg_min(distance, 0)

accuracy = 0.

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # Get nearest neighbor class label and compare it to its true label
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]), \
            "True Class:", np.argmax(Yte[i]))
        # Calculate accuracy
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)

```

## High-Level Overview

This file is located at `examples/2_BasicModels/nearest_neighbor.py` within the TensorFlow Examples repository.

### Purpose

This file implements one or more Python classes as part of the TensorFlow Examples repository.

### Context

Located in the `examples/` directory, specifically within `examples/2_BasicModels/`, this file is part of the main examples collection (defaulting to latest TensorFlow).



## Detailed Walkthrough

### Imports and Dependencies

This file imports the following modules:

- `from __future__ import print_function`
- `import numpy as np`
- `import tensorflow as tf`
- `from tensorflow.examples.tutorials.mnist import input_data`



## Inline Code Examples

### Example Usage

The code in this file demonstrates TensorFlow patterns and can be used as a reference for implementing similar functionality.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It demonstrates basic machine learning models.

### Design Patterns

- Uses TensorFlow framework for machine learning operations
- Implements object-oriented design with custom classes


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

- **Different Frameworks**: PyTorch, JAX, or MXNet could be used for similar functionality
- **Model Architectures**: Various network architectures can solve similar problems
- **Training Strategies**: Different optimizers, learning rates, and regularization techniques are possible



## Testing & Usage Notes

### Testing Recommendations

To test this module:

```bash
python examples/2_BasicModels/nearest_neighbor.py
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
- [logistic_regression_eager_api.py](logistic_regression_eager_api.py_docs.md)
- [random_forest.py](random_forest.py_docs.md)
- [word2vec.py](word2vec.py_docs.md)


See the [folder index](./index.md) for a complete list of related files.



## Keywords

__future__, training, testing, label, accuracy, mnist, print_function, input_data, tensorflow, tensorflow.examples.tutorials.mnist, tensorflow, numpy

---

*This documentation was automatically generated for comprehensive repository understanding.*
