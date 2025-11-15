# Documentation: tensorflow_v1/examples/6_MultiGPU/multigpu_basics.py

## File Metadata

- **File Path**: `tensorflow_v1/examples/6_MultiGPU/multigpu_basics.py`
- **File Size**: 2356 bytes
- **File Type**: .py
- **Purpose**: Multi-GPU training and parallel processing example

## Original Source

```python
from __future__ import print_function
'''
Basic Multi GPU computation example using TensorFlow library.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

'''
This tutorial requires your machine to have 2 GPUs
"/cpu:0": The CPU of your machine.
"/gpu:0": The first GPU of your machine
"/gpu:1": The second GPU of your machine
'''



import numpy as np
import tensorflow as tf
import datetime

# Processing Units logs
log_device_placement = True

# Num of multiplications to perform
n = 10

'''
Example: compute A^n + B^n on 2 GPUs
Results on 8 cores with 2 GTX-980:
 * Single GPU computation time: 0:00:11.277449
 * Multi GPU computation time: 0:00:07.131701
'''
# Create random large matrix
A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')

# Create a graph to store results
c1 = []
c2 = []

def matpow(M, n):
    if n < 1: #Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))

'''
Single GPU computing
'''
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    b = tf.placeholder(tf.float32, [10000, 10000])
    # Compute A^n and B^n and store results in c1
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c1) #Addition of all elements in c1, i.e. A^n + B^n

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a:A, b:B})
t2_1 = datetime.datetime.now()


'''
Multi GPU computing
'''
# GPU:0 computes A^n
with tf.device('/gpu:0'):
    # Compute A^n and store result in c2
    a = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(a, n))

# GPU:1 computes B^n
with tf.device('/gpu:1'):
    # Compute B^n and store result in c2
    b = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c2) #Addition of all elements in c2, i.e. A^n + B^n

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a:A, b:B})
t2_2 = datetime.datetime.now()


print("Single GPU computation time: " + str(t2_1-t1_1))
print("Multi GPU computation time: " + str(t2_2-t1_2))

```

## High-Level Overview

This file is located at `tensorflow_v1/examples/6_MultiGPU/multigpu_basics.py` within the TensorFlow Examples repository.

### Purpose

This file defines functions and methods as part of the TensorFlow Examples repository.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/examples/6_MultiGPU/`, this file is part of the TensorFlow 1.x examples collection.



## Detailed Walkthrough

### Imports and Dependencies

This file imports the following modules:

- `from __future__ import print_function`
- `import numpy as np`
- `import tensorflow as tf`
- `import datetime`

### Functions and Methods

#### `matpow(M, n)`

This function handles matpow functionality.



## Inline Code Examples

### Example Usage

The code in this file demonstrates TensorFlow patterns and can be used as a reference for implementing similar functionality.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It shows how to leverage multiple GPUs for training.

### Design Patterns

- Uses TensorFlow framework for machine learning operations
- Organizes functionality into modular functions


## Performance & Complexity

### Performance Characteristics

- **GPU Acceleration**: This code is designed to leverage GPU hardware for accelerated computation
- **Scalability**: Implements multi-GPU training for handling large-scale models and datasets

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
python tensorflow_v1/examples/6_MultiGPU/multigpu_basics.py
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

- [multigpu_cnn.py](multigpu_cnn.py_docs.md)


See the [folder index](./index.md) for a complete list of related files.



## Keywords

datetime, __future__, print_function, tensorflow, tensorflow, matpow, numpy

---

*This documentation was automatically generated for comprehensive repository understanding.*
