# Documentation: examples/1_Introduction/basic_eager_api.py

## File Metadata

- **File Path**: `examples/1_Introduction/basic_eager_api.py`
- **File Size**: 2002 bytes
- **File Type**: .py
- **Purpose**: Introductory tutorial or example

## Original Source

```python
'''
Basic introduction to TensorFlow's Eager API.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/

What is Eager API?
" Eager execution is an imperative, define-by-run interface where operations are
executed immediately as they are called from Python. This makes it easier to
get started with TensorFlow, and can make research and development more
intuitive. A vast majority of the TensorFlow API remains the same whether eager
execution is enabled or not. As a result, the exact same code that constructs
TensorFlow graphs (e.g. using the layers API) can be executed imperatively
by using eager execution. Conversely, most models written with Eager enabled
can be converted to a graph that can be further optimized and/or extracted
for deployment in production without changing code. " - Rajat Monga

'''
from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

# Set Eager API
print("Setting Eager mode...")
tfe.enable_eager_execution()

# Define constant tensors
print("Define constant tensors")
a = tf.constant(2)
print("a = %i" % a)
b = tf.constant(3)
print("b = %i" % b)

# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")
c = a + b
print("a + b = %i" % c)
d = a * b
print("a * b = %i" % d)


# Full compatibility with Numpy
print("Mixing operations with Tensors and Numpy Arrays")

# Define constant tensors
a = tf.constant([[2., 1.],
                 [1., 0.]], dtype=tf.float32)
print("Tensor:\n a = %s" % a)
b = np.array([[3., 0.],
              [5., 1.]], dtype=np.float32)
print("NumpyArray:\n b = %s" % b)

# Run the operation without the need for tf.Session
print("Running operations, without tf.Session")

c = a + b
print("a + b = %s" % c)

d = tf.matmul(a, b)
print("a * b = %s" % d)

print("Iterate through Tensor 'a':")
for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        print(a[i][j])


```

## High-Level Overview

This file is located at `examples/1_Introduction/basic_eager_api.py` within the TensorFlow Examples repository.

### Purpose

This file as part of the TensorFlow Examples repository.

### Context

Located in the `examples/` directory, specifically within `examples/1_Introduction/`, this file is part of the main examples collection (defaulting to latest TensorFlow).



## Detailed Walkthrough

### Imports and Dependencies

This file imports the following modules:

- `from __future__ import absolute_import, division, print_function`
- `import numpy as np`
- `import tensorflow as tf`
- `import tensorflow.contrib.eager as tfe`



## Inline Code Examples

### Example Usage

The code in this file demonstrates TensorFlow patterns and can be used as a reference for implementing similar functionality.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It belongs to the introductory section, designed for beginners.

### Design Patterns

- Uses TensorFlow framework for machine learning operations


## Performance & Complexity

### Performance Characteristics


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
python examples/1_Introduction/basic_eager_api.py
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

- [basic_operations.py](basic_operations.py_docs.md)
- [helloworld.py](helloworld.py_docs.md)


See the [folder index](./index.md) for a complete list of related files.



## Keywords

absolute_import, __future__, tensorflow.contrib.eager, tensorflow, tensorflow, numpy

---

*This documentation was automatically generated for comprehensive repository understanding.*
