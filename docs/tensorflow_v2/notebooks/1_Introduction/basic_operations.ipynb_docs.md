# Documentation: tensorflow_v2/notebooks/1_Introduction/basic_operations.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/1_Introduction/basic_operations.ipynb`
- **File Size**: 3526 bytes
- **File Type**: .ipynb
- **Purpose**: Jupyter notebook with interactive code examples

## Original Source

```
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Basic Tensor Operations\n",
    "\n",
    "Basic tensor operations using TensorFlow v2.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from __future__ import print_function\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define tensor constants.\n",
    "a = tf.constant(2)\n",
    "b = tf.constant(3)\n",
    "c = tf.constant(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "add = 5\n",
      "sub = -1\n",
      "mul = 6\n",
      "div = 0.6666666666666666\n"
     ]
    }
   ],
   "source": [
    "# Various tensor operations.\n",
    "# Note: Tensors also support python operators (+, *, ...)\n",
    "add = tf.add(a, b)\n",
    "sub = tf.subtract(a, b)\n",
    "mul = tf.multiply(a, b)\n",
    "div = tf.divide(a, b)\n",
    "\n",
    "# Access tensors value.\n",
    "print(\"add =\", add.numpy())\n",
    "print(\"sub =\", sub.numpy())\n",
    "print(\"mul =\", mul.numpy())\n",
    "print(\"div =\", div.numpy())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "mean = 3\n",
      "sum = 10\n"
     ]
    }
   ],
   "source": [
    "# Some more operations.\n",
    "mean = tf.reduce_mean([a, b, c])\n",
    "sum = tf.reduce_sum([a, b, c])\n",
    "\n",
    "# Access tensors value.\n",
    "print(\"mean =\", mean.numpy())\n",
    "print(\"sum =\", sum.numpy())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Matrix multiplications.\n",
    "matrix1 = tf.constant([[1., 2.], [3., 4.]])\n",
    "matrix2 = tf.constant([[5., 6.], [7., 8.]])\n",
    "\n",
    "product = tf.matmul(matrix1, matrix2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<tf.Tensor: id=31, shape=(2, 2), dtype=float32, numpy=\n",
       "array([[19., 22.],\n",
       "       [43., 50.]], dtype=float32)>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Display Tensor.\n",
    "product"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[19., 22.],\n",
       "       [43., 50.]], dtype=float32)"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Convert Tensor to Numpy.\n",
    "product.numpy()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/1_Introduction/basic_operations.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/1_Introduction/`, this file is part of the TensorFlow 2.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 8
- Code cells: 7
- Markdown cells: 1



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v2/notebooks/1_Introduction/basic_operations.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



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

- **Different Frameworks**: PyTorch, JAX, or MXNet could be used for similar functionality
- **Model Architectures**: Various network architectures can solve similar problems
- **Training Strategies**: Different optimizers, learning rates, and regularization techniques are possible



## Testing & Usage Notes

### Testing Recommendations

To test this notebook:

```bash
jupyter notebook
```

Then:
1. Open the notebook
2. Run all cells sequentially
3. Verify outputs and visualizations

### Usage Notes

- Ensure TensorFlow is installed: `pip install tensorflow`
- Some examples may require additional dependencies
- GPU support is optional but recommended for large models



## Related Files

### Related Files in Repository

Files in the same directory:

- [helloworld.ipynb](helloworld.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

tensorflow

---

*This documentation was automatically generated for comprehensive repository understanding.*
