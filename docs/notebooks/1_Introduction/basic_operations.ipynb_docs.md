# Documentation: notebooks/1_Introduction/basic_operations.ipynb

## File Metadata

- **File Path**: `notebooks/1_Introduction/basic_operations.ipynb`
- **File Size**: 5297 bytes
- **File Type**: .ipynb
- **Purpose**: Jupyter notebook with interactive code examples

## Original Source

```
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Basic Operations example using TensorFlow library.\n",
    "# Author: Aymeric Damien\n",
    "# Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Basic constant operations\n",
    "# The value returned by the constructor represents the output\n",
    "# of the Constant op.\n",
    "a = tf.constant(2)\n",
    "b = tf.constant(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "a=2, b=3\n",
      "Addition with constants: 5\n",
      "Multiplication with constants: 6\n"
     ]
    }
   ],
   "source": [
    "# Launch the default graph.\n",
    "with tf.Session() as sess:\n",
    "    print \"a: %i\" % sess.run(a), \"b: %i\" % sess.run(b)\n",
    "    print \"Addition with constants: %i\" % sess.run(a+b)\n",
    "    print \"Multiplication with constants: %i\" % sess.run(a*b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Basic Operations with variable as graph input\n",
    "# The value returned by the constructor represents the output\n",
    "# of the Variable op. (define as input when running session)\n",
    "# tf Graph input\n",
    "a = tf.placeholder(tf.int16)\n",
    "b = tf.placeholder(tf.int16)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Define some operations\n",
    "add = tf.add(a, b)\n",
    "mul = tf.multiply(a, b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Addition with variables: 5\n",
      "Multiplication with variables: 6\n"
     ]
    }
   ],
   "source": [
    "# Launch the default graph.\n",
    "with tf.Session() as sess:\n",
    "    # Run every operation with variable input\n",
    "    print \"Addition with variables: %i\" % sess.run(add, feed_dict={a: 2, b: 3})\n",
    "    print \"Multiplication with variables: %i\" % sess.run(mul, feed_dict={a: 2, b: 3})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# ----------------\n",
    "# More in details:\n",
    "# Matrix Multiplication from TensorFlow official tutorial\n",
    "\n",
    "# Create a Constant op that produces a 1x2 matrix.  The op is\n",
    "# added as a node to the default graph.\n",
    "#\n",
    "# The value returned by the constructor represents the output\n",
    "# of the Constant op.\n",
    "matrix1 = tf.constant([[3., 3.]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Create another Constant that produces a 2x1 matrix.\n",
    "matrix2 = tf.constant([[2.],[2.]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.\n",
    "# The returned value, 'product', represents the result of the matrix\n",
    "# multiplication.\n",
    "product = tf.matmul(matrix1, matrix2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 12.]]\n"
     ]
    }
   ],
   "source": [
    "# To run the matmul op we call the session 'run()' method, passing 'product'\n",
    "# which represents the output of the matmul op.  This indicates to the call\n",
    "# that we want to get the output of the matmul op back.\n",
    "#\n",
    "# All inputs needed by the op are run automatically by the session.  They\n",
    "# typically are run in parallel.\n",
    "#\n",
    "# The call 'run(product)' thus causes the execution of threes ops in the\n",
    "# graph: the two constants and matmul.\n",
    "#\n",
    "# The output of the op is returned in 'result' as a numpy `ndarray` object.\n",
    "with tf.Session() as sess:\n",
    "    result = sess.run(product)\n",
    "    print result"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "IPython (Python 2.7)",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2.0
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

```

## High-Level Overview

This file is located at `notebooks/1_Introduction/basic_operations.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `notebooks/` directory, specifically within `notebooks/1_Introduction/`, this file is part of the notebook-based tutorials.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 11
- Code cells: 11
- Markdown cells: 0



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook notebooks/1_Introduction/basic_operations.ipynb
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

- [basic_eager_api.ipynb](basic_eager_api.ipynb_docs.md)
- [helloworld.ipynb](helloworld.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

tensorflow

---

*This documentation was automatically generated for comprehensive repository understanding.*
