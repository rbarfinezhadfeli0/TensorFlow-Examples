# Documentation: notebooks/1_Introduction/basic_eager_api.ipynb

## File Metadata

- **File Path**: `notebooks/1_Introduction/basic_eager_api.ipynb`
- **File Size**: 5799 bytes
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
    "# Basic introduction to TensorFlow's Eager API\n",
    "\n",
    "A simple introduction to get started with TensorFlow's Eager API.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### What is TensorFlow's Eager API ?\n",
    "\n",
    "*Eager execution is an imperative, define-by-run interface where operations are\n",
    "executed immediately as they are called from Python. This makes it easier to\n",
    "get started with TensorFlow, and can make research and development more\n",
    "intuitive. A vast majority of the TensorFlow API remains the same whether eager\n",
    "execution is enabled or not. As a result, the exact same code that constructs\n",
    "TensorFlow graphs (e.g. using the layers API) can be executed imperatively\n",
    "by using eager execution. Conversely, most models written with Eager enabled\n",
    "can be converted to a graph that can be further optimized and/or extracted\n",
    "for deployment in production without changing code. - Rajat Monga*\n",
    "\n",
    "More info: https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from __future__ import absolute_import, division, print_function\n",
    "\n",
    "import numpy as np\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Setting Eager mode...\n"
     ]
    }
   ],
   "source": [
    "# Set Eager API\n",
    "print(\"Setting Eager mode...\")\n",
    "tf.enable_eager_execution()\n",
    "tfe = tf.contrib.eager"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Define constant tensors\n",
      "a = 2\n",
      "b = 3\n"
     ]
    }
   ],
   "source": [
    "# Define constant tensors\n",
    "print(\"Define constant tensors\")\n",
    "a = tf.constant(2)\n",
    "print(\"a = %i\" % a)\n",
    "b = tf.constant(3)\n",
    "print(\"b = %i\" % b)"
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
      "Running operations, without tf.Session\n",
      "a + b = 5\n",
      "a * b = 6\n"
     ]
    }
   ],
   "source": [
    "# Run the operation without the need for tf.Session\n",
    "print(\"Running operations, without tf.Session\")\n",
    "c = a + b\n",
    "print(\"a + b = %i\" % c)\n",
    "d = a * b\n",
    "print(\"a * b = %i\" % d)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mixing operations with Tensors and Numpy Arrays\n",
      "Tensor:\n",
      " a = tf.Tensor(\n",
      "[[2. 1.]\n",
      " [1. 0.]], shape=(2, 2), dtype=float32)\n",
      "NumpyArray:\n",
      " b = [[3. 0.]\n",
      " [5. 1.]]\n"
     ]
    }
   ],
   "source": [
    "# Full compatibility with Numpy\n",
    "print(\"Mixing operations with Tensors and Numpy Arrays\")\n",
    "\n",
    "# Define constant tensors\n",
    "a = tf.constant([[2., 1.],\n",
    "                 [1., 0.]], dtype=tf.float32)\n",
    "print(\"Tensor:\\n a = %s\" % a)\n",
    "b = np.array([[3., 0.],\n",
    "              [5., 1.]], dtype=np.float32)\n",
    "print(\"NumpyArray:\\n b = %s\" % b)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running operations, without tf.Session\n",
      "a + b = tf.Tensor(\n",
      "[[5. 1.]\n",
      " [6. 1.]], shape=(2, 2), dtype=float32)\n",
      "a * b = tf.Tensor(\n",
      "[[11.  1.]\n",
      " [ 3.  0.]], shape=(2, 2), dtype=float32)\n"
     ]
    }
   ],
   "source": [
    "# Run the operation without the need for tf.Session\n",
    "print(\"Running operations, without tf.Session\")\n",
    "\n",
    "c = a + b\n",
    "print(\"a + b = %s\" % c)\n",
    "\n",
    "d = tf.matmul(a, b)\n",
    "print(\"a * b = %s\" % d)"
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
      "Iterate through Tensor 'a':\n",
      "tf.Tensor(2.0, shape=(), dtype=float32)\n",
      "tf.Tensor(1.0, shape=(), dtype=float32)\n",
      "tf.Tensor(1.0, shape=(), dtype=float32)\n",
      "tf.Tensor(0.0, shape=(), dtype=float32)\n"
     ]
    }
   ],
   "source": [
    "print(\"Iterate through Tensor 'a':\")\n",
    "for i in range(a.shape[0]):\n",
    "    for j in range(a.shape[1]):\n",
    "        print(a[i][j])"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [default]",
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
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}

```

## High-Level Overview

This file is located at `notebooks/1_Introduction/basic_eager_api.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `notebooks/` directory, specifically within `notebooks/1_Introduction/`, this file is part of the notebook-based tutorials.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 9
- Code cells: 7
- Markdown cells: 2



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook notebooks/1_Introduction/basic_eager_api.ipynb
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

- **Graph Mode**: This uses eager execution; graph mode (TF 1.x style) is an alternative
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

- [basic_operations.ipynb](basic_operations.ipynb_docs.md)
- [helloworld.ipynb](helloworld.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

tensorflow

---

*This documentation was automatically generated for comprehensive repository understanding.*
