# Documentation: notebooks/6_MultiGPU/multigpu_basics.ipynb

## File Metadata

- **File Path**: `notebooks/6_MultiGPU/multigpu_basics.ipynb`
- **File Size**: 4385 bytes
- **File Type**: .ipynb
- **Purpose**: Jupyter notebook with interactive code examples

## Original Source

```
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": true
   },
   "source": [
    "# Multi-GPU Basics\n",
    "\n",
    "Basic Multi-GPU computation example using TensorFlow library.\n",
    "\n",
    "This tutorial requires your machine to have 2 GPUs\n",
    "\"/cpu:0\": The CPU of your machine.\n",
    "\"/gpu:0\": The first GPU of your machine\n",
    "\"/gpu:1\": The second GPU of your machine\n",
    "For this example, we are using 2 GTX-980\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
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
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import datetime"
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
    "#Processing Units logs\n",
    "log_device_placement = True\n",
    "\n",
    "#num of multiplications to perform\n",
    "n = 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Example: compute A^n + B^n on 2 GPUs\n",
    "\n",
    "# Create random large matrix\n",
    "A = np.random.rand(1e4, 1e4).astype('float32')\n",
    "B = np.random.rand(1e4, 1e4).astype('float32')\n",
    "\n",
    "# Creates a graph to store results\n",
    "c1 = []\n",
    "c2 = []\n",
    "\n",
    "# Define matrix power\n",
    "def matpow(M, n):\n",
    "    if n < 1: #Abstract cases where n < 1\n",
    "        return M\n",
    "    else:\n",
    "        return tf.matmul(M, matpow(M, n-1))"
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
    "# Single GPU computing\n",
    "\n",
    "with tf.device('/gpu:0'):\n",
    "    a = tf.constant(A)\n",
    "    b = tf.constant(B)\n",
    "    #compute A^n and B^n and store results in c1\n",
    "    c1.append(matpow(a, n))\n",
    "    c1.append(matpow(b, n))\n",
    "\n",
    "with tf.device('/cpu:0'):\n",
    "  sum = tf.add_n(c1) #Addition of all elements in c1, i.e. A^n + B^n\n",
    "\n",
    "t1_1 = datetime.datetime.now()\n",
    "with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:\n",
    "    # Runs the op.\n",
    "    sess.run(sum)\n",
    "t2_1 = datetime.datetime.now()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Multi GPU computing\n",
    "# GPU:0 computes A^n\n",
    "with tf.device('/gpu:0'):\n",
    "    #compute A^n and store result in c2\n",
    "    a = tf.constant(A)\n",
    "    c2.append(matpow(a, n))\n",
    "\n",
    "#GPU:1 computes B^n\n",
    "with tf.device('/gpu:1'):\n",
    "    #compute B^n and store result in c2\n",
    "    b = tf.constant(B)\n",
    "    c2.append(matpow(b, n))\n",
    "\n",
    "with tf.device('/cpu:0'):\n",
    "  sum = tf.add_n(c2) #Addition of all elements in c2, i.e. A^n + B^n\n",
    "\n",
    "t1_2 = datetime.datetime.now()\n",
    "with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:\n",
    "    # Runs the op.\n",
    "    sess.run(sum)\n",
    "t2_2 = datetime.datetime.now()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Single GPU computation time: 0:00:11.833497\n",
      "Multi GPU computation time: 0:00:07.085913\n"
     ]
    }
   ],
   "source": [
    "print \"Single GPU computation time: \" + str(t2_1-t1_1)\n",
    "print \"Multi GPU computation time: \" + str(t2_2-t1_2)"
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
 "nbformat_minor": 0
}

```

## High-Level Overview

This file is located at `notebooks/6_MultiGPU/multigpu_basics.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `notebooks/` directory, specifically within `notebooks/6_MultiGPU/`, this file is part of the notebook-based tutorials.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 7
- Code cells: 6
- Markdown cells: 1



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook notebooks/6_MultiGPU/multigpu_basics.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It shows how to leverage multiple GPUs for training.

### Design Patterns

- Uses TensorFlow framework for machine learning operations


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

- [multigpu_cnn.ipynb](multigpu_cnn.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

tensorflow

---

*This documentation was automatically generated for comprehensive repository understanding.*
