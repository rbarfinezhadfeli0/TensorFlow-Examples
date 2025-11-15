# Documentation: tensorflow_v1/notebooks/6_MultiGPU/multigpu_cnn.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/6_MultiGPU/multigpu_cnn.ipynb`
- **File Size**: 14411 bytes
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
    "# Multi-GPU Training Example\n",
    "\n",
    "Train a convolutional neural network on multiple GPU with TensorFlow.\n",
    "\n",
    "This example is using TensorFlow layers, see 'convolutional_network_raw' example\n",
    "for a raw TensorFlow implementation with variables.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Training with multiple GPU cards\n",
    "\n",
    "In this example, we are using data parallelism to split the training accross multiple GPUs. Each GPU has a full replica of the neural network model, and the weights (i.e. variables) are updated synchronously by waiting that each GPU process its batch of data.\n",
    "\n",
    "First, each GPU process a distinct batch of data and compute the corresponding gradients, then, all gradients are accumulated in the CPU and averaged. The model weights are finally updated with the gradients averaged, and the new model weights are sent back to each GPU, to repeat the training process.\n",
    "\n",
    "<img src=\"https://www.tensorflow.org/images/Parallelism.png\" alt=\"Parallelism\" style=\"width: 400px;\"/>\n",
    "\n",
    "## MNIST Dataset Overview\n",
    "\n",
    "This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 1. For simplicity, each image has been flatten and converted to a 1-D numpy array of 784 features (28*28).\n",
    "\n",
    "![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)\n",
    "\n",
    "More info: http://yann.lecun.com/exdb/mnist/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting /tmp/data/train-images-idx3-ubyte.gz\n",
      "Extracting /tmp/data/train-labels-idx1-ubyte.gz\n",
      "Extracting /tmp/data/t10k-images-idx3-ubyte.gz\n",
      "Extracting /tmp/data/t10k-labels-idx1-ubyte.gz\n"
     ]
    }
   ],
   "source": [
    "from __future__ import print_function\n",
    "\n",
    "import numpy as np\n",
    "import tensorflow as tf\n",
    "import time\n",
    "\n",
    "# Import MNIST data\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"/tmp/data/\", one_hot=True)\n",
    "\n",
    "# Parameters\n",
    "num_gpus = 2\n",
    "num_steps = 200\n",
    "learning_rate = 0.001\n",
    "batch_size = 1024\n",
    "display_step = 10\n",
    "\n",
    "# Network Parameters\n",
    "num_input = 784 # MNIST data input (img shape: 28*28)\n",
    "num_classes = 10 # MNIST total classes (0-9 digits)\n",
    "dropout = 0.75 # Dropout, probability to keep units"
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
    "# Build a convolutional neural network\n",
    "def conv_net(x, n_classes, dropout, reuse, is_training):\n",
    "    # Define a scope for reusing the variables\n",
    "    with tf.variable_scope('ConvNet', reuse=reuse):\n",
    "        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)\n",
    "        # Reshape to match picture format [Height x Width x Channel]\n",
    "        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]\n",
    "        x = tf.reshape(x, shape=[-1, 28, 28, 1])\n",
    "\n",
    "        # Convolution Layer with 64 filters and a kernel size of 5\n",
    "        x = tf.layers.conv2d(x, 64, 5, activation=tf.nn.relu)\n",
    "        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2\n",
    "        x = tf.layers.max_pooling2d(x, 2, 2)\n",
    "\n",
    "        # Convolution Layer with 256 filters and a kernel size of 5\n",
    "        x = tf.layers.conv2d(x, 256, 3, activation=tf.nn.relu)\n",
    "        # Convolution Layer with 512 filters and a kernel size of 5\n",
    "        x = tf.layers.conv2d(x, 512, 3, activation=tf.nn.relu)\n",
    "        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2\n",
    "        x = tf.layers.max_pooling2d(x, 2, 2)\n",
    "\n",
    "        # Flatten the data to a 1-D vector for the fully connected layer\n",
    "        x = tf.contrib.layers.flatten(x)\n",
    "\n",
    "        # Fully connected layer (in contrib folder for now)\n",
    "        x = tf.layers.dense(x, 2048)\n",
    "        # Apply Dropout (if is_training is False, dropout is not applied)\n",
    "        x = tf.layers.dropout(x, rate=dropout, training=is_training)\n",
    "\n",
    "        # Fully connected layer (in contrib folder for now)\n",
    "        x = tf.layers.dense(x, 1024)\n",
    "        # Apply Dropout (if is_training is False, dropout is not applied)\n",
    "        x = tf.layers.dropout(x, rate=dropout, training=is_training)\n",
    "\n",
    "        # Output layer, class prediction\n",
    "        out = tf.layers.dense(x, n_classes)\n",
    "        # Because 'softmax_cross_entropy_with_logits' loss already apply\n",
    "        # softmax, we only apply softmax to testing network\n",
    "        out = tf.nn.softmax(out) if not is_training else out\n",
    "\n",
    "    return out"
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
    "# Build the function to average the gradients\n",
    "def average_gradients(tower_grads):\n",
    "    average_grads = []\n",
    "    for grad_and_vars in zip(*tower_grads):\n",
    "        # Note that each grad_and_vars looks like the following:\n",
    "        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))\n",
    "        grads = []\n",
    "        for g, _ in grad_and_vars:\n",
    "            # Add 0 dimension to the gradients to represent the tower.\n",
    "            expanded_g = tf.expand_dims(g, 0)\n",
    "\n",
    "            # Append on a 'tower' dimension which we will average over below.\n",
    "            grads.append(expanded_g)\n",
    "\n",
    "        # Average over the 'tower' dimension.\n",
    "        grad = tf.concat(grads, 0)\n",
    "        grad = tf.reduce_mean(grad, 0)\n",
    "\n",
    "        # Keep in mind that the Variables are redundant because they are shared\n",
    "        # across towers. So .. we will just return the first tower's pointer to\n",
    "        # the Variable.\n",
    "        v = grad_and_vars[0][1]\n",
    "        grad_and_var = (grad, v)\n",
    "        average_grads.append(grad_and_var)\n",
    "    return average_grads"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# By default, all variables will be placed on '/gpu:0'\n",
    "# So we need a custom device function, to assign all variables to '/cpu:0'\n",
    "# Note: If GPUs are peered, '/gpu:0' can be a faster option\n",
    "PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']\n",
    "\n",
    "def assign_to_device(device, ps_device='/cpu:0'):\n",
    "    def _assign(op):\n",
    "        node_def = op if isinstance(op, tf.NodeDef) else op.node_def\n",
    "        if node_def.op in PS_OPS:\n",
    "            return \"/\" + ps_device\n",
    "        else:\n",
    "            return device\n",
    "\n",
    "    return _assign"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false,
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Step 1: Minibatch Loss= 2.4077, Training Accuracy= 0.123, 682 Examples/sec\n",
      "Step 10: Minibatch Loss= 1.0067, Training Accuracy= 0.765, 6528 Examples/sec\n",
      "Step 20: Minibatch Loss= 0.2442, Training Accuracy= 0.945, 6803 Examples/sec\n",
      "Step 30: Minibatch Loss= 0.2013, Training Accuracy= 0.951, 6741 Examples/sec\n",
      "Step 40: Minibatch Loss= 0.1445, Training Accuracy= 0.962, 6700 Examples/sec\n",
      "Step 50: Minibatch Loss= 0.0940, Training Accuracy= 0.971, 6746 Examples/sec\n",
      "Step 60: Minibatch Loss= 0.0792, Training Accuracy= 0.977, 6627 Examples/sec\n",
      "Step 70: Minibatch Loss= 0.0593, Training Accuracy= 0.979, 6749 Examples/sec\n",
      "Step 80: Minibatch Loss= 0.0799, Training Accuracy= 0.984, 6368 Examples/sec\n",
      "Step 90: Minibatch Loss= 0.0614, Training Accuracy= 0.988, 6762 Examples/sec\n",
      "Step 100: Minibatch Loss= 0.0716, Training Accuracy= 0.983, 6338 Examples/sec\n",
      "Step 110: Minibatch Loss= 0.0531, Training Accuracy= 0.986, 6504 Examples/sec\n",
      "Step 120: Minibatch Loss= 0.0425, Training Accuracy= 0.990, 6721 Examples/sec\n",
      "Step 130: Minibatch Loss= 0.0473, Training Accuracy= 0.986, 6735 Examples/sec\n",
      "Step 140: Minibatch Loss= 0.0345, Training Accuracy= 0.991, 6636 Examples/sec\n",
      "Step 150: Minibatch Loss= 0.0419, Training Accuracy= 0.993, 6777 Examples/sec\n",
      "Step 160: Minibatch Loss= 0.0602, Training Accuracy= 0.984, 6392 Examples/sec\n",
      "Step 170: Minibatch Loss= 0.0425, Training Accuracy= 0.990, 6855 Examples/sec\n",
      "Step 180: Minibatch Loss= 0.0107, Training Accuracy= 0.998, 6804 Examples/sec\n",
      "Step 190: Minibatch Loss= 0.0204, Training Accuracy= 0.995, 6645 Examples/sec\n",
      "Step 200: Minibatch Loss= 0.0296, Training Accuracy= 0.993, 6747 Examples/sec\n",
      "Optimization Finished!\n",
      "Testing Accuracy: 0.990671\n"
     ]
    }
   ],
   "source": [
    "# Place all ops on CPU by default\n",
    "with tf.device('/cpu:0'):\n",
    
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/6_MultiGPU/multigpu_cnn.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/notebooks/6_MultiGPU/`, this file is part of the TensorFlow 1.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 7
- Code cells: 5
- Markdown cells: 2



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v1/notebooks/6_MultiGPU/multigpu_cnn.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It shows how to leverage multiple GPUs for training.

### Design Patterns

- Uses TensorFlow framework for machine learning operations
- Implements object-oriented design with custom classes


## Performance & Complexity

### Performance Characteristics

- **GPU Acceleration**: This code is designed to leverage GPU hardware for accelerated computation
- **Scalability**: Implements multi-GPU training for handling large-scale models and datasets
- **Batch Processing**: Uses batched operations for efficient data processing
- **Training Optimization**: Implements iterative training with multiple epochs
- **Computational Complexity**: Neural network operations can be computationally intensive
- **Memory Usage**: Deep learning models require significant memory for parameters and activations

For optimal performance, ensure appropriate hardware resources and TensorFlow GPU support if applicable.



## Security & Safety Considerations

### Security Considerations

As educational example code:

- **Input Validation**: Production use should add input validation and sanitization
- **Data Privacy**: Be cautious when training on sensitive data
- **Model Security**: Trained models can potentially leak information about training data
- **Data Sources**: Verify integrity of downloaded datasets and models



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

- [multigpu_basics.ipynb](multigpu_basics.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

activation, dataset, training, testing, gradient, relu, neural network, accuracy, convolution, loss, mnist, softmax, tensorflow, optimizer, pooling

---

*This documentation was automatically generated for comprehensive repository understanding.*
