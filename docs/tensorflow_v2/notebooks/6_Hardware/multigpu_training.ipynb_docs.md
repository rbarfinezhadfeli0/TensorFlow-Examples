# Documentation: tensorflow_v2/notebooks/6_Hardware/multigpu_training.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/6_Hardware/multigpu_training.ipynb`
- **File Size**: 16704 bytes
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
    "Train a convolutional neural network on multiple GPU with TensorFlow 2.0+.\n",
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
    "## CIFAR10 Dataset Overview\n",
    "\n",
    "The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.\n",
    "\n",
    "![CIFAR10 Dataset](https://storage.googleapis.com/kaggle-competitions/kaggle/3649/media/cifar-10.png)\n",
    "\n",
    "More info: https://www.cs.toronto.edu/~kriz/cifar.html"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from __future__ import absolute_import, division, print_function\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.keras import Model, layers\n",
    "import time\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# MNIST dataset parameters.\n",
    "num_classes = 10 # total classes (0-9 digits).\n",
    "num_gpus = 4\n",
    "\n",
    "# Training parameters.\n",
    "learning_rate = 0.001\n",
    "training_steps = 1000\n",
    "# Split batch size equally between GPUs.\n",
    "# Note: Reduce batch size if you encounter OOM Errors.\n",
    "batch_size = 1024 * num_gpus\n",
    "display_step = 20\n",
    "\n",
    "# Network parameters.\n",
    "conv1_filters = 64 # number of filters for 1st conv layer.\n",
    "conv2_filters = 128 # number of filters for 2nd conv layer.\n",
    "conv3_filters = 256 # number of filters for 2nd conv layer.\n",
    "fc1_units = 2048 # number of neurons for 1st fully-connected layer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare MNIST data.\n",
    "from tensorflow.keras.datasets import cifar10\n",
    "(x_train, y_train), (x_test, y_test) = cifar10.load_data()\n",
    "# Convert to float32.\n",
    "x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)\n",
    "# Normalize images value from [0, 255] to [0, 1].\n",
    "x_train, x_test = x_train / 255., x_test / 255.\n",
    "y_train, y_test = np.reshape(y_train, (-1)), np.reshape(y_test, (-1))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use tf.data API to shuffle and batch data.\n",
    "train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))\n",
    "train_data = train_data.repeat().shuffle(batch_size * 10).batch(batch_size).prefetch(num_gpus)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "class ConvNet(Model):\n",
    "    # Set layers.\n",
    "    def __init__(self):\n",
    "        super(ConvNet, self).__init__()\n",
    "        \n",
    "        # Convolution Layer with 64 filters and a kernel size of 3.\n",
    "        self.conv1_1 = layers.Conv2D(conv1_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)\n",
    "        self.conv1_2 = layers.Conv2D(conv1_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)\n",
    "        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. \n",
    "        self.maxpool1 = layers.MaxPool2D(2, strides=2)\n",
    "\n",
    "        # Convolution Layer with 128 filters and a kernel size of 3.\n",
    "        self.conv2_1 = layers.Conv2D(conv2_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)\n",
    "        self.conv2_2 = layers.Conv2D(conv2_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)\n",
    "        self.conv2_3 = layers.Conv2D(conv2_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)\n",
    "        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. \n",
    "        self.maxpool2 = layers.MaxPool2D(2, strides=2)\n",
    "\n",
    "        # Convolution Layer with 256 filters and a kernel size of 3.\n",
    "        self.conv3_1 = layers.Conv2D(conv3_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)\n",
    "        self.conv3_2 = layers.Conv2D(conv3_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)\n",
    "        self.conv3_3 = layers.Conv2D(conv3_filters, kernel_size=3, padding='SAME', activation=tf.nn.relu)\n",
    "\n",
    "        # Flatten the data to a 1-D vector for the fully connected layer.\n",
    "        self.flatten = layers.Flatten()\n",
    "\n",
    "        # Fully connected layer.\n",
    "        self.fc1 = layers.Dense(1024, activation=tf.nn.relu)\n",
    "        # Apply Dropout (if is_training is False, dropout is not applied).\n",
    "        self.dropout = layers.Dropout(rate=0.5)\n",
    "\n",
    "        # Output layer, class prediction.\n",
    "        self.out = layers.Dense(num_classes)\n",
    "\n",
    "    # Set forward pass.\n",
    "    @tf.function\n",
    "    def call(self, x, is_training=False):\n",
    "        x = self.conv1_1(x)\n",
    "        x = self.conv1_2(x)\n",
    "        x = self.maxpool1(x)\n",
    "        x = self.conv2_1(x)\n",
    "        x = self.conv2_2(x)\n",
    "        x = self.conv2_3(x)\n",
    "        x = self.maxpool2(x)\n",
    "        x = self.conv3_1(x)\n",
    "        x = self.conv3_2(x)\n",
    "        x = self.conv3_3(x)\n",
    "        x = self.flatten(x)\n",
    "        x = self.fc1(x)\n",
    "        x = self.dropout(x, training=is_training)\n",
    "        x = self.out(x)\n",
    "        if not is_training:\n",
    "            # tf cross entropy expect logits without softmax, so only\n",
    "            # apply softmax when not training.\n",
    "            x = tf.nn.softmax(x)\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cross-Entropy Loss.\n",
    "# Note that this will apply 'softmax' to the logits.\n",
    "@tf.function\n",
    "def cross_entropy_loss(x, y):\n",
    "    # Convert labels to int 64 for tf cross-entropy function.\n",
    "    y = tf.cast(y, tf.int64)\n",
    "    # Apply softmax to logits and compute cross-entropy.\n",
    "    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)\n",
    "    # Average loss across the batch.\n",
    "    return tf.reduce_mean(loss)\n",
    "\n",
    "# Accuracy metric.\n",
    "@tf.function\n",
    "def accuracy(y_pred, y_true):\n",
    "    # Predicted class is the index of highest score in prediction vector (i.e. argmax).\n",
    "    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))\n",
    "    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)\n",
    "    \n",
    "\n",
    "@tf.function\n",
    "def backprop(batch_x, batch_y, trainable_variables):\n",
    "    # Wrap computation inside a GradientTape for automatic differentiation.\n",
    "    with tf.GradientTape() as g:\n",
    "        # Forward pass.\n",
    "        pred = conv_net(batch_x, is_training=True)\n",
    "        # Compute loss.\n",
    "        loss = cross_entropy_loss(pred, batch_y)\n",
    "        # Compute gradients.\n",
    "        gradients = g.gradient(loss, trainable_variables)\n",
    "    return gradients\n",
    "\n",
    "# Build the function to average the gradients.\n",
    "@tf.function\n",
    "def average_gradients(tower_grads):\n",
    "    avg_grads = []\n",
    "    for tgrads in zip(*tower_grads):\n",
    "        grads = []\n",
    "        for g in tgrads:\n",
    "            expanded_g = tf.expand_dims(g, 0)\n",
    "            grads.append(expanded_g)\n",
    "        \n",
    "        grad = tf.concat(axis=0, values=grads)\n",
    "        grad = tf.reduce_mean(grad, 0)\n",
    "        \n",
    "        avg_grads.append(grad)\n",
    "        \n",
    "    return avg_grads"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "with tf.device('/cpu:0'):\n",
    "    # Build convnet.\n",
    "    conv_net = ConvNet()\n",
    "    # Stochastic gradient descent optimizer.\n",
    "    optimizer = tf.optimizers.Adam(learning_rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optimization process.\n",
    "def run_optimization(x, y):\n",
    "    # Save gradients for all GPUs.\n",
    "    tower_grads = []\n",
    "    # Variables to update, i.e. trainable variables.\n",
    "    trainable_variables = conv_net.trainable_variables\n",
    "\n",
    "    with tf.device('/cpu:0'):\n",
    "        for i in range(num_gpus):\n",
    "            # Split data between GPUs.\n",
    "            gpu_batch_size = int(batch_size/num_gpus)\n",
    "            batch_x = x[i * gpu_batch_size: (i+1) * gpu_batch_size]\n",
    "            batch_y = y
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/6_Hardware/multigpu_training.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/6_Hardware/`, this file is part of the TensorFlow 2.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 11
- Code cells: 9
- Markdown cells: 2



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v2/notebooks/6_Hardware/multigpu_training.ipynb
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
- **File Permissions**: Ensure saved models and checkpoints have appropriate access controls



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

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

activation, dataset, training, forward pass, gradient, relu, neural network, accuracy, convolution, loss, mnist, softmax, tensorflow, optimizer, pooling, cifar

---

*This documentation was automatically generated for comprehensive repository understanding.*
