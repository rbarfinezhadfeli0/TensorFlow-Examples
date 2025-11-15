# Documentation: tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb`
- **File Size**: 35656 bytes
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
    "# Convolutional Neural Network Example\n",
    "\n",
    "Build a convolutional neural network with TensorFlow v2.\n",
    "\n",
    "This example is using a low-level approach to better understand all mechanics behind building convolutional neural networks and the training process.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## CNN Overview\n",
    "\n",
    "![CNN](http://personal.ie.cuhk.edu.hk/~ccloy/project_target_code/images/fig3.png)\n",
    "\n",
    "## MNIST Dataset Overview\n",
    "\n",
    "This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 255. \n",
    "\n",
    "In this example, each image will be converted to float32 and normalized to [0, 1].\n",
    "\n",
    "![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)\n",
    "\n",
    "More info: http://yann.lecun.com/exdb/mnist/"
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
    "\n",
    "# Training parameters.\n",
    "learning_rate = 0.001\n",
    "training_steps = 200\n",
    "batch_size = 128\n",
    "display_step = 10\n",
    "\n",
    "# Network parameters.\n",
    "conv1_filters = 32 # number of filters for 1st conv layer.\n",
    "conv2_filters = 64 # number of filters for 2nd conv layer.\n",
    "fc1_units = 1024 # number of neurons for 1st fully-connected layer."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare MNIST data.\n",
    "from tensorflow.keras.datasets import mnist\n",
    "(x_train, y_train), (x_test, y_test) = mnist.load_data()\n",
    "# Convert to float32.\n",
    "x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)\n",
    "# Normalize images value from [0, 255] to [0, 1].\n",
    "x_train, x_test = x_train / 255., x_test / 255."
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
    "train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create TF Model.\n",
    "class ConvNet(Model):\n",
    "    # Set layers.\n",
    "    def __init__(self):\n",
    "        super(ConvNet, self).__init__()\n",
    "        # Convolution Layer with 32 filters and a kernel size of 5.\n",
    "        self.conv1 = layers.Conv2D(32, kernel_size=5, activation=tf.nn.relu)\n",
    "        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. \n",
    "        self.maxpool1 = layers.MaxPool2D(2, strides=2)\n",
    "\n",
    "        # Convolution Layer with 64 filters and a kernel size of 3.\n",
    "        self.conv2 = layers.Conv2D(64, kernel_size=3, activation=tf.nn.relu)\n",
    "        # Max Pooling (down-sampling) with kernel size of 2 and strides of 2. \n",
    "        self.maxpool2 = layers.MaxPool2D(2, strides=2)\n",
    "\n",
    "        # Flatten the data to a 1-D vector for the fully connected layer.\n",
    "        self.flatten = layers.Flatten()\n",
    "\n",
    "        # Fully connected layer.\n",
    "        self.fc1 = layers.Dense(1024)\n",
    "        # Apply Dropout (if is_training is False, dropout is not applied).\n",
    "        self.dropout = layers.Dropout(rate=0.5)\n",
    "\n",
    "        # Output layer, class prediction.\n",
    "        self.out = layers.Dense(num_classes)\n",
    "\n",
    "    # Set forward pass.\n",
    "    def call(self, x, is_training=False):\n",
    "        x = tf.reshape(x, [-1, 28, 28, 1])\n",
    "        x = self.conv1(x)\n",
    "        x = self.maxpool1(x)\n",
    "        x = self.conv2(x)\n",
    "        x = self.maxpool2(x)\n",
    "        x = self.flatten(x)\n",
    "        x = self.fc1(x)\n",
    "        x = self.dropout(x, training=is_training)\n",
    "        x = self.out(x)\n",
    "        if not is_training:\n",
    "            # tf cross entropy expect logits without softmax, so only\n",
    "            # apply softmax when not training.\n",
    "            x = tf.nn.softmax(x)\n",
    "        return x\n",
    "\n",
    "# Build neural network model.\n",
    "conv_net = ConvNet()"
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
    "def cross_entropy_loss(x, y):\n",
    "    # Convert labels to int 64 for tf cross-entropy function.\n",
    "    y = tf.cast(y, tf.int64)\n",
    "    # Apply softmax to logits and compute cross-entropy.\n",
    "    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)\n",
    "    # Average loss across the batch.\n",
    "    return tf.reduce_mean(loss)\n",
    "\n",
    "# Accuracy metric.\n",
    "def accuracy(y_pred, y_true):\n",
    "    # Predicted class is the index of highest score in prediction vector (i.e. argmax).\n",
    "    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))\n",
    "    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)\n",
    "\n",
    "# Stochastic gradient descent optimizer.\n",
    "optimizer = tf.optimizers.Adam(learning_rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optimization process. \n",
    "def run_optimization(x, y):\n",
    "    # Wrap computation inside a GradientTape for automatic differentiation.\n",
    "    with tf.GradientTape() as g:\n",
    "        # Forward pass.\n",
    "        pred = conv_net(x, is_training=True)\n",
    "        # Compute loss.\n",
    "        loss = cross_entropy_loss(pred, y)\n",
    "        \n",
    "    # Variables to update, i.e. trainable variables.\n",
    "    trainable_variables = conv_net.trainable_variables\n",
    "\n",
    "    # Compute gradients.\n",
    "    gradients = g.gradient(loss, trainable_variables)\n",
    "    \n",
    "    # Update W and b following gradients.\n",
    "    optimizer.apply_gradients(zip(gradients, trainable_variables))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "step: 10, loss: 1.877234, accuracy: 0.789062\n",
      "step: 20, loss: 1.609390, accuracy: 0.898438\n",
      "step: 30, loss: 1.555458, accuracy: 0.960938\n",
      "step: 40, loss: 1.588296, accuracy: 0.921875\n",
      "step: 50, loss: 1.561057, accuracy: 0.929688\n",
      "step: 60, loss: 1.539851, accuracy: 0.945312\n",
      "step: 70, loss: 1.527458, accuracy: 0.976562\n",
      "step: 80, loss: 1.526701, accuracy: 0.945312\n",
      "step: 90, loss: 1.522610, accuracy: 0.968750\n",
      "step: 100, loss: 1.514970, accuracy: 0.968750\n",
      "step: 110, loss: 1.489902, accuracy: 0.976562\n",
      "step: 120, loss: 1.520697, accuracy: 0.953125\n",
      "step: 130, loss: 1.494326, accuracy: 0.968750\n",
      "step: 140, loss: 1.501781, accuracy: 0.984375\n",
      "step: 150, loss: 1.506588, accuracy: 0.976562\n",
      "step: 160, loss: 1.493378, accuracy: 0.984375\n",
      "step: 170, loss: 1.494006, accuracy: 0.984375\n",
      "step: 180, loss: 1.509782, accuracy: 0.953125\n",
      "step: 190, loss: 1.516123, accuracy: 0.953125\n",
      "step: 200, loss: 1.515508, accuracy: 0.953125\n"
     ]
    }
   ],
   "source": [
    "# Run training for the given number of steps.\n",
    "for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):\n",
    "    # Run the optimization to update W and b values.\n",
    "    run_optimization(batch_x, batch_y)\n",
    "    \n",
    "    if step % display_step == 0:\n",
    "        pred = conv_net(batch_x)\n",
    "        loss = cross_entropy_loss(pred, batch_y)\n",
    "        acc = accuracy(pred, batch_y)\n",
    "        print(\"step: %i, loss: %f, accuracy: %f\" % (step, loss, acc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test Accuracy: 0.977700\n"
     ]
    }
   ],
   "source": [
    "# Test model on validation set.\n",
    "pred = conv_net(x_test)\n",
    "print(\"Test Accuracy: %f\" % accuracy(pred, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize predictions.\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAP8AAAD8CAYAAAC4nHJkAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvhp/UCwAADQNJREFUeJzt3W+MVfWdx/HPZylNjPQBWLHEgnQb3bgaAzoaE3AzamxYbYKN1NQHGzbZMH2AZps0ZA1PypMmjemfrU9IpikpJtS
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/3_NeuralNetworks/`, this file is part of the TensorFlow 2.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 13
- Code cells: 11
- Markdown cells: 2



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It implements neural network architectures and techniques.

### Design Patterns

- Uses TensorFlow framework for machine learning operations
- Implements object-oriented design with custom classes


## Performance & Complexity

### Performance Characteristics

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

- [autoencoder.ipynb](autoencoder.ipynb_docs.md)
- [bidirectional_rnn.ipynb](bidirectional_rnn.ipynb_docs.md)
- [convolutional_network_raw.ipynb](convolutional_network_raw.ipynb_docs.md)
- [dcgan.ipynb](dcgan.ipynb_docs.md)
- [dynamic_rnn.ipynb](dynamic_rnn.ipynb_docs.md)
- [neural_network.ipynb](neural_network.ipynb_docs.md)
- [neural_network_raw.ipynb](neural_network_raw.ipynb_docs.md)
- [recurrent_network.ipynb](recurrent_network.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

training, gradient, relu, neural network, convolution, pooling, activation, validation, accuracy, mnist, softmax, loss, optimizer, dataset, testing, forward pass, gan, tensorflow, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
