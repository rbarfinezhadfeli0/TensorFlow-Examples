# Documentation: tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb`
- **File Size**: 34934 bytes
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
    "# Neural Network Example\n",
    "\n",
    "Build a 2-hidden layers fully connected neural network (a.k.a multilayer perceptron) with TensorFlow v2.\n",
    "\n",
    "This example is using a low-level approach to better understand all mechanics behind building neural networks and the training process.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Neural Network Overview\n",
    "\n",
    "<img src=\"http://cs231n.github.io/assets/nn1/neural_net2.jpeg\" alt=\"nn\" style=\"width: 400px;\"/>\n",
    "\n",
    "## MNIST Dataset Overview\n",
    "\n",
    "This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 255. \n",
    "\n",
    "In this example, each image will be converted to float32, normalized to [0, 1] and flattened to a 1-D array of 784 features (28*28).\n",
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
    "num_features = 784 # data features (img shape: 28*28).\n",
    "\n",
    "# Training parameters.\n",
    "learning_rate = 0.1\n",
    "training_steps = 2000\n",
    "batch_size = 256\n",
    "display_step = 100\n",
    "\n",
    "# Network parameters.\n",
    "n_hidden_1 = 128 # 1st layer number of neurons.\n",
    "n_hidden_2 = 256 # 2nd layer number of neurons."
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
    "# Flatten images to 1-D vector of 784 features (28*28).\n",
    "x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])\n",
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
    "class NeuralNet(Model):\n",
    "    # Set layers.\n",
    "    def __init__(self):\n",
    "        super(NeuralNet, self).__init__()\n",
    "        # First fully-connected hidden layer.\n",
    "        self.fc1 = layers.Dense(n_hidden_1, activation=tf.nn.relu)\n",
    "        # First fully-connected hidden layer.\n",
    "        self.fc2 = layers.Dense(n_hidden_2, activation=tf.nn.relu)\n",
    "        # Second fully-connecter hidden layer.\n",
    "        self.out = layers.Dense(num_classes)\n",
    "\n",
    "    # Set forward pass.\n",
    "    def call(self, x, is_training=False):\n",
    "        x = self.fc1(x)\n",
    "        x = self.fc2(x)\n",
    "        x = self.out(x)\n",
    "        if not is_training:\n",
    "            # tf cross entropy expect logits without softmax, so only\n",
    "            # apply softmax when not training.\n",
    "            x = tf.nn.softmax(x)\n",
    "        return x\n",
    "\n",
    "# Build neural network model.\n",
    "neural_net = NeuralNet()"
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
    "optimizer = tf.optimizers.SGD(learning_rate)"
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
    "        pred = neural_net(x, is_training=True)\n",
    "        # Compute loss.\n",
    "        loss = cross_entropy_loss(pred, y)\n",
    "        \n",
    "    # Variables to update, i.e. trainable variables.\n",
    "    trainable_variables = neural_net.trainable_variables\n",
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
      "step: 100, loss: 2.031049, accuracy: 0.535156\n",
      "step: 200, loss: 1.821917, accuracy: 0.722656\n",
      "step: 300, loss: 1.764789, accuracy: 0.753906\n",
      "step: 400, loss: 1.677593, accuracy: 0.859375\n",
      "step: 500, loss: 1.643402, accuracy: 0.867188\n",
      "step: 600, loss: 1.645116, accuracy: 0.859375\n",
      "step: 700, loss: 1.618012, accuracy: 0.878906\n",
      "step: 800, loss: 1.618097, accuracy: 0.878906\n",
      "step: 900, loss: 1.616565, accuracy: 0.875000\n",
      "step: 1000, loss: 1.599962, accuracy: 0.894531\n",
      "step: 1100, loss: 1.593849, accuracy: 0.910156\n",
      "step: 1200, loss: 1.594491, accuracy: 0.886719\n",
      "step: 1300, loss: 1.622147, accuracy: 0.859375\n",
      "step: 1400, loss: 1.547483, accuracy: 0.937500\n",
      "step: 1500, loss: 1.581775, accuracy: 0.898438\n",
      "step: 1600, loss: 1.555893, accuracy: 0.929688\n",
      "step: 1700, loss: 1.578076, accuracy: 0.898438\n",
      "step: 1800, loss: 1.584776, accuracy: 0.882812\n",
      "step: 1900, loss: 1.563029, accuracy: 0.921875\n",
      "step: 2000, loss: 1.569637, accuracy: 0.902344\n"
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
    "        pred = neural_net(batch_x, is_training=True)\n",
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
      "Test Accuracy: 0.920700\n"
     ]
    }
   ],
   "source": [
    "# Test model on validation set.\n",
    "pred = neural_net(x_test, is_training=False)\n",
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
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAP8AAAD8CAYAAAC4nHJkAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvhp/UCwAADQNJREFUeJzt3W+MVfWdx/HPZylNjPQBWLHEgnQb3bgaAzoaE3AzamxYbYKN1NQHGzbZMH2AZps0ZA1PypMmjemfrU9IpikpJtSWhFbRGBeDGylRGwejBYpQICzMgkAzJgUT0yDfPphDO8W5v3u5/84dv+9XQube8z1/vrnhM+ecOefcnyNCAPL5h7obAFAPwg8kRfiBpAg/kBThB5Ii/EBShB9IivADSRF+IKnP9HNjtrmdEOixiHAr83W057e9wvZB24dtP9nJugD0l9u9t9/2LEmHJD0gaVzSW5Iei4jfF5Zhzw/0WD/2/HdJOhwRRyPiz5J+IWllB+sD0EedhP96SSemvB+vpv0d2yO2x2yPdbAtAF3WyR/8pju0+MRhfUSMShqVOOwHBkkne/5xSQunvP+ipJOdtQOgXzoJ/1uSbrT9JduflfQNSdu70xaAXmv7sD8iLth+XNL/SJolaVNE7O9aZwB6qu1LfW1tjHN+oOf6cpMPgJmL8ANJEX4gKcIPJEX4gaQIP5AU4QeSIvxAUoQfSIrwA0kRfiApwg8kRfiBpAg/kBThB5Ii/EBShB9IivADSRF+ICnCDyRF+IGkCD+QFOEHkiL8QFKEH0iK8ANJEX4gKcIPJEX4gaTaHqJbkmwfk3RO0seSLkTEUDeaAtB7HYW/cm9E/LEL6wHQRxz2A0l1Gv6QtMP2Htsj3WgIQH90eti/LCJO2p4v6RXb70XErqkzVL8U+MUADBhHRHdWZG+QdD4ivl+YpzsbA9BQRLiV+do+7Ld9te3PXXot6SuS9rW7PgD91clh/3WSfm370np+HhE
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb` within the TensorFlow Examples repository.

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
jupyter notebook tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb
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
- [convolutional_network.ipynb](convolutional_network.ipynb_docs.md)
- [convolutional_network_raw.ipynb](convolutional_network_raw.ipynb_docs.md)
- [dcgan.ipynb](dcgan.ipynb_docs.md)
- [dynamic_rnn.ipynb](dynamic_rnn.ipynb_docs.md)
- [neural_network_raw.ipynb](neural_network_raw.ipynb_docs.md)
- [recurrent_network.ipynb](recurrent_network.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

activation, dataset, training, testing, validation, forward pass, gradient, relu, neural network, accuracy, gan, loss, mnist, softmax, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
