# Documentation: tensorflow_v2/notebooks/3_NeuralNetworks/neural_network_raw.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/3_NeuralNetworks/neural_network_raw.ipynb`
- **File Size**: 35894 bytes
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
    "learning_rate = 0.001\n",
    "training_steps = 3000\n",
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
    "# Store layers weight & bias\n",
    "\n",
    "# A random value generator to initialize weights.\n",
    "random_normal = tf.initializers.RandomNormal()\n",
    "\n",
    "weights = {\n",
    "    'h1': tf.Variable(random_normal([num_features, n_hidden_1])),\n",
    "    'h2': tf.Variable(random_normal([n_hidden_1, n_hidden_2])),\n",
    "    'out': tf.Variable(random_normal([n_hidden_2, num_classes]))\n",
    "}\n",
    "biases = {\n",
    "    'b1': tf.Variable(tf.zeros([n_hidden_1])),\n",
    "    'b2': tf.Variable(tf.zeros([n_hidden_2])),\n",
    "    'out': tf.Variable(tf.zeros([num_classes]))\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create model.\n",
    "def neural_net(x):\n",
    "    # Hidden fully connected layer with 128 neurons.\n",
    "    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])\n",
    "    # Apply sigmoid to layer_1 output for non-linearity.\n",
    "    layer_1 = tf.nn.sigmoid(layer_1)\n",
    "    \n",
    "    # Hidden fully connected layer with 256 neurons.\n",
    "    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])\n",
    "    # Apply sigmoid to layer_2 output for non-linearity.\n",
    "    layer_2 = tf.nn.sigmoid(layer_2)\n",
    "    \n",
    "    # Output fully connected layer with a neuron for each class.\n",
    "    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']\n",
    "    # Apply softmax to normalize the logits to a probability distribution.\n",
    "    return tf.nn.softmax(out_layer)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cross-Entropy loss function.\n",
    "def cross_entropy(y_pred, y_true):\n",
    "    # Encode label to a one hot vector.\n",
    "    y_true = tf.one_hot(y_true, depth=num_classes)\n",
    "    # Clip prediction values to avoid log(0) error.\n",
    "    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)\n",
    "    # Compute cross-entropy.\n",
    "    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))\n",
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
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optimization process. \n",
    "def run_optimization(x, y):\n",
    "    # Wrap computation inside a GradientTape for automatic differentiation.\n",
    "    with tf.GradientTape() as g:\n",
    "        pred = neural_net(x)\n",
    "        loss = cross_entropy(pred, y)\n",
    "        \n",
    "    # Variables to update, i.e. trainable variables.\n",
    "    trainable_variables = list(weights.values()) + list(biases.values())\n",
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
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "step: 100, loss: 571.445923, accuracy: 0.222656\n",
      "step: 200, loss: 405.567535, accuracy: 0.488281\n",
      "step: 300, loss: 252.089172, accuracy: 0.660156\n",
      "step: 400, loss: 192.252136, accuracy: 0.792969\n",
      "step: 500, loss: 129.173553, accuracy: 0.855469\n",
      "step: 600, loss: 125.191071, accuracy: 0.859375\n",
      "step: 700, loss: 103.346634, accuracy: 0.890625\n",
      "step: 800, loss: 120.199402, accuracy: 0.871094\n",
      "step: 900, loss: 95.674088, accuracy: 0.890625\n",
      "step: 1000, loss: 113.775406, accuracy: 0.878906\n",
      "step: 1100, loss: 68.457413, accuracy: 0.941406\n",
      "step: 1200, loss: 80.773163, accuracy: 0.914062\n",
      "step: 1300, loss: 85.862785, accuracy: 0.902344\n",
      "step: 1400, loss: 63.480415, accuracy: 0.949219\n",
      "step: 1500, loss: 77.139435, accuracy: 0.910156\n",
      "step: 1600, loss: 88.129692, accuracy: 0.933594\n",
      "step: 1700, loss: 92.199730, accuracy: 0.906250\n",
      "step: 1800, loss: 90.150421, accuracy: 0.886719\n",
      "step: 1900, loss: 48.567772, accuracy: 0.949219\n",
      "step: 2000, loss: 54.002838, accuracy: 0.941406\n",
      "step: 2100, loss: 58.536209, accuracy: 0.933594\n",
      "step: 2200, loss: 47.156784, accuracy: 0.949219\n",
      "step: 2300, loss: 55.344498, accuracy: 0.949219\n",
      "step: 2400, loss: 70.956612, accuracy: 0.925781\n",
      "step: 2500, loss: 76.179062, accuracy: 0.917969\n",
      "step: 2600, loss: 44.956696, accuracy: 0.929688\n",
      "step: 2700, loss: 56.581280, accuracy: 0.941406\n",
      "step: 2800, loss: 57.775612, accuracy: 0.937500\n",
      "step: 2900, loss: 46.005424, accuracy: 0.960938\n",
      "step: 3000, loss: 51.832504, accuracy: 0.953125\n"
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
    "        pred = neural_net(batch_x)\n",
    "        loss = cross_entropy(pred, batch_y)\n",
    "        acc = accuracy(pred, batch_y)\n",
    "        print(\"step: %i, loss: %f, accuracy: %f\" % (step, loss, acc))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Test Accuracy: 0.937600\n"
     ]
    }
   ],
   "source": [
    "# Test model on validation set.\n",
    "pred = neural_net(x_test)\n",
    "print(\"Test Accuracy: %f\" % accuracy(pred, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize predictions.\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAPsAAAD4CAYAAAAq5pAIAAAAOXRFWHRTb2Z0d2FyZQBN
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/3_NeuralNetworks/neural_network_raw.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/3_NeuralNetworks/`, this file is part of the TensorFlow 2.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 15
- Code cells: 13
- Markdown cells: 2



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v2/notebooks/3_NeuralNetworks/neural_network_raw.ipynb
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

- **High-Level APIs**: This uses low-level operations; `tf.keras` provides higher-level abstractions
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
- [neural_network.ipynb](neural_network.ipynb_docs.md)
- [recurrent_network.ipynb](recurrent_network.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

rnn, dataset, training, testing, validation, gradient, neural network, accuracy, gan, loss, mnist, sigmoid, tensorflow, softmax, optimizer, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
