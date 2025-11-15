# Documentation: tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb`
- **File Size**: 36608 bytes
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
    "# Create some wrappers for simplicity.\n",
    "def conv2d(x, W, b, strides=1):\n",
    "    # Conv2D wrapper, with bias and relu activation.\n",
    "    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')\n",
    "    x = tf.nn.bias_add(x, b)\n",
    "    return tf.nn.relu(x)\n",
    "\n",
    "def maxpool2d(x, k=2):\n",
    "    # MaxPool2D wrapper.\n",
    "    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Store layers weight & bias\n",
    "\n",
    "# A random value generator to initialize weights.\n",
    "random_normal = tf.initializers.RandomNormal()\n",
    "\n",
    "weights = {\n",
    "    # Conv Layer 1: 5x5 conv, 1 input, 32 filters (MNIST has 1 color channel only).\n",
    "    'wc1': tf.Variable(random_normal([5, 5, 1, conv1_filters])),\n",
    "    # Conv Layer 2: 5x5 conv, 32 inputs, 64 filters.\n",
    "    'wc2': tf.Variable(random_normal([5, 5, conv1_filters, conv2_filters])),\n",
    "    # FC Layer 1: 7*7*64 inputs, 1024 units.\n",
    "    'wd1': tf.Variable(random_normal([7*7*64, fc1_units])),\n",
    "    # FC Out Layer: 1024 inputs, 10 units (total number of classes)\n",
    "    'out': tf.Variable(random_normal([fc1_units, num_classes]))\n",
    "}\n",
    "\n",
    "biases = {\n",
    "    'bc1': tf.Variable(tf.zeros([conv1_filters])),\n",
    "    'bc2': tf.Variable(tf.zeros([conv2_filters])),\n",
    "    'bd1': tf.Variable(tf.zeros([fc1_units])),\n",
    "    'out': tf.Variable(tf.zeros([num_classes]))\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create model\n",
    "def conv_net(x):\n",
    "    \n",
    "    # Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images.\n",
    "    x = tf.reshape(x, [-1, 28, 28, 1])\n",
    "\n",
    "    # Convolution Layer. Output shape: [-1, 28, 28, 32].\n",
    "    conv1 = conv2d(x, weights['wc1'], biases['bc1'])\n",
    "    \n",
    "    # Max Pooling (down-sampling). Output shape: [-1, 14, 14, 32].\n",
    "    conv1 = maxpool2d(conv1, k=2)\n",
    "\n",
    "    # Convolution Layer. Output shape: [-1, 14, 14, 64].\n",
    "    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])\n",
    "    \n",
    "    # Max Pooling (down-sampling). Output shape: [-1, 7, 7, 64].\n",
    "    conv2 = maxpool2d(conv2, k=2)\n",
    "\n",
    "    # Reshape conv2 output to fit fully connected layer input, Output shape: [-1, 7*7*64].\n",
    "    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])\n",
    "    \n",
    "    # Fully connected layer, Output shape: [-1, 1024].\n",
    "    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])\n",
    "    # Apply ReLU to fc1 output for non-linearity.\n",
    "    fc1 = tf.nn.relu(fc1)\n",
    "\n",
    "    # Fully connected layer, Output shape: [-1, 10].\n",
    "    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])\n",
    "    # Apply softmax to normalize the logits to a probability distribution.\n",
    "    return tf.nn.softmax(out)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
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
    "# ADAM optimizer.\n",
    "optimizer = tf.optimizers.Adam(learning_rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optimization process. \n",
    "def run_optimization(x, y):\n",
    "    # Wrap computation inside a GradientTape for automatic differentiation.\n",
    "    with tf.GradientTape() as g:\n",
    "        pred = conv_net(x)\n",
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
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "step: 10, loss: 72.370056, accuracy: 0.851562\n",
      "step: 20, loss: 53.936745, accuracy: 0.882812\n",
      "step: 30, loss: 29.929554, accuracy: 0.921875\n",
      "step: 40, loss: 28.075102, accuracy: 0.953125\n",
      "step: 50, loss: 19.366310, accuracy: 0.960938\n",
      "step: 60, loss: 20.398090, accuracy: 0.945312\n",
      "step: 70, loss: 29.320951, accuracy: 0.960938\n",
      "step: 80, loss: 9.121045, accuracy: 0.984375\n",
      "step: 90, loss: 11.680668, accuracy: 0.976562\n",
      "step: 100, loss: 12.413654, accuracy: 0.976562\n",
      "step: 110, loss: 6.675493, accuracy: 0.984375\n",
      "step: 120, loss: 8.730624, accuracy: 0.984375\n",
      "step: 130, loss: 13.608270, accuracy: 0.960938\n",
      "step: 140, loss: 12.859011, accuracy: 0.968750\n",
      "step: 150, loss: 9.110849, accuracy: 0.976562\n",
      "step: 160, loss: 5.832032, accuracy: 0.984375\n",
      "step: 170, loss: 6.996647, accuracy: 0.968750\n",
      "step: 180, loss: 5.325038, accuracy: 0.992188\n",
      "step: 190, loss: 8.866342, accuracy: 0.984375\n",
      "step: 200, loss: 2.626245, accuracy: 1.000000\n"
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
    "        loss = cross_entropy(pred, batch_y)\n",
    "        acc = accuracy(pred, batch_y)\n",
    "        print(\"step: %i, loss: %f, accuracy: %f\" % (step, loss, acc))"
   ]
  },
  {
   "cell_type": "code",

... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb` within the TensorFlow Examples repository.

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
jupyter notebook tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb
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
- [dcgan.ipynb](dcgan.ipynb_docs.md)
- [dynamic_rnn.ipynb](dynamic_rnn.ipynb_docs.md)
- [neural_network.ipynb](neural_network.ipynb_docs.md)
- [neural_network_raw.ipynb](neural_network_raw.ipynb_docs.md)
- [recurrent_network.ipynb](recurrent_network.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

activation, dataset, training, testing, validation, gradient, relu, neural network, accuracy, gan, loss, mnist, convolution, softmax, tensorflow, optimizer, pooling, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
