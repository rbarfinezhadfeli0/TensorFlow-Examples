# Documentation: tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb`
- **File Size**: 12114 bytes
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
    "# Convolutional Neural Network Example\n",
    "\n",
    "Build a convolutional neural network with TensorFlow.\n",
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
    "This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 1. For simplicity, each image has been flattened and converted to a 1-D numpy array of 784 features (28*28).\n",
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
    "from __future__ import division, print_function, absolute_import\n",
    "\n",
    "import tensorflow as tf\n",
    "\n",
    "# Import MNIST data\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"/tmp/data/\", one_hot=True)"
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
    "# Training Parameters\n",
    "learning_rate = 0.001\n",
    "num_steps = 500\n",
    "batch_size = 128\n",
    "display_step = 10\n",
    "\n",
    "# Network Parameters\n",
    "num_input = 784 # MNIST data input (img shape: 28*28)\n",
    "num_classes = 10 # MNIST total classes (0-9 digits)\n",
    "dropout = 0.75 # Dropout, probability to keep units\n",
    "\n",
    "# tf Graph input\n",
    "X = tf.placeholder(tf.float32, [None, num_input])\n",
    "Y = tf.placeholder(tf.float32, [None, num_classes])\n",
    "keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)"
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
    "# Create some wrappers for simplicity\n",
    "def conv2d(x, W, b, strides=1):\n",
    "    # Conv2D wrapper, with bias and relu activation\n",
    "    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')\n",
    "    x = tf.nn.bias_add(x, b)\n",
    "    return tf.nn.relu(x)\n",
    "\n",
    "\n",
    "def maxpool2d(x, k=2):\n",
    "    # MaxPool2D wrapper\n",
    "    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],\n",
    "                          padding='SAME')\n",
    "\n",
    "\n",
    "# Create model\n",
    "def conv_net(x, weights, biases, dropout):\n",
    "    # MNIST data input is a 1-D vector of 784 features (28*28 pixels)\n",
    "    # Reshape to match picture format [Height x Width x Channel]\n",
    "    # Tensor input become 4-D: [Batch Size, Height, Width, Channel]\n",
    "    x = tf.reshape(x, shape=[-1, 28, 28, 1])\n",
    "\n",
    "    # Convolution Layer\n",
    "    conv1 = conv2d(x, weights['wc1'], biases['bc1'])\n",
    "    # Max Pooling (down-sampling)\n",
    "    conv1 = maxpool2d(conv1, k=2)\n",
    "\n",
    "    # Convolution Layer\n",
    "    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])\n",
    "    # Max Pooling (down-sampling)\n",
    "    conv2 = maxpool2d(conv2, k=2)\n",
    "\n",
    "    # Fully connected layer\n",
    "    # Reshape conv2 output to fit fully connected layer input\n",
    "    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])\n",
    "    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])\n",
    "    fc1 = tf.nn.relu(fc1)\n",
    "    # Apply Dropout\n",
    "    fc1 = tf.nn.dropout(fc1, dropout)\n",
    "\n",
    "    # Output, class prediction\n",
    "    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])\n",
    "    return out"
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
    "# Store layers weight & bias\n",
    "weights = {\n",
    "    # 5x5 conv, 1 input, 32 outputs\n",
    "    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32])),\n",
    "    # 5x5 conv, 32 inputs, 64 outputs\n",
    "    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64])),\n",
    "    # fully connected, 7*7*64 inputs, 1024 outputs\n",
    "    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024])),\n",
    "    # 1024 inputs, 10 outputs (class prediction)\n",
    "    'out': tf.Variable(tf.random_normal([1024, num_classes]))\n",
    "}\n",
    "\n",
    "biases = {\n",
    "    'bc1': tf.Variable(tf.random_normal([32])),\n",
    "    'bc2': tf.Variable(tf.random_normal([64])),\n",
    "    'bd1': tf.Variable(tf.random_normal([1024])),\n",
    "    'out': tf.Variable(tf.random_normal([num_classes]))\n",
    "}\n",
    "\n",
    "# Construct model\n",
    "logits = conv_net(X, weights, biases, keep_prob)\n",
    "prediction = tf.nn.softmax(logits)\n",
    "\n",
    "# Define loss and optimizer\n",
    "loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\n",
    "    logits=logits, labels=Y))\n",
    "optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\n",
    "train_op = optimizer.minimize(loss_op)\n",
    "\n",
    "\n",
    "# Evaluate model\n",
    "correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))\n",
    "accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))\n",
    "\n",
    "# Initialize the variables (i.e. assign their default value)\n",
    "init = tf.global_variables_initializer()"
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
      "Step 1, Minibatch Loss= 63763.3047, Training Accuracy= 0.141\n",
      "Step 10, Minibatch Loss= 26429.6680, Training Accuracy= 0.242\n",
      "Step 20, Minibatch Loss= 12171.8584, Training Accuracy= 0.586\n",
      "Step 30, Minibatch Loss= 6306.6318, Training Accuracy= 0.734\n",
      "Step 40, Minibatch Loss= 5113.7583, Training Accuracy= 0.711\n",
      "Step 50, Minibatch Loss= 4022.2131, Training Accuracy= 0.805\n",
      "Step 60, Minibatch Loss= 3125.4949, Training Accuracy= 0.867\n",
      "Step 70, Minibatch Loss= 2225.4875, Training Accuracy= 0.875\n",
      "Step 80, Minibatch Loss= 1843.3540, Training Accuracy= 0.867\n",
      "Step 90, Minibatch Loss= 1715.7744, Training Accuracy= 0.875\n",
      "Step 100, Minibatch Loss= 2611.2708, Training Accuracy= 0.906\n",
      "Step 110, Minibatch Loss= 4804.0913, Training Accuracy= 0.875\n",
      "Step 120, Minibatch Loss= 1067.5258, Training Accuracy= 0.938\n",
      "Step 130, Minibatch Loss= 2519.1514, Training Accuracy= 0.898\n",
      "Step 140, Minibatch Loss= 2687.9292, Training Accuracy= 0.906\n",
      "Step 150, Minibatch Loss= 1983.4077, Training Accuracy= 0.938\n",
      "Step 160, Minibatch Loss= 2844.6553, Training Accuracy= 0.930\n",
      "Step 170, Minibatch Loss= 3602.2524, Training Accuracy= 0.914\n",
      "Step 180, Minibatch Loss= 175.3922, Training Accuracy= 0.961\n",
      "Step 190, Minibatch Loss= 645.1918, Training Accuracy= 0.945\n",
      "Step 200, Minibatch Loss= 1147.6567, Training Accuracy= 0.938\n",
      "Step 210, Minibatch Loss= 1140.4148, Training Accuracy= 0.914\n",
      "Step 220, Minibatch Loss= 1572.8756, Training Accuracy= 0.906\n",
      "Step 230, Minibatch Loss= 1292.9274, Training Accuracy= 0.898\n",
      "Step 240, Minibatch Loss= 1501.4623, Training Accuracy= 0.953\n",
      "Step 250, Minibatch Loss= 1908.2997, Training Accuracy= 0.898\n",
      "Step 260, Minibatch Loss= 2182.2380, Training Accuracy= 0.898\n",
      "Step 270, Minibatch Loss= 487.5807, Training Accuracy= 0.961\n",
      "Step 280, Minibatch Loss= 1284.1130, Training Accuracy= 0.945\n",
      "Step 290, Minibatch Loss= 1232.4919, Training Accuracy= 0.891\n",
      "Step 300, Minibatch Loss= 1198.8336, Training Accuracy= 0.945\n",
      "Step 310, Minibatch Loss= 2010.5345, Training Accuracy= 0.906\n",
      "Step 320, Minibatch Loss= 786.3917, Training Accuracy= 0.945\n",
      "Step 330, Minibatch Loss= 1408.3556, Training Accuracy= 0.898\n",
      "Step 340, Minibatch Loss= 1453.7538, Training Accuracy= 0.953\n",
      "Step 350, Minibatch Loss= 999.8901, Training Accuracy= 0.906\n",
      "Step 360, Minibatch Loss= 914.3958, Training Accuracy= 0.961\n",
      "Step 370, Minibatch Loss= 488.0052, Training Accuracy= 0.938\n",
      "Step 380, Minibatch Loss= 1070.8710, Training Accuracy= 0.922\n",
      "Step 390, Minibatch Loss= 151.4658, Training Accuracy= 0.961\n",
      "Step 400, Minibatch Loss= 555.3539, Training Accuracy= 0.953\n",
      "Step 410, Minibatch Loss= 765.5746, Training Accuracy= 0.945\n",
      "Step 420, Minibatch Loss= 326.9393, Training Accuracy= 0.969\n",
      "Step 430, Minibatch Loss= 530.8968, Training Accuracy= 0.977\n",
      "Step 440, Minibatch Loss= 463.3909, Training Accuracy= 0.977\n",
      "Step 450, Minibatch Loss= 362.2226, Training Accuracy= 0.977\n",
      "Step 460, Minibatch Loss= 414.0034, Training Accuracy= 0.953\n",
      "Step 470, Minibatch Loss= 583.4587, Training Accuracy= 0.945\n",
      "Step 480, Minibatch Los
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/notebooks/3_NeuralNetworks/`, this file is part of the TensorFlow 1.x examples collection.



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
jupyter notebook tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb
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
- [gan.ipynb](gan.ipynb_docs.md)
- [neural_network.ipynb](neural_network.ipynb_docs.md)
- [neural_network_eager_api.ipynb](neural_network_eager_api.ipynb_docs.md)
- [neural_network_raw.ipynb](neural_network_raw.ipynb_docs.md)
- [recurrent_network.ipynb](recurrent_network.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

activation, dataset, training, testing, relu, neural network, accuracy, convolution, loss, mnist, softmax, tensorflow, optimizer, pooling, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
