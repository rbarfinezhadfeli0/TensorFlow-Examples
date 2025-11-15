# Documentation: notebooks/3_NeuralNetworks/dcgan.ipynb

## File Metadata

- **File Path**: `notebooks/3_NeuralNetworks/dcgan.ipynb`
- **File Size**: 50310 bytes
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
    "# Deep Convolutional Generative Adversarial Network Example\n",
    "\n",
    "Build a deep convolutional generative adversarial network (DCGAN) to generate digit images from a noise distribution with TensorFlow.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## DCGAN Overview\n",
    "\n",
    "<img src=\"https://camo.githubusercontent.com/45e147fc9dfcf6a8e5df2c9b985078258b9974e3/68747470733a2f2f63646e2d696d616765732d312e6d656469756d2e636f6d2f6d61782f313030302f312a33394e6e6e695f6e685044614c7539416e544c6f57772e706e67\" alt=\"dcgan\" style=\"width: 1000px;\"/>\n",
    "\n",
    "References:\n",
    "- [Unsupervised representation learning with deep convolutional generative adversarial networks](https://arxiv.org/pdf/1511.06434). A Radford, L Metz, S Chintala, 2016.\n",
    "- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html). X Glorot, Y Bengio. Aistats 9, 249-256\n",
    "- [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167). Sergey Ioffe, Christian Szegedy. 2015.\n",
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
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from __future__ import division, print_function, absolute_import\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
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
    "# Import MNIST data\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"/tmp/data/\", one_hot=True)"
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
    "# Training Params\n",
    "num_steps = 10000\n",
    "batch_size = 128\n",
    "lr_generator = 0.002\n",
    "lr_discriminator = 0.002\n",
    "\n",
    "# Network Params\n",
    "image_dim = 784 # 28*28 pixels * 1 channel\n",
    "noise_dim = 100 # Noise data points"
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
    "# Build Networks\n",
    "# Network Inputs\n",
    "noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])\n",
    "real_image_input = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])\n",
    "# A boolean to indicate batch normalization if it is training or inference time\n",
    "is_training = tf.placeholder(tf.bool)\n",
    "\n",
    "#LeakyReLU activation\n",
    "def leakyrelu(x, alpha=0.2):\n",
    "    return 0.5 * (1 + alpha) * x + 0.5 * (1 - alpha) * abs(x)\n",
    "\n",
    "# Generator Network\n",
    "# Input: Noise, Output: Image\n",
    "# Note that batch normalization has different behavior at training and inference time,\n",
    "# we then use a placeholder to indicates the layer if we are training or not.\n",
    "def generator(x, reuse=False):\n",
    "    with tf.variable_scope('Generator', reuse=reuse):\n",
    "        # TensorFlow Layers automatically create variables and calculate their\n",
    "        # shape, based on the input.\n",
    "        x = tf.layers.dense(x, units=7 * 7 * 128)\n",
    "        x = tf.layers.batch_normalization(x, training=is_training)\n",
    "        x = tf.nn.relu(x)\n",
    "        # Reshape to a 4-D array of images: (batch, height, width, channels)\n",
    "        # New shape: (batch, 7, 7, 128)\n",
    "        x = tf.reshape(x, shape=[-1, 7, 7, 128])\n",
    "        # Deconvolution, image shape: (batch, 14, 14, 64)\n",
    "        x = tf.layers.conv2d_transpose(x, 64, 5, strides=2, padding='same')\n",
    "        x = tf.layers.batch_normalization(x, training=is_training)\n",
    "        x = tf.nn.relu(x)\n",
    "        # Deconvolution, image shape: (batch, 28, 28, 1)\n",
    "        x = tf.layers.conv2d_transpose(x, 1, 5, strides=2, padding='same')\n",
    "        # Apply tanh for better stability - clip values to [-1, 1].\n",
    "        x = tf.nn.tanh(x)\n",
    "        return x\n",
    "\n",
    "\n",
    "# Discriminator Network\n",
    "# Input: Image, Output: Prediction Real/Fake Image\n",
    "def discriminator(x, reuse=False):\n",
    "    with tf.variable_scope('Discriminator', reuse=reuse):\n",
    "        # Typical convolutional neural network to classify images.\n",
    "        x = tf.layers.conv2d(x, 64, 5, strides=2, padding='same')\n",
    "        x = tf.layers.batch_normalization(x, training=is_training)\n",
    "        x = leakyrelu(x)\n",
    "        x = tf.layers.conv2d(x, 128, 5, strides=2, padding='same')\n",
    "        x = tf.layers.batch_normalization(x, training=is_training)\n",
    "        x = leakyrelu(x)\n",
    "        # Flatten\n",
    "        x = tf.reshape(x, shape=[-1, 7*7*128])\n",
    "        x = tf.layers.dense(x, 1024)\n",
    "        x = tf.layers.batch_normalization(x, training=is_training)\n",
    "        x = leakyrelu(x)\n",
    "        # Output 2 classes: Real and Fake images\n",
    "        x = tf.layers.dense(x, 2)\n",
    "    return x\n",
    "\n",
    "# Build Generator Network\n",
    "gen_sample = generator(noise_input)\n",
    "\n",
    "# Build 2 Discriminator Networks (one from noise input, one from generated samples)\n",
    "disc_real = discriminator(real_image_input)\n",
    "disc_fake = discriminator(gen_sample, reuse=True)\n",
    "\n",
    "# Build the stacked generator/discriminator\n",
    "stacked_gan = discriminator(gen_sample, reuse=True)\n",
    "\n",
    "# Build Loss (Labels for real images: 1, for fake images: 0)\n",
    "# Discriminator Loss for real and fake samples\n",
    "disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\n",
    "    logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))\n",
    "disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\n",
    "    logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))\n",
    "# Sum both loss\n",
    "disc_loss = disc_loss_real + disc_loss_fake\n",
    "# Generator Loss (The generator tries to fool the discriminator, thus labels are 1)\n",
    "gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\n",
    "    logits=stacked_gan, labels=tf.ones([batch_size], dtype=tf.int32)))\n",
    "\n",
    "# Build Optimizers\n",
    "optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_generator, beta1=0.5, beta2=0.999)\n",
    "optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_discriminator, beta1=0.5, beta2=0.999)\n",
    "\n",
    "# Training Variables for each optimizer\n",
    "# By default in TensorFlow, all variables are updated by each optimizer, so we\n",
    "# need to precise for each one of them the specific variables to update.\n",
    "# Generator Network Variables\n",
    "gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')\n",
    "# Discriminator Network Variables\n",
    "disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')\n",
    "\n",
    "# Create training operations\n",
    "# TensorFlow UPDATE_OPS collection holds all batch norm operation to update the moving mean/stddev\n",
    "gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')\n",
    "# `control_dependencies` ensure that the `gen_update_ops` will be run before the `minimize` op (backprop)\n",
    "with tf.control_dependencies(gen_update_ops):\n",
    "    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)\n",
    "disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')\n",
    "with tf.control_dependencies(disc_update_ops):\n",
    "    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)\n",
    "    \n",
    "# Initialize the variables (i.e. assign their default value)\n",
    "init = tf.global_variables_initializer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Step 1: Generator Loss: 3.590350, Discriminator Loss: 1.907586\n",
      "Step 500: Generator Loss: 1.254698, Discriminator Loss: 1.005236\n",
      "Step 1000: Generator Loss: 1.730409, Discriminator Loss: 0.837684\n",
      "Step 1500: Generator Loss: 1.962198, Discriminator Loss: 0.618827\n",
      "Step 2000: Generator Loss: 2.767945, Discriminator Loss: 0.378071\n",
      "Step 2500: Generator Loss: 2.370605, Discriminator Loss: 0.561247\n",
      "Step 3000: Generator Loss: 3.427798, Discriminator Loss: 0.402951\n",
      "Step 3500: Generator Loss: 4.904454, Discriminator Loss: 
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `notebooks/3_NeuralNetworks/dcgan.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `notebooks/` directory, specifically within `notebooks/3_NeuralNetworks/`, this file is part of the notebook-based tutorials.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 8
- Code cells: 6
- Markdown cells: 2



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook notebooks/3_NeuralNetworks/dcgan.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It implements neural network architectures and techniques.

### Design Patterns

- Uses TensorFlow framework for machine learning operations


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

activation, rnn, dataset, training, testing, relu, neural network, lstm, gan, loss, mnist, convolution, softmax, tensorflow, optimizer, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
