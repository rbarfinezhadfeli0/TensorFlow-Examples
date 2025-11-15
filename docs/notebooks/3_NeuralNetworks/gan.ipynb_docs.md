# Documentation: notebooks/3_NeuralNetworks/gan.ipynb

## File Metadata

- **File Path**: `notebooks/3_NeuralNetworks/gan.ipynb`
- **File Size**: 46898 bytes
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
    "# Generative Adversarial Network Example\n",
    "\n",
    "Build a generative adversarial network (GAN) to generate digit images from a noise distribution with TensorFlow.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## GAN Overview\n",
    "\n",
    "<img src=\"http://www.timzhangyuxuan.com/static/images/project_DCGAN/structure.png\" alt=\"nn\" style=\"width: 800px;\"/>\n",
    "\n",
    "References:\n",
    "- [Generative adversarial nets](https://arxiv.org/pdf/1406.2661.pdf). I Goodfellow, J Pouget-Abadie, M Mirza, B Xu, D Warde-Farley, S Ozair, Y. Bengio. Advances in neural information processing systems, 2672-2680.\n",
    "- [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html). X Glorot, Y Bengio. Aistats 9, 249-256\n",
    "\n",
    "Other tutorials:\n",
    "- [Generative Adversarial Networks Explained](http://kvfrans.com/generative-adversial-networks-explained/). Kevin Frans.\n",
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
    "num_steps = 70000\n",
    "batch_size = 128\n",
    "learning_rate = 0.0002\n",
    "\n",
    "# Network Params\n",
    "image_dim = 784 # 28*28 pixels\n",
    "gen_hidden_dim = 256\n",
    "disc_hidden_dim = 256\n",
    "noise_dim = 100 # Noise data points\n",
    "\n",
    "# A custom initialization (see Xavier Glorot init)\n",
    "def glorot_init(shape):\n",
    "    return tf.random_normal(shape=shape, stddev=1. / tf.sqrt(shape[0] / 2.))"
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
    "    'gen_hidden1': tf.Variable(glorot_init([noise_dim, gen_hidden_dim])),\n",
    "    'gen_out': tf.Variable(glorot_init([gen_hidden_dim, image_dim])),\n",
    "    'disc_hidden1': tf.Variable(glorot_init([image_dim, disc_hidden_dim])),\n",
    "    'disc_out': tf.Variable(glorot_init([disc_hidden_dim, 1])),\n",
    "}\n",
    "biases = {\n",
    "    'gen_hidden1': tf.Variable(tf.zeros([gen_hidden_dim])),\n",
    "    'gen_out': tf.Variable(tf.zeros([image_dim])),\n",
    "    'disc_hidden1': tf.Variable(tf.zeros([disc_hidden_dim])),\n",
    "    'disc_out': tf.Variable(tf.zeros([1])),\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Generator\n",
    "def generator(x):\n",
    "    hidden_layer = tf.matmul(x, weights['gen_hidden1'])\n",
    "    hidden_layer = tf.add(hidden_layer, biases['gen_hidden1'])\n",
    "    hidden_layer = tf.nn.relu(hidden_layer)\n",
    "    out_layer = tf.matmul(hidden_layer, weights['gen_out'])\n",
    "    out_layer = tf.add(out_layer, biases['gen_out'])\n",
    "    out_layer = tf.nn.sigmoid(out_layer)\n",
    "    return out_layer\n",
    "\n",
    "\n",
    "# Discriminator\n",
    "def discriminator(x):\n",
    "    hidden_layer = tf.matmul(x, weights['disc_hidden1'])\n",
    "    hidden_layer = tf.add(hidden_layer, biases['disc_hidden1'])\n",
    "    hidden_layer = tf.nn.relu(hidden_layer)\n",
    "    out_layer = tf.matmul(hidden_layer, weights['disc_out'])\n",
    "    out_layer = tf.add(out_layer, biases['disc_out'])\n",
    "    out_layer = tf.nn.sigmoid(out_layer)\n",
    "    return out_layer\n",
    "\n",
    "# Build Networks\n",
    "# Network Inputs\n",
    "gen_input = tf.placeholder(tf.float32, shape=[None, noise_dim], name='input_noise')\n",
    "disc_input = tf.placeholder(tf.float32, shape=[None, image_dim], name='disc_input')\n",
    "\n",
    "# Build Generator Network\n",
    "gen_sample = generator(gen_input)\n",
    "\n",
    "# Build 2 Discriminator Networks (one from noise input, one from generated samples)\n",
    "disc_real = discriminator(disc_input)\n",
    "disc_fake = discriminator(gen_sample)\n",
    "\n",
    "# Build Loss\n",
    "gen_loss = -tf.reduce_mean(tf.log(disc_fake))\n",
    "disc_loss = -tf.reduce_mean(tf.log(disc_real) + tf.log(1. - disc_fake))\n",
    "\n",
    "# Build Optimizers\n",
    "optimizer_gen = tf.train.AdamOptimizer(learning_rate=learning_rate)\n",
    "optimizer_disc = tf.train.AdamOptimizer(learning_rate=learning_rate)\n",
    "\n",
    "# Training Variables for each optimizer\n",
    "# By default in TensorFlow, all variables are updated by each optimizer, so we\n",
    "# need to precise for each one of them the specific variables to update.\n",
    "# Generator Network Variables\n",
    "gen_vars = [weights['gen_hidden1'], weights['gen_out'],\n",
    "            biases['gen_hidden1'], biases['gen_out']]\n",
    "# Discriminator Network Variables\n",
    "disc_vars = [weights['disc_hidden1'], weights['disc_out'],\n",
    "            biases['disc_hidden1'], biases['disc_out']]\n",
    "\n",
    "# Create training operations\n",
    "train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)\n",
    "train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)\n",
    "\n",
    "# Initialize the variables (i.e. assign their default value)\n",
    "init = tf.global_variables_initializer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Step 1: Generator Loss: 0.774581, Discriminator Loss: 1.300602\n",
      "Step 2000: Generator Loss: 4.521158, Discriminator Loss: 0.030166\n",
      "Step 4000: Generator Loss: 3.685439, Discriminator Loss: 0.125958\n",
      "Step 6000: Generator Loss: 4.412449, Discriminator Loss: 0.097088\n",
      "Step 8000: Generator Loss: 3.996747, Discriminator Loss: 0.150800\n",
      "Step 10000: Generator Loss: 3.850827, Discriminator Loss: 0.225699\n",
      "Step 12000: Generator Loss: 2.950704, Discriminator Loss: 0.279967\n",
      "Step 14000: Generator Loss: 3.741951, Discriminator Loss: 0.241062\n",
      "Step 16000: Generator Loss: 3.117743, Discriminator Loss: 0.432293\n",
      "Step 18000: Generator Loss: 3.647199, Discriminator Loss: 0.278121\n",
      "Step 20000: Generator Loss: 3.186711, Discriminator Loss: 0.313830\n",
      "Step 22000: Generator Loss: 3.737114, Discriminator Loss: 0.201730\n",
      "Step 24000: Generator Loss: 3.042442, Discriminator Loss: 0.454414\n",
      "Step 26000: Generator Loss: 3.340376, Discriminator Loss: 0.249428\n",
      "Step 28000: Generator Loss: 3.423218, Discriminator Loss: 0.369653\n",
      "Step 30000: Generator Loss: 3.219242, Discriminator Loss: 0.463535\n",
      "Step 32000: Generator Loss: 3.313017, Discriminator Loss: 0.276070\n",
      "Step 34000: Generator Loss: 3.413397, Discriminator Loss: 0.367721\n",
      "Step 36000: Generator Loss: 3.240625, Discriminator Loss: 0.446160\n",
      "Step 38000: Generator Loss: 3.175355, Discriminator Loss: 0.377628\n",
      "Step 40000: Generator Loss: 3.154558, Discriminator Loss: 0.478812\n",
      "Step 42000: Generator Loss: 3.210753, Discriminator Loss: 0.497502\n",
      "Step 44000: Generator Loss: 2.883431, Discriminator Loss: 0.395812\n",
      "Step 46000: Generator Loss: 2.584176, Discriminator Loss: 0.420783\n",
      "Step 48000: Generator Loss: 2.581381, Discriminator Loss: 0.469289\n",
      "Step 50000: Generator Loss: 2.752729, Discriminator Loss: 0.373544\n",
      "Step 52000: Generator Loss: 2.649749, Discriminator Loss: 0.463755\n",
      "Step 54000: Generator Loss: 2.468188, Discriminator Loss: 0.556129\n",
      "Step 56000: Generator Loss: 2.653330, Discriminator Loss: 0.377572\n",
      "Step 58000: Generator Loss: 2.697943, Discriminator Loss: 0.424133\n",
      "Step 60000: Generator Loss: 2.835973, Discriminator Loss: 0.413252\n",
      "Step 62000: Generator Loss: 2.751346, Discriminator Loss: 0.403332\n",
      "Step 64000: Generator Loss: 3.212001, Discriminator Loss: 0.534427\n",
      "Step 66000: Generator Loss: 2.878227, Discriminator Loss: 0.431244\n",
      "Step 68000: Generator Loss: 3.104266, Discriminator Loss: 0.426825\n",
      "Step 70000: Generator Loss: 2.871485, Discriminator Lo
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `notebooks/3_NeuralNetworks/gan.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `notebooks/` directory, specifically within `notebooks/3_NeuralNetworks/`, this file is part of the notebook-based tutorials.



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
jupyter notebook notebooks/3_NeuralNetworks/gan.ipynb
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
- [dcgan.ipynb](dcgan.ipynb_docs.md)
- [dynamic_rnn.ipynb](dynamic_rnn.ipynb_docs.md)
- [neural_network.ipynb](neural_network.ipynb_docs.md)
- [neural_network_eager_api.ipynb](neural_network_eager_api.ipynb_docs.md)
- [neural_network_raw.ipynb](neural_network_raw.ipynb_docs.md)
- [recurrent_network.ipynb](recurrent_network.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

dataset, training, testing, relu, neural network, gan, loss, mnist, sigmoid, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
