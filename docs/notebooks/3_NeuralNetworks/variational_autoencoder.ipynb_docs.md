# Documentation: notebooks/3_NeuralNetworks/variational_autoencoder.ipynb

## File Metadata

- **File Path**: `notebooks/3_NeuralNetworks/variational_autoencoder.ipynb`
- **File Size**: 298726 bytes
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
    "# Variational Auto-Encoder Example\n",
    "\n",
    "Build a variational auto-encoder (VAE) to generate digit images from a noise distribution with TensorFlow.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## VAE Overview\n",
    "\n",
    "<img src=\"http://kvfrans.com/content/images/2016/08/vae.jpg\" alt=\"vae\" style=\"width: 800px;\"/>\n",
    "\n",
    "References:\n",
    "- [Auto-Encoding Variational Bayes The International Conference on Learning Representations](https://arxiv.org/abs/1312.6114) (ICLR), Banff, 2014. D.P. Kingma, M. Welling\n",
    "- [Understanding the difficulty of training deep feedforward neural networks](www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.../AISTATS2010_Glorot.pdf). X Glorot, Y Bengio. Aistats 9, 249-256\n",
    "\n",
    "Other tutorials:\n",
    "- [Variational Auto Encoder Explained](http://kvfrans.com/variational-autoencoders-explained/). Kevin Frans.\n",
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
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "from scipy.stats import norm\n",
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
      "Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.\n",
      "Extracting /tmp/data/train-images-idx3-ubyte.gz\n",
      "Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.\n",
      "Extracting /tmp/data/train-labels-idx1-ubyte.gz\n",
      "Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.\n",
      "Extracting /tmp/data/t10k-images-idx3-ubyte.gz\n",
      "Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.\n",
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
    "# Parameters\n",
    "learning_rate = 0.001\n",
    "num_steps = 30000\n",
    "batch_size = 64\n",
    "\n",
    "# Network Parameters\n",
    "image_dim = 784 # MNIST images are 28x28 pixels\n",
    "hidden_dim = 512\n",
    "latent_dim = 2\n",
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
    "# Variables\n",
    "weights = {\n",
    "    'encoder_h1': tf.Variable(glorot_init([image_dim, hidden_dim])),\n",
    "    'z_mean': tf.Variable(glorot_init([hidden_dim, latent_dim])),\n",
    "    'z_std': tf.Variable(glorot_init([hidden_dim, latent_dim])),\n",
    "    'decoder_h1': tf.Variable(glorot_init([latent_dim, hidden_dim])),\n",
    "    'decoder_out': tf.Variable(glorot_init([hidden_dim, image_dim]))\n",
    "}\n",
    "biases = {\n",
    "    'encoder_b1': tf.Variable(glorot_init([hidden_dim])),\n",
    "    'z_mean': tf.Variable(glorot_init([latent_dim])),\n",
    "    'z_std': tf.Variable(glorot_init([latent_dim])),\n",
    "    'decoder_b1': tf.Variable(glorot_init([hidden_dim])),\n",
    "    'decoder_out': tf.Variable(glorot_init([image_dim]))\n",
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
    "# Building the encoder\n",
    "input_image = tf.placeholder(tf.float32, shape=[None, image_dim])\n",
    "encoder = tf.matmul(input_image, weights['encoder_h1']) + biases['encoder_b1']\n",
    "encoder = tf.nn.tanh(encoder)\n",
    "z_mean = tf.matmul(encoder, weights['z_mean']) + biases['z_mean']\n",
    "z_std = tf.matmul(encoder, weights['z_std']) + biases['z_std']\n",
    "\n",
    "# Sampler: Normal (gaussian) random distribution\n",
    "eps = tf.random_normal(tf.shape(z_std), dtype=tf.float32, mean=0., stddev=1.0,\n",
    "                       name='epsilon')\n",
    "z = z_mean + tf.exp(z_std / 2) * eps\n",
    "\n",
    "# Building the decoder (with scope to re-use these layers later)\n",
    "decoder = tf.matmul(z, weights['decoder_h1']) + biases['decoder_b1']\n",
    "decoder = tf.nn.tanh(decoder)\n",
    "decoder = tf.matmul(decoder, weights['decoder_out']) + biases['decoder_out']\n",
    "decoder = tf.nn.sigmoid(decoder)"
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
    "# Define VAE Loss\n",
    "def vae_loss(x_reconstructed, x_true):\n",
    "    # Reconstruction loss\n",
    "    encode_decode_loss = x_true * tf.log(1e-10 + x_reconstructed) \\\n",
    "                         + (1 - x_true) * tf.log(1e-10 + 1 - x_reconstructed)\n",
    "    encode_decode_loss = -tf.reduce_sum(encode_decode_loss, 1)\n",
    "    # KL Divergence loss\n",
    "    kl_div_loss = 1 + z_std - tf.square(z_mean) - tf.exp(z_std)\n",
    "    kl_div_loss = -0.5 * tf.reduce_sum(kl_div_loss, 1)\n",
    "    return tf.reduce_mean(encode_decode_loss + kl_div_loss)\n",
    "\n",
    "loss_op = vae_loss(decoder, input_image)\n",
    "optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)\n",
    "train_op = optimizer.minimize(loss_op)\n",
    "\n",
    "# Initialize the variables (i.e. assign their default value)\n",
    "init = tf.global_variables_initializer()"
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
      "Step 1, Loss: 645.076538\n",
      "Step 1000, Loss: 173.018188\n",
      "Step 2000, Loss: 165.299225\n",
      "Step 3000, Loss: 172.933685\n",
      "Step 4000, Loss: 161.475052\n",
      "Step 5000, Loss: 179.529831\n",
      "Step 6000, Loss: 166.430023\n",
      "Step 7000, Loss: 167.152176\n",
      "Step 8000, Loss: 159.920242\n",
      "Step 9000, Loss: 160.172363\n",
      "Step 10000, Loss: 150.077652\n",
      "Step 11000, Loss: 162.774567\n",
      "Step 12000, Loss: 156.187820\n",
      "Step 13000, Loss: 148.331573\n",
      "Step 14000, Loss: 153.757202\n",
      "Step 15000, Loss: 158.050598\n",
      "Step 16000, Loss: 163.068939\n",
      "Step 17000, Loss: 152.765152\n",
      "Step 18000, Loss: 151.136353\n",
      "Step 19000, Loss: 157.889664\n",
      "Step 20000, Loss: 149.112473\n",
      "Step 21000, Loss: 151.694885\n",
      "Step 22000, Loss: 153.153229\n",
      "Step 23000, Loss: 152.662323\n",
      "Step 24000, Loss: 150.556198\n",
      "Step 25000, Loss: 142.779984\n",
      "Step 26000, Loss: 148.985382\n",
      "Step 27000, Loss: 150.923401\n",
      "Step 28000, Loss: 161.761551\n",
      "Step 29000, Loss: 144.045578\n",
      "Step 30000, Loss: 151.272964\n"
     ]
    }
   ],
   "source": [
    "# Start Training\n",
    "# Start a new TF session\n",
    "sess = tf.Session()\n",
    "\n",
    "# Run the initializer\n",
    "sess.run(init)\n",
    "\n",
    "# Training\n",
    "for i in range(1, num_steps+1):\n",
    "    # Prepare Data\n",
    "    # Get the next batch of MNIST data (only images are needed, not labels)\n",
    "    batch_x, _ = mnist.train.next_batch(batch_size)\n",
    "\n",
    "    # Train\n",
    "    feed_dict = {input_image: batch_x}\n",
    "    _, l = sess.run([train_op, loss_op], feed_dict=feed_dict)\n",
    "    if i % 1000 == 0 or i == 1:\n",
    "        print('Step %i, Loss: %f' % (i, l))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAeoAAAHhCAYAAAChqv35AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAIABJREFUeJzsvXlw3Od93//a+97FYnexABb3fRAgCJLgTVG8RFmyFVuJ\nLcmWPEk7cafJ1NNp/0jaOk2n/aOJc007aWeayS+u4+gY1ZIsV5JpSyZFijd4ACBAEPd9LnYXu4u9\nj98fzPMUlHBSUqO4+57hiKCAz37xfJ/n+Vzvz+ejyGaz5JBDDjnkkEMOX0wo/6EfIIcccsghhxxy\nWB85RZ1DDjnkkEMOX2DkFHUOOeSQQw45fIGRU9Q55JBDDjnk8AVGTlHnkEMOOeSQwxcYOUWdQw45\n5JBDDl9gfG6KWqFQnFEoFPcVCsWQQqH4vc/rc3LIIYcccsjhVxmKz6OOWqFQqIAB4BQwBdwAns9m\ns32f+YflkEMOOeSQw68wPi+PugMYymazI9lsNgG8CjzzOX1WDjnkkEMOOfzKQv05yfUAk6u+ngL2\nrffNCoUi1x4thxxyyCGH/9fgzWazrs2+6R+MTKZQKH5boVB0KhSKzn+oZ8jhVwcKheJzk/lZy14t\n9/N47hxyyOEfDca38k2fl0c9DZSu+rrk7/9NIpvN/g/gf8AnPWpxgYn8+WeVR1coFCiVSrRaLYlE\ngnQ6Lf999Wd8/OutQK1WS7nZbJZMJrOunO3IFs8sLvRMJvMJOY+6VmItMpmMlKtQKB5aF/GZ210P\no9GIQqHAYDCQTCZJpVIkEolPPL/4s51nLigooKqqinQ6TSAQYGZmhng8LuWk0+lty1UoFOh0OgwG\nA2fOnCGdTpPJZLh79y5TU1NyjZLJpFyf7chWKpVYrVaKi4tpbW1ldnaWubk5AoEA4XCYWCz20HvY\nrny1Wo3RaMTj8aBQKFhZWSGTybCyskIgENj2M6+GUqlEp9Oh1+sxGAwolUoikQiRSIRYLPbIcsWz\nq1QqVCoVOp2OTCZDPB6X6/9poVQqUavVaDQauQc/S16OeOZH2XMbQayLuEs+6ztQoVB8Qu6n/Qy1\nWi3lZLNZ+RmfVrZSqUSj
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `notebooks/3_NeuralNetworks/variational_autoencoder.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `notebooks/` directory, specifically within `notebooks/3_NeuralNetworks/`, this file is part of the notebook-based tutorials.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 10
- Code cells: 8
- Markdown cells: 2



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook notebooks/3_NeuralNetworks/variational_autoencoder.ipynb
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
- [gan.ipynb](gan.ipynb_docs.md)
- [neural_network.ipynb](neural_network.ipynb_docs.md)
- [neural_network_eager_api.ipynb](neural_network_eager_api.ipynb_docs.md)
- [neural_network_raw.ipynb](neural_network_raw.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

autoencoder, rnn, dataset, training, testing, neural network, gan, loss, mnist, sigmoid, tensorflow, optimizer, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
