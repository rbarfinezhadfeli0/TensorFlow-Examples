# Documentation: tensorflow_v2/notebooks/3_NeuralNetworks/dcgan.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/3_NeuralNetworks/dcgan.ipynb`
- **File Size**: 51453 bytes
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
    "# Deep Convolutional Generative Adversarial Network Example\n",
    "\n",
    "Build a deep convolutional generative adversarial network (DCGAN) to generate digit images from a noise distribution with TensorFlow v2.\n",
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
    "This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 255. \n",
    "\n",
    "In this example, each image will be converted to float32 and normalized from [0, 255] to [0, 1].\n",
    "\n",
    "![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)\n",
    "\n",
    "More info: http://yann.lecun.com/exdb/mnist/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
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
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# MNIST Dataset parameters.\n",
    "num_features = 784 # data features (img shape: 28*28).\n",
    "\n",
    "# Training parameters.\n",
    "lr_generator = 0.0002\n",
    "lr_discriminator = 0.0002\n",
    "training_steps = 20000\n",
    "batch_size = 128\n",
    "display_step = 500\n",
    "\n",
    "# Network parameters.\n",
    "noise_dim = 100 # Noise data points."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
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
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use tf.data API to shuffle and batch data.\n",
    "train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))\n",
    "train_data = train_data.repeat().shuffle(10000).batch(batch_size).prefetch(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create TF Model.\n",
    "class Generator(Model):\n",
    "    # Set layers.\n",
    "    def __init__(self):\n",
    "        super(Generator, self).__init__()\n",
    "        self.fc1 = layers.Dense(7 * 7 * 128)\n",
    "        self.bn1 = layers.BatchNormalization()\n",
    "        self.conv2tr1 = layers.Conv2DTranspose(64, 5, strides=2, padding='SAME')\n",
    "        self.bn2 = layers.BatchNormalization()\n",
    "        self.conv2tr2 = layers.Conv2DTranspose(1, 5, strides=2, padding='SAME')\n",
    "\n",
    "    # Set forward pass.\n",
    "    def call(self, x, is_training=False):\n",
    "        x = self.fc1(x)\n",
    "        x = self.bn1(x, training=is_training)\n",
    "        x = tf.nn.leaky_relu(x)\n",
    "        # Reshape to a 4-D array of images: (batch, height, width, channels)\n",
    "        # New shape: (batch, 7, 7, 128)\n",
    "        x = tf.reshape(x, shape=[-1, 7, 7, 128])\n",
    "        # Deconvolution, image shape: (batch, 14, 14, 64)\n",
    "        x = self.conv2tr1(x)\n",
    "        x = self.bn2(x, training=is_training)\n",
    "        x = tf.nn.leaky_relu(x)\n",
    "        # Deconvolution, image shape: (batch, 28, 28, 1)\n",
    "        x = self.conv2tr2(x)\n",
    "        x = tf.nn.tanh(x)\n",
    "        return x\n",
    "\n",
    "# Generator Network\n",
    "# Input: Noise, Output: Image\n",
    "# Note that batch normalization has different behavior at training and inference time,\n",
    "# we then use a placeholder to indicates the layer if we are training or not.\n",
    "class Discriminator(Model):\n",
    "    # Set layers.\n",
    "    def __init__(self):\n",
    "        super(Discriminator, self).__init__()\n",
    "        self.conv1 = layers.Conv2D(64, 5, strides=2, padding='SAME')\n",
    "        self.bn1 = layers.BatchNormalization()\n",
    "        self.conv2 = layers.Conv2D(128, 5, strides=2, padding='SAME')\n",
    "        self.bn2 = layers.BatchNormalization()\n",
    "        self.flatten = layers.Flatten()\n",
    "        self.fc1 = layers.Dense(1024)\n",
    "        self.bn3 = layers.BatchNormalization()\n",
    "        self.fc2 = layers.Dense(2)\n",
    "\n",
    "    # Set forward pass.\n",
    "    def call(self, x, is_training=False):\n",
    "        x = tf.reshape(x, [-1, 28, 28, 1])\n",
    "        x = self.conv1(x)\n",
    "        x = self.bn1(x, training=is_training)\n",
    "        x = tf.nn.leaky_relu(x)\n",
    "        x = self.conv2(x)\n",
    "        x = self.bn2(x, training=is_training)\n",
    "        x = tf.nn.leaky_relu(x)\n",
    "        x = self.flatten(x)\n",
    "        x = self.fc1(x)\n",
    "        x = self.bn3(x, training=is_training)\n",
    "        x = tf.nn.leaky_relu(x)\n",
    "        return self.fc2(x)\n",
    "\n",
    "# Build neural network model.\n",
    "generator = Generator()\n",
    "discriminator = Discriminator()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Losses.\n",
    "def generator_loss(reconstructed_image):\n",
    "    gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\n",
    "        logits=reconstructed_image, labels=tf.ones([batch_size], dtype=tf.int32)))\n",
    "    return gen_loss\n",
    "\n",
    "def discriminator_loss(disc_fake, disc_real):\n",
    "    disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\n",
    "        logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))\n",
    "    disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\n",
    "        logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))\n",
    "    return disc_loss_real + disc_loss_fake\n",
    "\n",
    "# Optimizers.\n",
    "optimizer_gen = tf.optimizers.Adam(learning_rate=lr_generator)#, beta_1=0.5, beta_2=0.999)\n",
    "optimizer_disc = tf.optimizers.Adam(learning_rate=lr_discriminator)#, beta_1=0.5, beta_2=0.999)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optimization process. Inputs: real image and noise.\n",
    "def run_optimization(real_images):\n",
    "    \n",
    "    # Rescale to [-1, 1], the input range of the discriminator\n",
    "    real_images = real_images * 2. - 1.\n",
    "\n",
    "    # Generate noise.\n",
    "    noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)\n",
    "    \n",
    "    with tf.GradientTape() as g:\n",
    "            \n",
    "        fake_images = generator(noise, is_training=True)\n",
    "        disc_fake = discriminator(fake_images, is_training=True)\n",
    "        disc_real = discriminator(real_images, is_training=True)\n",
    "\n",
    "        disc_loss = discriminator_loss(disc_fake, disc_real)\n",
    "            \n",
    "    # Training Variables for each optimizer\n",
    "    gradients_disc = g.gradient(disc_loss,  discriminator.trainable_variables)\n",
    "    optimizer_disc.apply_gradients(zip(gradients_disc,  discriminator.trainable_variables))\n",
    "    \n",
    "    # Generate noise.\n",
    "    noise = np.random.normal(-1., 1., size=[batch_size, noise_dim]).astype(np.float32)\n",
    "    \n",
    "    with tf.GradientTape() as g:\n",
    "            \n",
    "        fake_images = generator(noise, is_training=True)\n",
    "        disc_fake = discriminator(fake_images, is_training=True)\n",
    "\n",
    "        gen_loss = generator_loss(disc_fake)\n",
    "            \n",
    "    gradients_gen = g.gradient(gen_loss, generator.trainable_variables)\n",
    "    optimizer_gen.apply_gradients(zip(gradients_gen, generator.trainable_variables))\n",
    "    \n",
    "    return gen_loss, disc_loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "initial: gen_loss: 0.694535, disc_loss: 1.403878\n",
      "step: 500, gen_loss: 2.154825, disc_loss: 0.429040\n",
      "step: 1000, gen_loss: 2.103464, disc_loss: 0.502638\n",
      "step: 1500, gen_loss: 2.282369, disc_loss: 0.572065\n",
      "step: 2000, gen_loss: 2.7
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/3_NeuralNetworks/dcgan.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/3_NeuralNetworks/`, this file is part of the TensorFlow 2.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 12
- Code cells: 10
- Markdown cells: 2



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v2/notebooks/3_NeuralNetworks/dcgan.ipynb
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
- [convolutional_network.ipynb](convolutional_network.ipynb_docs.md)
- [convolutional_network_raw.ipynb](convolutional_network_raw.ipynb_docs.md)
- [dynamic_rnn.ipynb](dynamic_rnn.ipynb_docs.md)
- [neural_network.ipynb](neural_network.ipynb_docs.md)
- [neural_network_raw.ipynb](neural_network_raw.ipynb_docs.md)
- [recurrent_network.ipynb](recurrent_network.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

rnn, dataset, training, testing, forward pass, gradient, relu, neural network, convolution, gan, loss, mnist, softmax, tensorflow, optimizer, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
