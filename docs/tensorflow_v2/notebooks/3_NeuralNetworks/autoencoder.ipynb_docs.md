# Documentation: tensorflow_v2/notebooks/3_NeuralNetworks/autoencoder.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/3_NeuralNetworks/autoencoder.ipynb`
- **File Size**: 44104 bytes
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
    "# Auto-Encoder Example\n",
    "\n",
    "Build a 2 layers auto-encoder with TensorFlow v2 to compress images to a lower latent space and then reconstruct them.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Auto-Encoder Overview\n",
    "\n",
    "<img src=\"http://kvfrans.com/content/images/2016/08/autoenc.jpg\" alt=\"ae\" style=\"width: 800px;\"/>\n",
    "\n",
    "References:\n",
    "- [Gradient-based learning applied to document recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf). Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Proceedings of the IEEE, 86(11):2278-2324, November 1998.\n",
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
    "# MNIST Dataset parameters.\n",
    "num_features = 784 # data features (img shape: 28*28).\n",
    "\n",
    "# Training parameters.\n",
    "learning_rate = 0.01\n",
    "training_steps = 20000\n",
    "batch_size = 256\n",
    "display_step = 1000\n",
    "\n",
    "# Network Parameters\n",
    "num_hidden_1 = 128 # 1st layer num features.\n",
    "num_hidden_2 = 64 # 2nd layer num features (the latent dim)."
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
    "x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)\n",
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
    "train_data = train_data.repeat().shuffle(10000).batch(batch_size).prefetch(1)\n",
    "\n",
    "test_data = tf.data.Dataset.from_tensor_slices((x_test, y_test))\n",
    "test_data = test_data.repeat().batch(batch_size).prefetch(1)"
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
    "    'encoder_h1': tf.Variable(random_normal([num_features, num_hidden_1])),\n",
    "    'encoder_h2': tf.Variable(random_normal([num_hidden_1, num_hidden_2])),\n",
    "    'decoder_h1': tf.Variable(random_normal([num_hidden_2, num_hidden_1])),\n",
    "    'decoder_h2': tf.Variable(random_normal([num_hidden_1, num_features])),\n",
    "}\n",
    "biases = {\n",
    "    'encoder_b1': tf.Variable(random_normal([num_hidden_1])),\n",
    "    'encoder_b2': tf.Variable(random_normal([num_hidden_2])),\n",
    "    'decoder_b1': tf.Variable(random_normal([num_hidden_1])),\n",
    "    'decoder_b2': tf.Variable(random_normal([num_features])),\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Building the encoder.\n",
    "def encoder(x):\n",
    "    # Encoder Hidden layer with sigmoid activation.\n",
    "    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),\n",
    "                                   biases['encoder_b1']))\n",
    "    # Encoder Hidden layer with sigmoid activation.\n",
    "    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),\n",
    "                                   biases['encoder_b2']))\n",
    "    return layer_2\n",
    "\n",
    "\n",
    "# Building the decoder.\n",
    "def decoder(x):\n",
    "    # Decoder Hidden layer with sigmoid activation.\n",
    "    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),\n",
    "                                   biases['decoder_b1']))\n",
    "    # Decoder Hidden layer with sigmoid activation.\n",
    "    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),\n",
    "                                   biases['decoder_b2']))\n",
    "    return layer_2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mean square loss between original images and reconstructed ones.\n",
    "def mean_square(reconstructed, original):\n",
    "    return tf.reduce_mean(tf.pow(original - reconstructed, 2))\n",
    "\n",
    "# Adam optimizer.\n",
    "optimizer = tf.optimizers.Adam(learning_rate=learning_rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optimization process. \n",
    "def run_optimization(x):\n",
    "    # Wrap computation inside a GradientTape for automatic differentiation.\n",
    "    with tf.GradientTape() as g:\n",
    "        reconstructed_image = decoder(encoder(x))\n",
    "        loss = mean_square(reconstructed_image, x)\n",
    "\n",
    "    # Variables to update, i.e. trainable variables.\n",
    "    trainable_variables = weights.values() + biases.values()\n",
    "    \n",
    "    # Compute gradients.\n",
    "    gradients = g.gradient(loss, trainable_variables)\n",
    "    \n",
    "    # Update W and b following gradients.\n",
    "    optimizer.apply_gradients(zip(gradients, trainable_variables))\n",
    "    \n",
    "    return loss"
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
      "step: 0, loss: 0.234978\n",
      "step: 1000, loss: 0.014881\n",
      "step: 2000, loss: 0.010402\n",
      "step: 3000, loss: 0.008817\n",
      "step: 4000, loss: 0.007337\n",
      "step: 5000, loss: 0.006399\n",
      "step: 6000, loss: 0.006039\n",
      "step: 7000, loss: 0.005042\n",
      "step: 8000, loss: 0.005235\n",
      "step: 9000, loss: 0.004838\n",
      "step: 10000, loss: 0.004552\n",
      "step: 11000, loss: 0.004717\n",
      "step: 12000, loss: 0.004550\n",
      "step: 13000, loss: 0.004633\n",
      "step: 14000, loss: 0.004469\n",
      "step: 15000, loss: 0.004503\n",
      "step: 16000, loss: 0.003971\n",
      "step: 17000, loss: 0.004258\n",
      "step: 18000, loss: 0.004012\n",
      "step: 19000, loss: 0.003703\n",
      "step: 20000, loss: 0.003933\n"
     ]
    }
   ],
   "source": [
    "# Run training for the given number of steps.\n",
    "for step, (batch_x, _) in enumerate(train_data.take(training_steps + 1)):\n",
    "    \n",
    "    # Run the optimization.\n",
    "    loss = run_optimization(batch_x)\n",
    "    \n",
    "    if step % display_step == 0:\n",
    "        print(\"step: %i, loss: %f\" % (step, loss))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Testing and Visualization.\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original Images\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAQUAAAD8CAYAAAB+fLH0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvhp/UCwAAIABJREFUeJztnXm8VXP3x99LKTI0UPQkMmVMKZSpkAwViVIUCWUuT6YKjzKVKeFJZCxTkilj0pOZKDSX0oMuKfWQRL/E9/fHOevuu/c9p3Pvmfa+567363Ve557v2WefdfY59/v9fNd3fdcS5xyGYRjKZmEbYBhGtLBOwTAMH9YpGIbhwzoFwzB8WKdgGIYP6xQMw/BhnYJhGD5y0imIyAkiskhElojIwFy8h2EYuUGyHbwkIlWAr4B2QBHwGXCGc25+Vt/IMIycUDUH5zwEWOKcWwogIuOBTkDSTkFELKzSMHLPKudc3VQH5WL60ABYVuJxUbzNh4j0FZEZIjIjBzYYhlGab8tyUC6UgiRoK6UEnHNjgDFgSsEwokQulEIR0LDE452AH3LwPoZh5IBcdAqfAXuKyK4iUg3oDkzKwfsYhpEDsj59cM5tFJFLgclAFeBR59y8bL+PYRi5IetLkmkZYT6FMnHllVcCsOWWWwJwwAEHANClSxffcaNHjwbg448/BuCJJ57Il4lGtJnpnDso1UEW0WgYhg9TChWAZ599FiitCFLx9ddfA3DssccC8N1332XXsJBo3LgxAAsXLgSgf//+ANx3332h2VRWttpqKwDuuOMOAC644AIAZs6cCUDXrl0B+PbbMq0elhdTCoZhlJ9cxCkYWSKVQtCRcvLkyQDstttuAJx00kkA7L777gD06NEDgGHDhuXO2Dxy4IEHAvD3338DUFRUFKY55aJ+/foA9OnTB/A+Q4sWLQDo2LEjAKNGjQrBuhimFAzD8GFKIYIcdFBs2te5c2df+7x5sZXdk08+GYBVq1YB8NtvvwFQrVo1AD755BMAmjZtCsB2222XY4vzS7NmzQBYt24dAC+++GKY5pSJunVjWw7Gjh0bsiWpMaVgGIaPCqkUdI6t87IffohFUa9fv56nnnoKgB9//BGAJUuWhGBhZui8UyS2jUQVwvHHHw/A8uXLE77uiiuuAGDffff1tb/22ms5sTPf7L///gBceumlQMWIv+jXrx8Ap5xyCgCHHHLIJo9v3bo1AJttFhuvZ82axXvvvZdDC0tjSsEwDB8VMk5h6dKlADRq1CjpMWvXrgW8UTZd1LN9++23AzBjRv52eu+yyy6A91n+97//bfL4WbNmAd6IqmicwrRp07JtYl5RhThhwgQAjj76aAD
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/3_NeuralNetworks/autoencoder.ipynb` within the TensorFlow Examples repository.

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
jupyter notebook tensorflow_v2/notebooks/3_NeuralNetworks/autoencoder.ipynb
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

- [bidirectional_rnn.ipynb](bidirectional_rnn.ipynb_docs.md)
- [convolutional_network.ipynb](convolutional_network.ipynb_docs.md)
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

activation, rnn, dataset, training, testing, gradient, gan, loss, mnist, sigmoid, tensorflow, optimizer, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
