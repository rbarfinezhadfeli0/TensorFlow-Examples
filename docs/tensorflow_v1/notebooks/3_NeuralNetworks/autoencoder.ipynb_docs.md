# Documentation: tensorflow_v1/notebooks/3_NeuralNetworks/autoencoder.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/3_NeuralNetworks/autoencoder.ipynb`
- **File Size**: 47239 bytes
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
    "Build a 2 layers auto-encoder with TensorFlow to compress images to a lower latent space and then reconstruct them.\n",
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
   "metadata": {},
   "outputs": [],
   "source": [
    "from __future__ import division, print_function, absolute_import\n",
    "\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt"
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
    "# Training Parameters\n",
    "learning_rate = 0.01\n",
    "num_steps = 30000\n",
    "batch_size = 256\n",
    "\n",
    "display_step = 1000\n",
    "examples_to_show = 10\n",
    "\n",
    "# Network Parameters\n",
    "num_hidden_1 = 256 # 1st layer num features\n",
    "num_hidden_2 = 128 # 2nd layer num features (the latent dim)\n",
    "num_input = 784 # MNIST data input (img shape: 28*28)\n",
    "\n",
    "# tf Graph input (only pictures)\n",
    "X = tf.placeholder(\"float\", [None, num_input])\n",
    "\n",
    "weights = {\n",
    "    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),\n",
    "    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),\n",
    "    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),\n",
    "    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),\n",
    "}\n",
    "biases = {\n",
    "    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),\n",
    "    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),\n",
    "    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),\n",
    "    'decoder_b2': tf.Variable(tf.random_normal([num_input])),\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Building the encoder\n",
    "def encoder(x):\n",
    "    # Encoder Hidden layer with sigmoid activation #1\n",
    "    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),\n",
    "                                   biases['encoder_b1']))\n",
    "    # Encoder Hidden layer with sigmoid activation #2\n",
    "    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),\n",
    "                                   biases['encoder_b2']))\n",
    "    return layer_2\n",
    "\n",
    "\n",
    "# Building the decoder\n",
    "def decoder(x):\n",
    "    # Decoder Hidden layer with sigmoid activation #1\n",
    "    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),\n",
    "                                   biases['decoder_b1']))\n",
    "    # Decoder Hidden layer with sigmoid activation #2\n",
    "    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),\n",
    "                                   biases['decoder_b2']))\n",
    "    return layer_2\n",
    "\n",
    "# Construct model\n",
    "encoder_op = encoder(X)\n",
    "decoder_op = decoder(encoder_op)\n",
    "\n",
    "# Prediction\n",
    "y_pred = decoder_op\n",
    "# Targets (Labels) are the input data.\n",
    "y_true = X\n",
    "\n",
    "# Define loss and optimizer, minimize the squared error\n",
    "loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))\n",
    "optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)\n",
    "\n",
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
      "Step 1: Minibatch Loss: 0.438300\n",
      "Step 1000: Minibatch Loss: 0.146586\n",
      "Step 2000: Minibatch Loss: 0.130722\n",
      "Step 3000: Minibatch Loss: 0.117178\n",
      "Step 4000: Minibatch Loss: 0.109027\n",
      "Step 5000: Minibatch Loss: 0.102582\n",
      "Step 6000: Minibatch Loss: 0.099183\n",
      "Step 7000: Minibatch Loss: 0.095619\n",
      "Step 8000: Minibatch Loss: 0.089006\n",
      "Step 9000: Minibatch Loss: 0.087125\n",
      "Step 10000: Minibatch Loss: 0.083930\n",
      "Step 11000: Minibatch Loss: 0.077512\n",
      "Step 12000: Minibatch Loss: 0.077137\n",
      "Step 13000: Minibatch Loss: 0.073983\n",
      "Step 14000: Minibatch Loss: 0.074218\n",
      "Step 15000: Minibatch Loss: 0.074492\n",
      "Step 16000: Minibatch Loss: 0.074374\n",
      "Step 17000: Minibatch Loss: 0.070909\n",
      "Step 18000: Minibatch Loss: 0.069438\n",
      "Step 19000: Minibatch Loss: 0.068245\n",
      "Step 20000: Minibatch Loss: 0.068402\n",
      "Step 21000: Minibatch Loss: 0.067113\n",
      "Step 22000: Minibatch Loss: 0.068241\n",
      "Step 23000: Minibatch Loss: 0.062454\n",
      "Step 24000: Minibatch Loss: 0.059754\n",
      "Step 25000: Minibatch Loss: 0.058687\n",
      "Step 26000: Minibatch Loss: 0.059107\n",
      "Step 27000: Minibatch Loss: 0.055788\n",
      "Step 28000: Minibatch Loss: 0.057263\n",
      "Step 29000: Minibatch Loss: 0.056391\n",
      "Step 30000: Minibatch Loss: 0.057672\n"
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
    "    # Run optimization op (backprop) and cost op (to get loss value)\n",
    "    _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})\n",
    "    # Display logs per step\n",
    "    if i % display_step == 0 or i == 1:\n",
    "        print('Step %i: Minibatch Loss: %f' % (i, l))"
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
      "Original Images\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAQUAAAD8CAYAAAB+fLH0AAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAAIABJREFUeJztnXncVfP2x9/rNpehUImiqJAIKRFyiyIpMjRchEiGhPqR\nOXNJZYgoJK5UMpTMjZdUV0o0qtuVikQq1FXi+/vjnPXsZ5/nOZ357HOe1vv1el6ns/c+e6+9z+n7\n/XzXd33XEucchmEYyt+CNsAwjNzCGgXDMHxYo2AYhg9rFAzD8GGNgmEYPqxRMAzDhzUKhmH4yEij\nICJnishyEVkpIv0ycQ3DMDKDpDt4SURKAV8DZwBrgc+ALs65JWm9kGEYGaF0Bs7ZFFjpnFsFICJj\ngQ5A1EZBRCys0jAyz0/OuaqxDsrE8OFAYE2h92vD23yISA8RmSci8zJgg2EYRVkdz0GZUApx4Zwb\nAYwAUwqGkUtkQimsA2oVel8zvM0wjDwgE43CZ0A9EakjImWBzsCkDFwnI/Tv35/+/fvjnPP9Gcbu\nQtqHD865nSJyPfABUAp4wTm3ON3XMQwjM2TEp+Ccexd4NxPnTjf9+/cHoEWLFgCcdtppvv1///vf\ns2xRdhgyZAgA3bp1A2DfffcN0hwjh7CIRsMwfAQ2+xA0qhDuueeeYvfPmDHD91pSaNSoEQAXX3wx\nAOPHjw/SHCMHMaVgGIaP3U4pxKsQ7r333ixZlF0uv/xyACpUqADAO++8E6Q5aaVKlSoATJ06FYCD\nDjoIgP322y8wmw444AAAPvnkEwAOPvhgAObNC8XsnXPOOQBs2LAhAOuKx5SCYRg+0r4gKikjshDR\nqLMK06dPL3a/KgNVEiWRk046qeD+v/76awCOP/54ALZv3x6YXamiMyfvv/8+4N3Ttm3bAGjVqhUA\nc+bMyZpNZcqUAeD+++8HoG/fvsUep899+PDhALz88suA9/3873//S6dZnzvnjo91kCkFwzB8lHif\ngikEjy5duhT0YNqr5rNCqFo1tOBv8uTJADRu3BigIAJVlUI2FYJy9dVXA9EVglK+fHkAbrzxRt/r\n7NmzAdi6dSsiAsAjjzwCeD6TTGFKwTAMHyVeKUSbZdidFIL2RurpBliyJH9z3lSrVg2AiRMnAtCk\nSRPf/kmTQkttdKYlCO66665it6uqufPOOwH4299C/fJZZ50FwBFHHAFAu3btAKhcuXKBUlB/Q6Yx\npWAYho8SqxTUhxC5lkHjEHYHhaD06dMHCM3br127FsjPSMbq1asDcN111wFwwgkn+PbPnDkTgEcf\nfRSATZs2ZdG6EDrzobERkbN7AwYMAGDRokW+7V9++aXvvc6oLF68uMB3ki1KXKOg/9kjGwMlHxY4\nlS4d+lq6du0KQKVKlQB4/fXXgcQDXZo2bVrwb/0xbt26NWU7s8WZZ54JeJL8xBNP9O1ftWoV4A2P\nfvvttyxa5+ftt98GvGHBX3/9BcCOHTsA+OWXX+I6T+3atYHQd6/nyhY2fDAMw0eJUgr9+/eP6ljM\nB4WgPPHEEwD07NkT8Hqbf//730D8SqFDhw4AtG7dumDba6+9ljY7M02NGjUAeOyxxwCoX7++b/+W\nLVsAbxovSIWg6HBBvzOdFr3hhhuA+B28DRo0AELh6Js3bwa8UOlMY0rBMAwfJUopaKKUwujUYz4s\ngT7yyCMBuOyyy3zbH3jgAQA+//zzhM6nCqFcuXIAbN68mSlTpqRoZfYYMWIEUFQhKLfccgvgTfMF\niS58Klu2rG/7ihUrABg1alRc51EHoya/AS/U+ZtvvknVzLgwpWAYho8SoRR2NeMQa+pRPxNttiKb\nU5fqPddgI2XChAkJnUeDey699FLf9sGDB7NmzZriP
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/3_NeuralNetworks/autoencoder.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/notebooks/3_NeuralNetworks/`, this file is part of the TensorFlow 1.x examples collection.



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
jupyter notebook tensorflow_v1/notebooks/3_NeuralNetworks/autoencoder.ipynb
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

- [bidirectional_rnn.ipynb](bidirectional_rnn.ipynb_docs.md)
- [convolutional_network.ipynb](convolutional_network.ipynb_docs.md)
- [convolutional_network_raw.ipynb](convolutional_network_raw.ipynb_docs.md)
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

activation, rnn, dataset, training, testing, gradient, gan, loss, mnist, sigmoid, tensorflow, optimizer, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
