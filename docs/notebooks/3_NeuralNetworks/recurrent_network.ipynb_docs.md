# Documentation: notebooks/3_NeuralNetworks/recurrent_network.ipynb

## File Metadata

- **File Path**: `notebooks/3_NeuralNetworks/recurrent_network.ipynb`
- **File Size**: 11292 bytes
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
    "# Recurrent Neural Network Example\n",
    "\n",
    "Build a recurrent neural network (LSTM) with TensorFlow.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## RNN Overview\n",
    "\n",
    "<img src=\"http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png\" alt=\"nn\" style=\"width: 600px;\"/>\n",
    "\n",
    "References:\n",
    "- [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf), Sepp Hochreiter & Jurgen Schmidhuber, Neural Computation 9(8): 1735-1780, 1997.\n",
    "\n",
    "## MNIST Dataset Overview\n",
    "\n",
    "This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 1. For simplicity, each image has been flattened and converted to a 1-D numpy array of 784 features (28*28).\n",
    "\n",
    "![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)\n",
    "\n",
    "To classify images using a recurrent neural network, we consider every image row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then handle 28 sequences of 28 timesteps for every sample.\n",
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
    "from __future__ import print_function\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.contrib import rnn\n",
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
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Training Parameters\n",
    "learning_rate = 0.001\n",
    "training_steps = 10000\n",
    "batch_size = 128\n",
    "display_step = 200\n",
    "\n",
    "# Network Parameters\n",
    "num_input = 28 # MNIST data input (img shape: 28*28)\n",
    "timesteps = 28 # timesteps\n",
    "num_hidden = 128 # hidden layer num of features\n",
    "num_classes = 10 # MNIST total classes (0-9 digits)\n",
    "\n",
    "# tf Graph input\n",
    "X = tf.placeholder(\"float\", [None, timesteps, num_input])\n",
    "Y = tf.placeholder(\"float\", [None, num_classes])"
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
    "# Define weights\n",
    "weights = {\n",
    "    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))\n",
    "}\n",
    "biases = {\n",
    "    'out': tf.Variable(tf.random_normal([num_classes]))\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "def RNN(x, weights, biases):\n",
    "\n",
    "    # Prepare data shape to match `rnn` function requirements\n",
    "    # Current data input shape: (batch_size, timesteps, n_input)\n",
    "    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)\n",
    "\n",
    "    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)\n",
    "    x = tf.unstack(x, timesteps, 1)\n",
    "\n",
    "    # Define a lstm cell with tensorflow\n",
    "    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)\n",
    "\n",
    "    # Get lstm cell output\n",
    "    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)\n",
    "\n",
    "    # Linear activation, using rnn inner loop last output\n",
    "    return tf.matmul(outputs[-1], weights['out']) + biases['out']"
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
    "logits = RNN(X, weights, biases)\n",
    "prediction = tf.nn.softmax(logits)\n",
    "\n",
    "# Define loss and optimizer\n",
    "loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\n",
    "    logits=logits, labels=Y))\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\n",
    "train_op = optimizer.minimize(loss_op)\n",
    "\n",
    "# Evaluate model (with test logits, for dropout to be disabled)\n",
    "correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))\n",
    "accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))\n",
    "\n",
    "# Initialize the variables (i.e. assign their default value)\n",
    "init = tf.global_variables_initializer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Step 1, Minibatch Loss= 2.6268, Training Accuracy= 0.102\n",
      "Step 200, Minibatch Loss= 2.0722, Training Accuracy= 0.328\n",
      "Step 400, Minibatch Loss= 1.9181, Training Accuracy= 0.336\n",
      "Step 600, Minibatch Loss= 1.8858, Training Accuracy= 0.336\n",
      "Step 800, Minibatch Loss= 1.7022, Training Accuracy= 0.422\n",
      "Step 1000, Minibatch Loss= 1.6365, Training Accuracy= 0.477\n",
      "Step 1200, Minibatch Loss= 1.6691, Training Accuracy= 0.516\n",
      "Step 1400, Minibatch Loss= 1.4626, Training Accuracy= 0.547\n",
      "Step 1600, Minibatch Loss= 1.4707, Training Accuracy= 0.539\n",
      "Step 1800, Minibatch Loss= 1.4087, Training Accuracy= 0.570\n",
      "Step 2000, Minibatch Loss= 1.3033, Training Accuracy= 0.570\n",
      "Step 2200, Minibatch Loss= 1.3773, Training Accuracy= 0.508\n",
      "Step 2400, Minibatch Loss= 1.3092, Training Accuracy= 0.570\n",
      "Step 2600, Minibatch Loss= 1.2272, Training Accuracy= 0.609\n",
      "Step 2800, Minibatch Loss= 1.1827, Training Accuracy= 0.633\n",
      "Step 3000, Minibatch Loss= 1.0453, Training Accuracy= 0.641\n",
      "Step 3200, Minibatch Loss= 1.0400, Training Accuracy= 0.648\n",
      "Step 3400, Minibatch Loss= 1.1145, Training Accuracy= 0.656\n",
      "Step 3600, Minibatch Loss= 0.9884, Training Accuracy= 0.688\n",
      "Step 3800, Minibatch Loss= 1.0395, Training Accuracy= 0.703\n",
      "Step 4000, Minibatch Loss= 1.0096, Training Accuracy= 0.664\n",
      "Step 4200, Minibatch Loss= 0.8806, Training Accuracy= 0.758\n",
      "Step 4400, Minibatch Loss= 0.9090, Training Accuracy= 0.766\n",
      "Step 4600, Minibatch Loss= 1.0060, Training Accuracy= 0.703\n",
      "Step 4800, Minibatch Loss= 0.8954, Training Accuracy= 0.703\n",
      "Step 5000, Minibatch Loss= 0.8163, Training Accuracy= 0.750\n",
      "Step 5200, Minibatch Loss= 0.7620, Training Accuracy= 0.773\n",
      "Step 5400, Minibatch Loss= 0.7388, Training Accuracy= 0.758\n",
      "Step 5600, Minibatch Loss= 0.7604, Training Accuracy= 0.695\n",
      "Step 5800, Minibatch Loss= 0.7459, Training Accuracy= 0.734\n",
      "Step 6000, Minibatch Loss= 0.7448, Training Accuracy= 0.734\n",
      "Step 6200, Minibatch Loss= 0.7208, Training Accuracy= 0.773\n",
      "Step 6400, Minibatch Loss= 0.6557, Training Accuracy= 0.773\n",
      "Step 6600, Minibatch Loss= 0.8616, Training Accuracy= 0.758\n",
      "Step 6800, Minibatch Loss= 0.6089, Training Accuracy= 0.773\n",
      "Step 7000, Minibatch Loss= 0.5020, Training Accuracy= 0.844\n",
      "Step 7200, Minibatch Loss= 0.5980, Training Accuracy= 0.812\n",
      "Step 7400, Minibatch Loss= 0.6786, Training Accuracy= 0.766\n",
      "Step 7600, Minibatch Loss= 0.4891, Training Accuracy= 0.859\n",
      "Step 7800, Minibatch Loss= 0.7042, Training Accuracy= 0.797\n",
      "Step 8000, Minibatch Loss= 0.4200, Training Accuracy= 0.859\n",
      "Step 8200, Minibatch Loss= 0.6442, Training Accuracy= 0.742\n",
      "Step 8400, Minibatch Loss= 0.5569, Training Accuracy= 0.828\n",
      "Step 8600, Minibatch Loss= 0.5838, Training Accuracy= 0.836\n",
      "Step 8800, Minibatch Loss= 0.5579, Training Accuracy= 0.812\n",
      "Step 9000, Minibatch Loss= 0.4337, Training Accuracy= 0.867\n",
      "Step 9200, Minibatch Loss= 0.4366, Training Accuracy= 0.844\n",
      "Step 9400, Minibatch Loss= 0.5051, Training Accuracy= 0.844\n",
      "Step 9600, Minibatch Loss= 0.5244, Training Accuracy= 0.805\n",
      "Step 9800, Minibatch Loss= 0.4932, Training Accuracy= 0.805\n",
      "Step 10000, Minibatch Loss= 0.4833, Training Accuracy= 0.852\n",
      "Optimization Finished!\n",
      "Testing Accuracy: 0.882812\n"
     ]
    }
   ],
   "source": [
    "# Start training\n",
    "with tf.Session() as sess:\n",
    "\n",
    "    # Run the initializer\n",
    "    sess.run(init)\n",
    "\n",
    "    for step in range(1, training_steps+1):\n",
    "        batch_x, batch_y = mnist.train.next_batch(batch_size)\n",
    "        # Reshape data to get 28 seq of 28 elements\n",
    "        batch_x = batch_x.reshape((batch_size, timesteps, num_input))\n",
    "        # Run optimization op (backprop)\n",
    "        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})\n",
    "        if step % display_step == 0 or step == 1:\n",
    "            # Calculate batch loss and accuracy\n",
    "            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,\n",
    "                                  
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `notebooks/3_NeuralNetworks/recurrent_network.ipynb` within the TensorFlow Examples repository.

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
jupyter notebook notebooks/3_NeuralNetworks/recurrent_network.ipynb
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

activation, rnn, dataset, training, testing, gradient, neural network, lstm, accuracy, loss, mnist, softmax, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
