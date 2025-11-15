# Documentation: notebooks/3_NeuralNetworks/neural_network.ipynb

## File Metadata

- **File Path**: `notebooks/3_NeuralNetworks/neural_network.ipynb`
- **File Size**: 30679 bytes
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
    "Build a 2-hidden layers fully connected neural network (a.k.a multilayer perceptron) with TensorFlow.\n",
    "\n",
    "This example is using some of TensorFlow higher-level wrappers (tf.estimators, tf.layers, tf.metrics, ...), you can check 'neural_network_raw' example for a raw, and more detailed TensorFlow implementation.\n",
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
    "# Import MNIST data\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"/tmp/data/\", one_hot=False)\n",
    "\n",
    "import tensorflow as tf\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np"
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
    "# Parameters\n",
    "learning_rate = 0.1\n",
    "num_steps = 1000\n",
    "batch_size = 128\n",
    "display_step = 100\n",
    "\n",
    "# Network Parameters\n",
    "n_hidden_1 = 256 # 1st layer number of neurons\n",
    "n_hidden_2 = 256 # 2nd layer number of neurons\n",
    "num_input = 784 # MNIST data input (img shape: 28*28)\n",
    "num_classes = 10 # MNIST total classes (0-9 digits)"
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
    "# Define the input function for training\n",
    "input_fn = tf.estimator.inputs.numpy_input_fn(\n",
    "    x={'images': mnist.train.images}, y=mnist.train.labels,\n",
    "    batch_size=batch_size, num_epochs=None, shuffle=True)"
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
    "# Define the neural network\n",
    "def neural_net(x_dict):\n",
    "    # TF Estimator input is a dict, in case of multiple inputs\n",
    "    x = x_dict['images']\n",
    "    # Hidden fully connected layer with 256 neurons\n",
    "    layer_1 = tf.layers.dense(x, n_hidden_1)\n",
    "    # Hidden fully connected layer with 256 neurons\n",
    "    layer_2 = tf.layers.dense(layer_1, n_hidden_2)\n",
    "    # Output fully connected layer with a neuron for each class\n",
    "    out_layer = tf.layers.dense(layer_2, num_classes)\n",
    "    return out_layer"
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
    "# Define the model function (following TF Estimator Template)\n",
    "def model_fn(features, labels, mode):\n",
    "    \n",
    "    # Build the neural network\n",
    "    logits = neural_net(features)\n",
    "    \n",
    "    # Predictions\n",
    "    pred_classes = tf.argmax(logits, axis=1)\n",
    "    pred_probas = tf.nn.softmax(logits)\n",
    "    \n",
    "    # If prediction mode, early return\n",
    "    if mode == tf.estimator.ModeKeys.PREDICT:\n",
    "        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) \n",
    "        \n",
    "    # Define loss and optimizer\n",
    "    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\n",
    "        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))\n",
    "    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\n",
    "    train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())\n",
    "    \n",
    "    # Evaluate the accuracy of the model\n",
    "    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)\n",
    "    \n",
    "    # TF Estimators requires to return a EstimatorSpec, that specify\n",
    "    # the different ops for training, evaluating, ...\n",
    "    estim_specs = tf.estimator.EstimatorSpec(\n",
    "      mode=mode,\n",
    "      predictions=pred_classes,\n",
    "      loss=loss_op,\n",
    "      train_op=train_op,\n",
    "      eval_metric_ops={'accuracy': acc_op})\n",
    "\n",
    "    return estim_specs"
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
      "INFO:tensorflow:Using default config.\n",
      "WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpu7vjLA\n",
      "INFO:tensorflow:Using config: {'_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_tf_random_seed': 1, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_save_checkpoints_steps': None, '_model_dir': '/tmp/tmpu7vjLA', '_save_summary_steps': 100}\n"
     ]
    }
   ],
   "source": [
    "# Build the Estimator\n",
    "model = tf.estimator.Estimator(model_fn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Create CheckpointSaverHook.\n",
      "INFO:tensorflow:Saving checkpoints for 1 into /tmp/tmpu7vjLA/model.ckpt.\n",
      "INFO:tensorflow:loss = 2.44919, step = 1\n",
      "INFO:tensorflow:global_step/sec: 602.544\n",
      "INFO:tensorflow:loss = 0.344767, step = 101 (0.167 sec)\n",
      "INFO:tensorflow:global_step/sec: 618.839\n",
      "INFO:tensorflow:loss = 0.277633, step = 201 (0.162 sec)\n",
      "INFO:tensorflow:global_step/sec: 626.418\n",
      "INFO:tensorflow:loss = 0.407796, step = 301 (0.160 sec)\n",
      "INFO:tensorflow:global_step/sec: 624.765\n",
      "INFO:tensorflow:loss = 0.376889, step = 401 (0.160 sec)\n",
      "INFO:tensorflow:global_step/sec: 624.091\n",
      "INFO:tensorflow:loss = 0.319697, step = 501 (0.160 sec)\n",
      "INFO:tensorflow:global_step/sec: 616.907\n",
      "INFO:tensorflow:loss = 0.39049, step = 601 (0.162 sec)\n",
      "INFO:tensorflow:global_step/sec: 623.371\n",
      "INFO:tensorflow:loss = 0.336831, step = 701 (0.161 sec)\n",
      "INFO:tensorflow:global_step/sec: 617.429\n",
      "INFO:tensorflow:loss = 0.312776, step = 801 (0.162 sec)\n",
      "INFO:tensorflow:global_step/sec: 620.825\n",
      "INFO:tensorflow:loss = 0.312817, step = 901 (0.161 sec)\n",
      "INFO:tensorflow:Saving checkpoints for 1000 into /tmp/tmpu7vjLA/model.ckpt.\n",
      "INFO:tensorflow:Loss for final step: 0.24931.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<tensorflow.python.estimator.estimator.Estimator at 0x7f21ac597690>"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Train the Model\n",
    "model.train(input_fn, steps=num_steps)"
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
      "INFO:tensorflow:Starting evaluation at 2017-08-21-13:57:02\n",
      "INFO:tensorflow:Restoring parameters from /tmp/tmpu7vjLA/model.ckpt-1000\n",
      "INFO:tensorflow:Finished evaluation at 2017-08-21-13:57:02\n",
      "INFO:tensorflow:Saving dict for global step 1000: accuracy = 0.9189, global_step = 1000, loss = 0.286567\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "{'accuracy': 0.91890001, 'global_step': 1000, 'loss': 0.28656715}"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Evaluate the Model\n",
    "# Define the input function for evaluating\n",
    "input_fn = tf.estimator.inputs.numpy_input_fn(\n",
    "    x={'images': mnist.test.images}, y=mnist.test.labels,\n",
    "    batch_size=batch_size, shuffle=False)\n",
    "# Use the Estimator 'evaluate' method\n",
    "model.evaluate(input_fn)"
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
      "INFO:tensorflow:Restoring parameters from /tmp/tmpu7vjLA/model.ckpt-1000\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAP8AAAD8CAYAAAC4nHJkAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAALEgAACxIB0t1+/AAADO5JREFUeJzt3V2IXfW5x/Hf76QpiOlFYjUMNpqeogerSKKjCMYS9Vhy\nYiEWg9SLkkLJ9CJKCyVU7EVzWaQv1JvAlIbGkmMrpNUoYmNjMQ1qcSJqEmNiElIzMW9lhCaCtNGn\nF7Nsp3H2f+/st7XH5/uBYfZez3p52Mxv1lp77bX/jggByOe/6m4AQD0IP5AU4QeSIvxAUoQfSIrw\nA0kRfiApwg8kRfiBpD7Vz43Z5uOEQI9FhFuZr6M9v+1ltvfZPmD7gU7WBaC/3O5n+23PkrRf0h2S\nxiW9LOneiHijsAx7fqDH+rHnv1HSgYg4FBF/l/RrSSs6
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `notebooks/3_NeuralNetworks/neural_network.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `notebooks/` directory, specifically within `notebooks/3_NeuralNetworks/`, this file is part of the notebook-based tutorials.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 11
- Code cells: 9
- Markdown cells: 2



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook notebooks/3_NeuralNetworks/neural_network.ipynb
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
- **File Permissions**: Ensure saved models and checkpoints have appropriate access controls



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
- [neural_network_eager_api.ipynb](neural_network_eager_api.ipynb_docs.md)
- [neural_network_raw.ipynb](neural_network_raw.ipynb_docs.md)
- [recurrent_network.ipynb](recurrent_network.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

rnn, dataset, training, testing, gradient, relu, neural network, accuracy, loss, mnist, softmax, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
