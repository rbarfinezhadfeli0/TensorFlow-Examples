# Documentation: tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network.ipynb`
- **File Size**: 33214 bytes
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
    "Build a convolutional neural network with TensorFlow.\n",
    "\n",
    "This example is using TensorFlow layers API, see 'convolutional_network_raw' example\n",
    "for a raw TensorFlow implementation with variables.\n",
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
    "from __future__ import division, print_function, absolute_import\n",
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
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training Parameters\n",
    "learning_rate = 0.001\n",
    "num_steps = 2000\n",
    "batch_size = 128\n",
    "\n",
    "# Network Parameters\n",
    "num_input = 784 # MNIST data input (img shape: 28*28)\n",
    "num_classes = 10 # MNIST total classes (0-9 digits)\n",
    "dropout = 0.25 # Dropout, probability to drop a unit"
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
    "# Create the neural network\n",
    "def conv_net(x_dict, n_classes, dropout, reuse, is_training):\n",
    "    \n",
    "    # Define a scope for reusing the variables\n",
    "    with tf.variable_scope('ConvNet', reuse=reuse):\n",
    "        # TF Estimator input is a dict, in case of multiple inputs\n",
    "        x = x_dict['images']\n",
    "\n",
    "        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)\n",
    "        # Reshape to match picture format [Height x Width x Channel]\n",
    "        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]\n",
    "        x = tf.reshape(x, shape=[-1, 28, 28, 1])\n",
    "\n",
    "        # Convolution Layer with 32 filters and a kernel size of 5\n",
    "        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)\n",
    "        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2\n",
    "        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)\n",
    "\n",
    "        # Convolution Layer with 64 filters and a kernel size of 3\n",
    "        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)\n",
    "        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2\n",
    "        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)\n",
    "\n",
    "        # Flatten the data to a 1-D vector for the fully connected layer\n",
    "        fc1 = tf.contrib.layers.flatten(conv2)\n",
    "\n",
    "        # Fully connected layer (in tf contrib folder for now)\n",
    "        fc1 = tf.layers.dense(fc1, 1024)\n",
    "        # Apply Dropout (if is_training is False, dropout is not applied)\n",
    "        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)\n",
    "\n",
    "        # Output layer, class prediction\n",
    "        out = tf.layers.dense(fc1, n_classes)\n",
    "\n",
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
    "# Define the model function (following TF Estimator Template)\n",
    "def model_fn(features, labels, mode):\n",
    "    \n",
    "    # Build the neural network\n",
    "    # Because Dropout have different behavior at training and prediction time, we\n",
    "    # need to create 2 distinct computation graphs that still share the same weights.\n",
    "    logits_train = conv_net(features, num_classes, dropout, reuse=False, is_training=True)\n",
    "    logits_test = conv_net(features, num_classes, dropout, reuse=True, is_training=False)\n",
    "    \n",
    "    # Predictions\n",
    "    pred_classes = tf.argmax(logits_test, axis=1)\n",
    "    pred_probas = tf.nn.softmax(logits_test)\n",
    "    \n",
    "    # If prediction mode, early return\n",
    "    if mode == tf.estimator.ModeKeys.PREDICT:\n",
    "        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes) \n",
    "        \n",
    "    # Define loss and optimizer\n",
    "    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(\n",
    "        logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))\n",
    "    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\n",
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
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Using default config.\n",
      "WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmpdhd6F4\n",
      "INFO:tensorflow:Using config: {'_save_checkpoints_secs': 600, '_session_config': None, '_keep_checkpoint_max': 5, '_tf_random_seed': 1, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_save_checkpoints_steps': None, '_model_dir': '/tmp/tmpdhd6F4', '_save_summary_steps': 100}\n"
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
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Create CheckpointSaverHook.\n",
      "INFO:tensorflow:Saving checkpoints for 1 into /tmp/tmpdhd6F4/model.ckpt.\n",
      "INFO:tensorflow:loss = 2.39026, step = 1\n",
      "INFO:tensorflow:global_step/sec: 238.314\n",
      "INFO:tensorflow:loss = 0.237997, step = 101 (0.421 sec)\n",
      "INFO:tensorflow:global_step/sec: 255.312\n",
      "INFO:tensorflow:loss = 0.0954537, step = 201 (0.392 sec)\n",
      "INFO:tensorflow:global_step/sec: 257.194\n",
      "INFO:tensorflow:loss = 0.121477, step = 301 (0.389 sec)\n",
      "INFO:tensorflow:global_step/sec: 255.018\n",
      "INFO:tensorflow:loss = 0.0539927, step = 401 (0.392 sec)\n",
      "INFO:tensorflow:global_step/sec: 254.293\n",
      "INFO:tensorflow:loss = 0.0440369, step = 501 (0.393 sec)\n",
      "INFO:tensorflow:global_step/sec: 256.501\n",
      "INFO:tensorflow:loss = 0.0247431, step = 601 (0.390 sec)\n",
      "INFO:tensorflow:global_step/sec: 252.956\n",
      "INFO:tensorflow:loss = 0.0738082, step = 701 (0.395 sec)\n",
      "INFO:tensorflow:global_step/sec: 253.222\n",
      "INFO:tensorflow:loss = 0.134998, step = 801 (0.395 sec)\n",
      "INFO:tensorflow:global_step/sec: 255.606\n",
      "INFO:tensorflow:loss = 0.00438448, step = 901 (0.391 sec)\n",
      "INFO:tensorflow:global_step/sec: 256.306\n",
      "INFO:tensorflow:loss = 0.0471991, step = 1001 (0.390 sec)\n",
      "INFO:tensorflow:global_step/sec: 255.352\n",
      "INFO:tensorflow:loss = 0.0371172, step = 1101 (0.392 sec)\n",
      "INFO:tensorflow:global_step/sec: 253.277\n",
      "INFO:tensorflow:loss = 0.0129522, step = 1201 (0.395 sec)\n",
      "INFO:tensorflow:global_step/sec: 252.49\n",
      "INFO:tensorflow:loss = 0.039862, step = 1301 (0.396 sec)\n",
      "INFO:tensorflow:global_step/sec: 253.902\n",
      "INFO:tensorflow:loss = 0.0520571, step = 1401 (0.394 sec)\n",
      "INFO:tensorflow:global_step/sec: 255.572\n",
      "INFO:tensorflow:loss = 0.0307549, step = 1501 (0.392 sec)\n",
      "INFO:tensorflow:global_step/sec: 254.32\n",
      "INFO:tensorflow:loss = 0.0108862, step = 1601 (0.393 sec)\n",
      "INFO:tensorflow:global_step/sec: 255.62\n",
      "INFO:tensorflow:loss = 0.0294434, step = 1701 (0.391 sec)\n",
      "INFO:tensorflow:global_step/sec: 254.349\n",
      "INFO:tensorflow:loss = 0.0179781, step = 1801 (0.393 sec)\n",
      "INFO:tensorflow:global_step/sec: 255.508\n",
      "INFO:tensorflow:loss = 0.0375271, step = 1901 (0.391 sec)\n",
      "INFO:tensorflow:Saving checkpoints for 2000 into /tmp/t
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/notebooks/3_NeuralNetworks/`, this file is part of the TensorFlow 1.x examples collection.



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
jupyter notebook tensorflow_v1/notebooks/3_NeuralNetworks/convolutional_network.ipynb
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

activation, rnn, dataset, training, testing, relu, neural network, accuracy, convolution, loss, mnist, softmax, tensorflow, optimizer, pooling, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
