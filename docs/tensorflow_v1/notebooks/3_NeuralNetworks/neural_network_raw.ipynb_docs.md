# Documentation: tensorflow_v1/notebooks/3_NeuralNetworks/neural_network_raw.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/3_NeuralNetworks/neural_network_raw.ipynb`
- **File Size**: 7146 bytes
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
    "# Neural Network Example\n",
    "\n",
    "Build a 2-hidden layers fully connected neural network (a.k.a multilayer perceptron) with TensorFlow.\n",
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
    "# Import MNIST data\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"/tmp/data/\", one_hot=True)\n",
    "\n",
    "import tensorflow as tf"
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
    "num_steps = 500\n",
    "batch_size = 128\n",
    "display_step = 100\n",
    "\n",
    "# Network Parameters\n",
    "n_hidden_1 = 256 # 1st layer number of neurons\n",
    "n_hidden_2 = 256 # 2nd layer number of neurons\n",
    "num_input = 784 # MNIST data input (img shape: 28*28)\n",
    "num_classes = 10 # MNIST total classes (0-9 digits)\n",
    "\n",
    "# tf Graph input\n",
    "X = tf.placeholder(\"float\", [None, num_input])\n",
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
    "# Store layers weight & bias\n",
    "weights = {\n",
    "    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),\n",
    "    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),\n",
    "    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))\n",
    "}\n",
    "biases = {\n",
    "    'b1': tf.Variable(tf.random_normal([n_hidden_1])),\n",
    "    'b2': tf.Variable(tf.random_normal([n_hidden_2])),\n",
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
    "# Create model\n",
    "def neural_net(x):\n",
    "    # Hidden fully connected layer with 256 neurons\n",
    "    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])\n",
    "    # Hidden fully connected layer with 256 neurons\n",
    "    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])\n",
    "    # Output fully connected layer with a neuron for each class\n",
    "    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']\n",
    "    return out_layer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Construct model\n",
    "logits = neural_net(X)\n",
    "\n",
    "# Define loss and optimizer\n",
    "loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\n",
    "    logits=logits, labels=Y))\n",
    "optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\n",
    "train_op = optimizer.minimize(loss_op)\n",
    "\n",
    "# Evaluate model (with test logits, for dropout to be disabled)\n",
    "correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))\n",
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
      "Step 1, Minibatch Loss= 13208.1406, Training Accuracy= 0.266\n",
      "Step 100, Minibatch Loss= 462.8610, Training Accuracy= 0.867\n",
      "Step 200, Minibatch Loss= 232.8298, Training Accuracy= 0.844\n",
      "Step 300, Minibatch Loss= 85.2141, Training Accuracy= 0.891\n",
      "Step 400, Minibatch Loss= 38.0552, Training Accuracy= 0.883\n",
      "Step 500, Minibatch Loss= 55.3689, Training Accuracy= 0.867\n",
      "Optimization Finished!\n",
      "Testing Accuracy: 0.8729\n"
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
    "    for step in range(1, num_steps+1):\n",
    "        batch_x, batch_y = mnist.train.next_batch(batch_size)\n",
    "        # Run optimization op (backprop)\n",
    "        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})\n",
    "        if step % display_step == 0 or step == 1:\n",
    "            # Calculate batch loss and accuracy\n",
    "            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,\n",
    "                                                                 Y: batch_y})\n",
    "            print(\"Step \" + str(step) + \", Minibatch Loss= \" + \\\n",
    "                  \"{:.4f}\".format(loss) + \", Training Accuracy= \" + \\\n",
    "                  \"{:.3f}\".format(acc))\n",
    "\n",
    "    print(\"Optimization Finished!\")\n",
    "\n",
    "    # Calculate accuracy for MNIST test images\n",
    "    print(\"Testing Accuracy:\", \\\n",
    "        sess.run(accuracy, feed_dict={X: mnist.test.images,\n",
    "                                      Y: mnist.test.labels}))"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}

```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/3_NeuralNetworks/neural_network_raw.ipynb` within the TensorFlow Examples repository.

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
jupyter notebook tensorflow_v1/notebooks/3_NeuralNetworks/neural_network_raw.ipynb
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
- [convolutional_network_raw.ipynb](convolutional_network_raw.ipynb_docs.md)
- [dcgan.ipynb](dcgan.ipynb_docs.md)
- [dynamic_rnn.ipynb](dynamic_rnn.ipynb_docs.md)
- [gan.ipynb](gan.ipynb_docs.md)
- [neural_network.ipynb](neural_network.ipynb_docs.md)
- [neural_network_eager_api.ipynb](neural_network_eager_api.ipynb_docs.md)
- [recurrent_network.ipynb](recurrent_network.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

dataset, training, testing, neural network, accuracy, loss, mnist, softmax, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
