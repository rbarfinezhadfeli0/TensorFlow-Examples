# Documentation: tensorflow_v1/notebooks/4_Utils/tensorboard_advanced.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/4_Utils/tensorboard_advanced.ipynb`
- **File Size**: 10238 bytes
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
    "# Tensorboard Advanced\n",
    "\n",
    "Advanced visualization using Tensorboard (weights, gradient, ...). This example is using the MNIST database of handwritten digits\n",
    "(http://yann.lecun.com/exdb/mnist/).\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
    "\n",
    "# Import MNIST data\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"/tmp/data/\", one_hot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Parameters\n",
    "learning_rate = 0.01\n",
    "training_epochs = 25\n",
    "batch_size = 100\n",
    "display_step = 1\n",
    "logs_path = '/tmp/tensorflow_logs/example/'\n",
    "\n",
    "# Network Parameters\n",
    "n_hidden_1 = 256 # 1st layer number of features\n",
    "n_hidden_2 = 256 # 2nd layer number of features\n",
    "n_input = 784 # MNIST data input (img shape: 28*28)\n",
    "n_classes = 10 # MNIST total classes (0-9 digits)\n",
    "\n",
    "# tf Graph Input\n",
    "# mnist data image of shape 28*28=784\n",
    "x = tf.placeholder(tf.float32, [None, 784], name='InputData')\n",
    "# 0-9 digits recognition => 10 classes\n",
    "y = tf.placeholder(tf.float32, [None, 10], name='LabelData')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Create model\n",
    "def multilayer_perceptron(x, weights, biases):\n",
    "    # Hidden layer with RELU activation\n",
    "    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])\n",
    "    layer_1 = tf.nn.relu(layer_1)\n",
    "    # Create a summary to visualize the first layer ReLU activation\n",
    "    tf.summary.histogram(\"relu1\", layer_1)\n",
    "    # Hidden layer with RELU activation\n",
    "    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])\n",
    "    layer_2 = tf.nn.relu(layer_2)\n",
    "    # Create another summary to visualize the second layer ReLU activation\n",
    "    tf.summary.histogram(\"relu2\", layer_2)\n",
    "    # Output layer\n",
    "    out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])\n",
    "    return out_layer\n",
    "\n",
    "# Store layers weight & bias\n",
    "weights = {\n",
    "    'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'),\n",
    "    'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'),\n",
    "    'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W3')\n",
    "}\n",
    "biases = {\n",
    "    'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'),\n",
    "    'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'),\n",
    "    'b3': tf.Variable(tf.random_normal([n_classes]), name='b3')\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Encapsulating all ops into scopes, making Tensorboard's Graph\n",
    "# Visualization more convenient\n",
    "with tf.name_scope('Model'):\n",
    "    # Build model\n",
    "    pred = multilayer_perceptron(x, weights, biases)\n",
    "\n",
    "with tf.name_scope('Loss'):\n",
    "    # Softmax Cross entropy (cost function)\n",
    "    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))\n",
    "\n",
    "with tf.name_scope('SGD'):\n",
    "    # Gradient Descent\n",
    "    optimizer = tf.train.GradientDescentOptimizer(learning_rate)\n",
    "    # Op to calculate every variable gradient\n",
    "    grads = tf.gradients(loss, tf.trainable_variables())\n",
    "    grads = list(zip(grads, tf.trainable_variables()))\n",
    "    # Op to update all variables according to their gradient\n",
    "    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)\n",
    "\n",
    "with tf.name_scope('Accuracy'):\n",
    "    # Accuracy\n",
    "    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))\n",
    "    acc = tf.reduce_mean(tf.cast(acc, tf.float32))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Initialize the variables (i.e. assign their default value)\n",
    "init = tf.global_variables_initializer()\n",
    "\n",
    "# Create a summary to monitor cost tensor\n",
    "tf.summary.scalar(\"loss\", loss)\n",
    "# Create a summary to monitor accuracy tensor\n",
    "tf.summary.scalar(\"accuracy\", acc)\n",
    "# Create summaries to visualize weights\n",
    "for var in tf.trainable_variables():\n",
    "    tf.summary.histogram(var.name, var)\n",
    "# Summarize all gradients\n",
    "for grad, var in grads:\n",
    "    tf.summary.histogram(var.name + '/gradient', grad)\n",
    "# Merge all summaries into a single op\n",
    "merged_summary_op = tf.summary.merge_all()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0001 cost= 59.570364205\n",
      "Epoch: 0002 cost= 13.585465186\n",
      "Epoch: 0003 cost= 8.379069252\n",
      "Epoch: 0004 cost= 6.005265894\n",
      "Epoch: 0005 cost= 4.498054792\n",
      "Epoch: 0006 cost= 3.503682522\n",
      "Epoch: 0007 cost= 2.822272765\n",
      "Epoch: 0008 cost= 2.306899852\n",
      "Epoch: 0009 cost= 1.912765543\n",
      "Epoch: 0010 cost= 1.597006118\n",
      "Epoch: 0011 cost= 1.330172869\n",
      "Epoch: 0012 cost= 1.142490618\n",
      "Epoch: 0013 cost= 0.939443911\n",
      "Epoch: 0014 cost= 0.820920588\n",
      "Epoch: 0015 cost= 0.702543302\n",
      "Epoch: 0016 cost= 0.604815631\n",
      "Epoch: 0017 cost= 0.505682561\n",
      "Epoch: 0018 cost= 0.439700446\n",
      "Epoch: 0019 cost= 0.378268929\n",
      "Epoch: 0020 cost= 0.299557848\n",
      "Epoch: 0021 cost= 0.269859066\n",
      "Epoch: 0022 cost= 0.230899029\n",
      "Epoch: 0023 cost= 0.183722090\n",
      "Epoch: 0024 cost= 0.164173368\n",
      "Epoch: 0025 cost= 0.142141250\n",
      "Optimization Finished!\n",
      "Accuracy: 0.9336\n",
      "Run the command line:\n",
      "--> tensorboard --logdir=/tmp/tensorflow_logs \n",
      "Then open http://0.0.0.0:6006/ into your web browser\n"
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
    "    # op to write logs to Tensorboard\n",
    "    summary_writer = tf.summary.FileWriter(logs_path,\n",
    "                                            graph=tf.get_default_graph())\n",
    "\n",
    "    # Training cycle\n",
    "    for epoch in range(training_epochs):\n",
    "        avg_cost = 0.\n",
    "        total_batch = int(mnist.train.num_examples/batch_size)\n",
    "        # Loop over all batches\n",
    "        for i in range(total_batch):\n",
    "            batch_xs, batch_ys = mnist.train.next_batch(batch_size)\n",
    "            # Run optimization op (backprop), cost op (to get loss value)\n",
    "            # and summary nodes\n",
    "            _, c, summary = sess.run([apply_grads, loss, merged_summary_op],\n",
    "                                     feed_dict={x: batch_xs, y: batch_ys})\n",
    "            # Write logs at every iteration\n",
    "            summary_writer.add_summary(summary, epoch * total_batch + i)\n",
    "            # Compute average loss\n",
    "            avg_cost += c / total_batch\n",
    "        # Display logs per epoch step\n",
    "        if (epoch+1) % display_step == 0:\n",
    "            print(\"Epoch:\", '%04d' % (epoch+1), \"cost=\", \"{:.9f}\".format(avg_cost))\n",
    "\n",
    "    print(\"Optimization Finished!\")\n",
    "\n",
    "    # Test model\n",
    "    # Calculate accuracy\n",
    "    print(\"Accuracy:\", acc.eval({x: mnist.test.images, y: mnist.test.labels}))\n",
    "\n",
    "    print(\"Run the command line:\\n\" \\\n",
    "          \"--> tensorboard --logdir=/tmp/tensorflow_logs \" \\\n",
    "          \"\\nThen open http://0.0.0.0:6006/ into your web browser\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Loss and Accuracy Visualization\n",
    "<img src=\"../../resources/img/tensorboard_advanced_1.png\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Computation Graph Visualization\n",
    "<img src=\"../../resources/img/tensorboard_advanced_2.png\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Weights and Gradients Visualization\n",
    "<img src=\"../../resources/img/tensorboard_advanced_3.png\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Activations Visualization\n",
    "<img src=\"../../resources/img/tensorboard_advanced_4.png\"/>"
   ]
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python [default]",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/4_Utils/tensorboard_advanced.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/notebooks/4_Utils/`, this file is part of the TensorFlow 1.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 11
- Code cells: 6
- Markdown cells: 5



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v1/notebooks/4_Utils/tensorboard_advanced.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It provides utility functions for TensorFlow operations.

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

- [save_restore_model.ipynb](save_restore_model.ipynb_docs.md)
- [tensorboard_basic.ipynb](tensorboard_basic.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

activation, training, gradient, relu, accuracy, loss, mnist, softmax, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
