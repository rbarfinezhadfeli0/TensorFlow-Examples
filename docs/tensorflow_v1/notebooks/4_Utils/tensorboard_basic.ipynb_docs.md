# Documentation: tensorflow_v1/notebooks/4_Utils/tensorboard_basic.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/4_Utils/tensorboard_basic.ipynb`
- **File Size**: 6959 bytes
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
    "# Tensorboard Basics\n",
    "\n",
    "Graph and Loss visualization using Tensorboard. This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/).\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
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
    "from __future__ import print_function\n",
    "\n",
    "import tensorflow as tf\n",
    "\n",
    "# Import MINST data\n",
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
    "display_epoch = 1\n",
    "logs_path = '/tmp/tensorflow_logs/example/'\n",
    "\n",
    "# tf Graph Input\n",
    "# mnist data image of shape 28*28=784\n",
    "x = tf.placeholder(tf.float32, [None, 784], name='InputData')\n",
    "# 0-9 digits recognition => 10 classes\n",
    "y = tf.placeholder(tf.float32, [None, 10], name='LabelData')\n",
    "\n",
    "# Set model weights\n",
    "W = tf.Variable(tf.zeros([784, 10]), name='Weights')\n",
    "b = tf.Variable(tf.zeros([10]), name='Bias')"
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
    "# Construct model and encapsulating all ops into scopes, making\n",
    "# Tensorboard's Graph visualization more convenient\n",
    "with tf.name_scope('Model'):\n",
    "    # Model\n",
    "    pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax\n",
    "with tf.name_scope('Loss'):\n",
    "    # Minimize error using cross entropy\n",
    "    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))\n",
    "with tf.name_scope('SGD'):\n",
    "    # Gradient Descent\n",
    "    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)\n",
    "with tf.name_scope('Accuracy'):\n",
    "    # Accuracy\n",
    "    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))\n",
    "    acc = tf.reduce_mean(tf.cast(acc, tf.float32))\n",
    "\n",
    "# Initializing the variables\n",
    "init = tf.global_variables_initializer()\n",
    "\n",
    "# Create a summary to monitor cost tensor\n",
    "tf.summary.scalar(\"loss\", cost)\n",
    "# Create a summary to monitor accuracy tensor\n",
    "tf.summary.scalar(\"accuracy\", acc)\n",
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
      "Epoch: 0001 cost= 1.182138961\n",
      "Epoch: 0002 cost= 0.664609327\n",
      "Epoch: 0003 cost= 0.552565036\n",
      "Epoch: 0004 cost= 0.498541865\n",
      "Epoch: 0005 cost= 0.465393374\n",
      "Epoch: 0006 cost= 0.442491178\n",
      "Epoch: 0007 cost= 0.425474149\n",
      "Epoch: 0008 cost= 0.412152022\n",
      "Epoch: 0009 cost= 0.401320939\n",
      "Epoch: 0010 cost= 0.392305281\n",
      "Epoch: 0011 cost= 0.384732356\n",
      "Epoch: 0012 cost= 0.378109478\n",
      "Epoch: 0013 cost= 0.372409370\n",
      "Epoch: 0014 cost= 0.367236996\n",
      "Epoch: 0015 cost= 0.362727492\n",
      "Epoch: 0016 cost= 0.358627345\n",
      "Epoch: 0017 cost= 0.354815522\n",
      "Epoch: 0018 cost= 0.351413656\n",
      "Epoch: 0019 cost= 0.348314827\n",
      "Epoch: 0020 cost= 0.345429416\n",
      "Epoch: 0021 cost= 0.342749324\n",
      "Epoch: 0022 cost= 0.340224642\n",
      "Epoch: 0023 cost= 0.337897302\n",
      "Epoch: 0024 cost= 0.335720168\n",
      "Epoch: 0025 cost= 0.333691911\n",
      "Optimization Finished!\n",
      "Accuracy: 0.9143\n",
      "Run the command line:\n",
      "--> tensorboard --logdir=/tmp/tensorflow_logs \n",
      "Then open http://0.0.0.0:6006/ into your web browser\n"
     ]
    }
   ],
   "source": [
    "# Start Training\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "\n",
    "    # op to write logs to Tensorboard\n",
    "    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())\n",
    "\n",
    "    # Training cycle\n",
    "    for epoch in range(training_epochs):\n",
    "        avg_cost = 0.\n",
    "        total_batch = int(mnist.train.num_examples / batch_size)\n",
    "        # Loop over all batches\n",
    "        for i in range(total_batch):\n",
    "            batch_xs, batch_ys = mnist.train.next_batch(batch_size)\n",
    "            # Run optimization op (backprop), cost op (to get loss value)\n",
    "            # and summary nodes\n",
    "            _, c, summary = sess.run([optimizer, cost, merged_summary_op],\n",
    "                                     feed_dict={x: batch_xs, y: batch_ys})\n",
    "            # Write logs at every iteration\n",
    "            summary_writer.add_summary(summary, epoch * total_batch + i)\n",
    "            # Compute average loss\n",
    "            avg_cost += c / total_batch\n",
    "        # Display logs per epoch step\n",
    "        if (epoch+1) % display_epoch == 0:\n",
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
    "\n",
    "<img src=\"../../resources/img/tensorboard_basic_1.png\"/>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Graph Visualization\n",
    "\n",
    "<img src=\"../../resources/img/tensorboard_basic_2.png\"/>"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [default]",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 1
}

```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/4_Utils/tensorboard_basic.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/notebooks/4_Utils/`, this file is part of the TensorFlow 1.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 7
- Code cells: 4
- Markdown cells: 3



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v1/notebooks/4_Utils/tensorboard_basic.ipynb
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
- [tensorboard_advanced.ipynb](tensorboard_advanced.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

training, gradient, accuracy, loss, mnist, softmax, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
