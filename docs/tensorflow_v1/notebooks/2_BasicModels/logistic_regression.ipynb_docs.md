# Documentation: tensorflow_v1/notebooks/2_BasicModels/logistic_regression.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/2_BasicModels/logistic_regression.ipynb`
- **File Size**: 5539 bytes
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
    "# Logistic Regression Example\n",
    "\n",
    "A logistic regression learning algorithm example using TensorFlow library.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
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
      "Extracting MNIST_data/train-images-idx3-ubyte.gz\n",
      "Extracting MNIST_data/train-labels-idx1-ubyte.gz\n",
      "Extracting MNIST_data/t10k-images-idx3-ubyte.gz\n",
      "Extracting MNIST_data/t10k-labels-idx1-ubyte.gz\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "\n",
    "# Import MINST data\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"/tmp/data/\", one_hot=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Parameters\n",
    "learning_rate = 0.01\n",
    "training_epochs = 25\n",
    "batch_size = 100\n",
    "display_step = 1\n",
    "\n",
    "# tf Graph Input\n",
    "x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784\n",
    "y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes\n",
    "\n",
    "# Set model weights\n",
    "W = tf.Variable(tf.zeros([784, 10]))\n",
    "b = tf.Variable(tf.zeros([10]))\n",
    "\n",
    "# Construct model\n",
    "pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax\n",
    "\n",
    "# Minimize error using cross entropy\n",
    "cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))\n",
    "# Gradient Descent\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)\n",
    "\n",
    "# Initialize the variables (i.e. assign their default value)\n",
    "init = tf.global_variables_initializer()"
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
      "Epoch: 0001 cost= 1.182138959\n",
      "Epoch: 0002 cost= 0.664778162\n",
      "Epoch: 0003 cost= 0.552686284\n",
      "Epoch: 0004 cost= 0.498628905\n",
      "Epoch: 0005 cost= 0.465469866\n",
      "Epoch: 0006 cost= 0.442537872\n",
      "Epoch: 0007 cost= 0.425462044\n",
      "Epoch: 0008 cost= 0.412185303\n",
      "Epoch: 0009 cost= 0.401311587\n",
      "Epoch: 0010 cost= 0.392326203\n",
      "Epoch: 0011 cost= 0.384736038\n",
      "Epoch: 0012 cost= 0.378137191\n",
      "Epoch: 0013 cost= 0.372363752\n",
      "Epoch: 0014 cost= 0.367308579\n",
      "Epoch: 0015 cost= 0.362704660\n",
      "Epoch: 0016 cost= 0.358588599\n",
      "Epoch: 0017 cost= 0.354823110\n"
     ]
    }
   ],
   "source": [
    "# Start training\n",
    "with tf.Session() as sess:\n",
    "    sess.run(init)\n",
    "\n",
    "    # Training cycle\n",
    "    for epoch in range(training_epochs):\n",
    "        avg_cost = 0.\n",
    "        total_batch = int(mnist.train.num_examples/batch_size)\n",
    "        # Loop over all batches\n",
    "        for i in range(total_batch):\n",
    "            batch_xs, batch_ys = mnist.train.next_batch(batch_size)\n",
    "            # Fit training using batch data\n",
    "            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,\n",
    "                                                          y: batch_ys})\n",
    "            # Compute average loss\n",
    "            avg_cost += c / total_batch\n",
    "        # Display logs per epoch step\n",
    "        if (epoch+1) % display_step == 0:\n",
    "            print \"Epoch:\", '%04d' % (epoch+1), \"cost=\", \"{:.9f}\".format(avg_cost)\n",
    "\n",
    "    print \"Optimization Finished!\"\n",
    "\n",
    "    # Test model\n",
    "    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))\n",
    "    # Calculate accuracy for 3000 examples\n",
    "    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))\n",
    "    print \"Accuracy:\", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]})"
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
    "version": 2
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
 "nbformat_minor": 0
}

```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/2_BasicModels/logistic_regression.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/notebooks/2_BasicModels/`, this file is part of the TensorFlow 1.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 5
- Code cells: 3
- Markdown cells: 2



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v1/notebooks/2_BasicModels/logistic_regression.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It demonstrates basic machine learning models.

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

- [gradient_boosted_decision_tree.ipynb](gradient_boosted_decision_tree.ipynb_docs.md)
- [kmeans.ipynb](kmeans.ipynb_docs.md)
- [linear_regression.ipynb](linear_regression.ipynb_docs.md)
- [linear_regression_eager_api.ipynb](linear_regression_eager_api.ipynb_docs.md)
- [logistic_regression_eager_api.ipynb](logistic_regression_eager_api.ipynb_docs.md)
- [nearest_neighbor.ipynb](nearest_neighbor.ipynb_docs.md)
- [random_forest.ipynb](random_forest.ipynb_docs.md)
- [word2vec.ipynb](word2vec.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

dataset, training, testing, regression, gradient, accuracy, loss, mnist, softmax, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
