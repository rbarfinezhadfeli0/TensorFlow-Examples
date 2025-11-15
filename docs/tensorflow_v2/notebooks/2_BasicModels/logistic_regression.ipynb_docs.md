# Documentation: tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb`
- **File Size**: 33510 bytes
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
    "# Logistic Regression Example\n",
    "\n",
    "Logistic regression implementation with TensorFlow v2 library.\n",
    "\n",
    "This example is using a low-level approach to better understand all mechanics behind the training process.\n",
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
    "# MNIST dataset parameters.\n",
    "num_classes = 10 # 0 to 9 digits\n",
    "num_features = 784 # 28*28\n",
    "\n",
    "# Training parameters.\n",
    "learning_rate = 0.01\n",
    "training_steps = 1000\n",
    "batch_size = 256\n",
    "display_step = 50"
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
    "x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)\n",
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
    "train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Weight of shape [784, 10], the 28*28 image features, and total number of classes.\n",
    "W = tf.Variable(tf.ones([num_features, num_classes]), name=\"weight\")\n",
    "# Bias of shape [10], the total number of classes.\n",
    "b = tf.Variable(tf.zeros([num_classes]), name=\"bias\")\n",
    "\n",
    "# Logistic regression (Wx + b).\n",
    "def logistic_regression(x):\n",
    "    # Apply softmax to normalize the logits to a probability distribution.\n",
    "    return tf.nn.softmax(tf.matmul(x, W) + b)\n",
    "\n",
    "# Cross-Entropy loss function.\n",
    "def cross_entropy(y_pred, y_true):\n",
    "    # Encode label to a one hot vector.\n",
    "    y_true = tf.one_hot(y_true, depth=num_classes)\n",
    "    # Clip prediction values to avoid log(0) error.\n",
    "    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)\n",
    "    # Compute cross-entropy.\n",
    "    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred),1))\n",
    "\n",
    "# Accuracy metric.\n",
    "def accuracy(y_pred, y_true):\n",
    "    # Predicted class is the index of highest score in prediction vector (i.e. argmax).\n",
    "    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))\n",
    "    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))\n",
    "\n",
    "# Stochastic gradient descent optimizer.\n",
    "optimizer = tf.optimizers.SGD(learning_rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optimization process. \n",
    "def run_optimization(x, y):\n",
    "    # Wrap computation inside a GradientTape for automatic differentiation.\n",
    "    with tf.GradientTape() as g:\n",
    "        pred = logistic_regression(x)\n",
    "        loss = cross_entropy(pred, y)\n",
    "\n",
    "    # Compute gradients.\n",
    "    gradients = g.gradient(loss, [W, b])\n",
    "    \n",
    "    # Update W and b following gradients.\n",
    "    optimizer.apply_gradients(zip(gradients, [W, b]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "step: 50, loss: 608.584717, accuracy: 0.824219\n",
      "step: 100, loss: 828.206482, accuracy: 0.765625\n",
      "step: 150, loss: 716.329407, accuracy: 0.746094\n",
      "step: 200, loss: 584.887634, accuracy: 0.820312\n",
      "step: 250, loss: 472.098114, accuracy: 0.871094\n",
      "step: 300, loss: 621.834595, accuracy: 0.832031\n",
      "step: 350, loss: 567.288818, accuracy: 0.714844\n",
      "step: 400, loss: 489.062988, accuracy: 0.847656\n",
      "step: 450, loss: 496.466675, accuracy: 0.843750\n",
      "step: 500, loss: 465.342224, accuracy: 0.875000\n",
      "step: 550, loss: 586.347168, accuracy: 0.855469\n",
      "step: 600, loss: 95.233109, accuracy: 0.906250\n",
      "step: 650, loss: 88.136490, accuracy: 0.910156\n",
      "step: 700, loss: 67.170349, accuracy: 0.937500\n",
      "step: 750, loss: 79.673691, accuracy: 0.921875\n",
      "step: 800, loss: 112.844872, accuracy: 0.914062\n",
      "step: 850, loss: 92.789581, accuracy: 0.894531\n",
      "step: 900, loss: 80.116165, accuracy: 0.921875\n",
      "step: 950, loss: 45.706650, accuracy: 0.925781\n",
      "step: 1000, loss: 72.986969, accuracy: 0.925781\n"
     ]
    }
   ],
   "source": [
    "# Run training for the given number of steps.\n",
    "for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):\n",
    "    # Run the optimization to update W and b values.\n",
    "    run_optimization(batch_x, batch_y)\n",
    "    \n",
    "    if step % display_step == 0:\n",
    "        pred = logistic_regression(batch_x)\n",
    "        loss = cross_entropy(pred, batch_y)\n",
    "        acc = accuracy(pred, batch_y)\n",
    "        print(\"step: %i, loss: %f, accuracy: %f\" % (step, loss, acc))"
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
      "Test Accuracy: 0.915100\n"
     ]
    }
   ],
   "source": [
    "# Test model on validation set.\n",
    "pred = logistic_regression(x_test)\n",
    "print(\"Test Accuracy: %f\" % accuracy(pred, y_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize predictions.\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAP8AAAD8CAYAAAC4nHJkAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADl0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uIDIuMi4yLCBodHRwOi8vbWF0cGxvdGxpYi5vcmcvhp/UCwAADQNJREFUeJzt3W+MVfWdx/HPZylNjPQBWLHEgnQb3bgaAzoaE3AzamxYbYKN1NQHGzbZMH2AZps0ZA1PypMmjemfrU9IpikpJtSWhFbRGBeDGylRGwejBYpQICzMgkAzJgUT0yDfPphDO8W5v3u5/84dv+9XQube8z1/vrnhM+ecOefcnyNCAPL5h7obAFAPwg8kRfiBpAg/kBThB5Ii/EBShB9IivADSRF+IKnP9HNjtrmdEOixiHAr83W057e9wvZB24dtP9nJugD0l9u9t9/2LEmHJD0gaVzSW5Iei4jfF5Zhzw/0WD/2/HdJOhwRRyPiz5J+IWllB+sD0EedhP96SSemvB+vpv0d2yO2x2yPdbAtAF3WyR/8pju0+MRhfUSMShqVOOwHBkkne/5xSQunvP+ipJOdtQOgXzoJ/1uSbrT9JduflfQNSdu70xaAXmv7sD8iLth+XNL/SJolaVNE7O9aZwB6qu1LfW1tjHN+oOf6cpMPgJmL8ANJEX4gKcIPJEX4gaQIP5AU4QeSIvxAUoQfSIrwA0kRfiApwg8kRfiBpAg/kBThB5Ii/EBShB9IivADSRF+ICnCDyRF+IGkCD+QFOEHkiL8QFKEH0iK8ANJEX4gKcIPJEX4gaTaHqJbkmwfk3RO0seSLkTEUDeaAtB7HYW/cm9E/LEL6wHQRxz2A0l1Gv6QtMP2Htsj3WgIQH90eti/LCJO2p4v6RXb70XErqkzVL8U+MUADBhHRHdWZG+QdD4ivl+YpzsbA9BQRLiV+do+7Ld9te3PXXot6SuS9rW7PgD91clh/3WSfm370np+HhEvd6UrAD3XtcP+ljbGYT/Qcz0/7AcwsxF+ICnCDyRF+IGkCD+QFOEHkurGU30prFq1qmFtzZo1xWVPnjxZrH/00UfF+pYtW4r1999/v2Ht8OHDxWWRF3t+ICnCDyRF+IGkCD+QFOEHkiL8QFKEH0iKR3pbdPTo0Ya1xYsX96+RaZw7d65hbf/+/X3sZLCMj483rD311FPFZcfGxrrdTt/wSC+AIsIPJEX4gaQIP5AU4QeSIvxAUoQfSIrn+VtUemb/tttuKy574MCBYv3mm28u1m+//fZifXh4uGHt7rvvLi574sSJYn3hwoXFeicuXLhQrJ89e7ZYX7BgQdvbPn78eLE+k6/zt4o9P5AU4QeSIvxAUoQfSIrwA0kRfiApwg8k1fR5ftubJH1V0pmIuLWaNk/SLyUtlnRM0qMR8UHTjc3g5/kH2dy5cxvWlixZUlx2z549xfqdd97ZVk+taDZewaFDh4r1ZvdPzJs3r2Ft7dq1xWU3btxYrA+ybj7P/zNJKy6b9qSknRFxo6Sd1XsAM0jT8EfELkkTl01eKWlz9XqzpIe73BeAHmv3nP+6iDglSdXP+d1rCUA/9PzeftsjkkZ6vR0AV6bdPf9p2wskqfp5ptGMETEaEUMRMdTmtgD0QLvh3y5pdfV6taTnu9MOgH5pGn7bz0p6Q9I/2R63/R+SvifpAdt/kPRA9R7ADML39mNgPfLII8X61q1bi/V9+/Y1rN17773FZScmLr/ANXPwvf0Aigg/kBThB5Ii/EBShB9IivADSXGpD7WZP7/8SMjevXs7Wn7VqlUNa9u2bSsuO5NxqQ9AEeEHkiL8QFKEH0iK8ANJEX4gKcIPJMUQ3ahNs6/Pvvbaa4v1Dz4of1v8wYMHr7inTNjzA0kRfiApwg8kRfiBpAg/kBThB5Ii/EBSPM+Pnlq2bFnD2quvvlpcdvbs2cX68PBwsb5r165i/dOK5/kBFBF+ICnCDyRF+IGkCD+QFOEHkiL8QFJNn+e3vUnSVyWdiYhbq2kbJK2RdLaabX1EvNSrJjFzPfjggw1rza7j79y5s1h/44032uoJk1rZ8/9M0opppv8oIpZU/wg+MMM0DX9E7JI00YdeAPRRJ+f8j9v+ne1Ntud2rSMAfdFu+DdK+rKkJZJOSfpBoxltj9gesz3W5rYA9EBb4Y+I0xHxcURclPQTSXcV5h2NiKGIGGq3SQDd11b4bS+Y8vZrkvZ1px0A/dLKpb5nJQ1L+rztcUnfkTRse4mkkHRM0jd72COAHuB5fnTkqquuKtZ3797dsHbLLbcUl73vvvuK9ddff71Yz4rn+QEUEX4gKcIPJEX4gaQIP5AU4QeSYohudGTdunXF+tKlSxvWXn755eKyXMrrLfb8QFKEH0iK8ANJEX4gKcIPJEX4gaQIP5AUj/Si6KGHHirWn3vuuWL9
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/2_BasicModels/`, this file is part of the TensorFlow 2.x examples collection.



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
jupyter notebook tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It demonstrates basic machine learning models.

### Design Patterns

- Uses TensorFlow framework for machine learning operations
- Implements object-oriented design with custom classes


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

- [gradient_boosted_trees.ipynb](gradient_boosted_trees.ipynb_docs.md)
- [linear_regression.ipynb](linear_regression.ipynb_docs.md)
- [word2vec.ipynb](word2vec.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

dataset, training, testing, regression, validation, gradient, accuracy, gan, loss, mnist, softmax, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
