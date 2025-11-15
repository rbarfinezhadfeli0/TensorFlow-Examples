# Documentation: tensorflow_v2/notebooks/4_Utils/tensorboard.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/4_Utils/tensorboard.ipynb`
- **File Size**: 12366 bytes
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
    "## Tensorboard\n",
    "Graph, Loss, Accuracy & Weights visualization using Tensorboard and TensorFlow v2. This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/).\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
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
    "# Path to save logs into.\n",
    "logs_path = '/tmp/tensorflow_logs/example/'\n",
    "\n",
    "# MNIST dataset parameters.\n",
    "num_classes = 10 # total classes (0-9 digits).\n",
    "num_features = 784 # data features (img shape: 28*28).\n",
    "\n",
    "# Training parameters.\n",
    "learning_rate = 0.001\n",
    "training_steps = 3000\n",
    "batch_size = 256\n",
    "display_step = 100\n",
    "\n",
    "# Network parameters.\n",
    "n_hidden_1 = 128 # 1st layer number of neurons.\n",
    "n_hidden_2 = 256 # 2nd layer number of neurons."
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
    "# Store layers weight & bias\n",
    "\n",
    "# A random value generator to initialize weights.\n",
    "random_normal = tf.initializers.RandomNormal()\n",
    "\n",
    "weights = {\n",
    "    'h1_weights': tf.Variable(random_normal([num_features, n_hidden_1]), name='h1_weights'),\n",
    "    'h2_weights': tf.Variable(random_normal([n_hidden_1, n_hidden_2]), name='h2_weights'),\n",
    "    'logits_weights': tf.Variable(random_normal([n_hidden_2, num_classes]), name='logits_weights')\n",
    "}\n",
    "biases = {\n",
    "    'h1_bias': tf.Variable(tf.zeros([n_hidden_1]), name='h1_bias'),\n",
    "    'h2_bias': tf.Variable(tf.zeros([n_hidden_2]), name='h2_bias'),\n",
    "    'logits_bias': tf.Variable(tf.zeros([num_classes]), name='logits_bias')\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Construct model and encapsulating all ops into scopes, making\n",
    "# Tensorboard's Graph visualization more convenient.\n",
    "\n",
    "# The computation graph to be traced.\n",
    "@tf.function\n",
    "def neural_net(x):\n",
    "    with tf.name_scope('Model'):\n",
    "        with tf.name_scope('HiddenLayer1'):\n",
    "        # Hidden fully connected layer with 128 neurons.\n",
    "            layer_1 = tf.add(tf.matmul(x, weights['h1_weights']), biases['h1_bias'])\n",
    "            # Apply sigmoid to layer_1 output for non-linearity.\n",
    "            layer_1 = tf.nn.sigmoid(layer_1)\n",
    "        with tf.name_scope('HiddenLayer2'):\n",
    "            # Hidden fully connected layer with 256 neurons.\n",
    "            layer_2 = tf.add(tf.matmul(layer_1, weights['h2_weights']), biases['h2_bias'])\n",
    "            # Apply sigmoid to layer_2 output for non-linearity.\n",
    "            layer_2 = tf.nn.sigmoid(layer_2)\n",
    "        with tf.name_scope('LogitsLayer'):\n",
    "            # Output fully connected layer with a neuron for each class.\n",
    "            out_layer = tf.matmul(layer_2, weights['logits_weights']) + biases['logits_bias']\n",
    "            # Apply softmax to normalize the logits to a probability distribution.\n",
    "            out_layer = tf.nn.softmax(out_layer)\n",
    "    return out_layer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cross-Entropy loss function.\n",
    "def cross_entropy(y_pred, y_true):\n",
    "    with tf.name_scope('CrossEntropyLoss'):\n",
    "        # Encode label to a one hot vector.\n",
    "        y_true = tf.one_hot(y_true, depth=num_classes)\n",
    "        # Clip prediction values to avoid log(0) error.\n",
    "        y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)\n",
    "        # Compute cross-entropy.\n",
    "        return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))\n",
    "\n",
    "# Accuracy metric.\n",
    "def accuracy(y_pred, y_true):\n",
    "    with tf.name_scope('Accuracy'):\n",
    "        # Predicted class is the index of highest score in prediction vector (i.e. argmax).\n",
    "        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))\n",
    "        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)\n",
    "\n",
    "# Stochastic gradient descent optimizer.\n",
    "with tf.name_scope('Optimizer'):\n",
    "    optimizer = tf.optimizers.SGD(learning_rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Optimization process. \n",
    "def run_optimization(x, y):\n",
    "    # Wrap computation inside a GradientTape for automatic differentiation.\n",
    "    with tf.GradientTape() as g:\n",
    "        pred = neural_net(x)\n",
    "        loss = cross_entropy(pred, y)\n",
    "        \n",
    "    # Variables to update, i.e. trainable variables.\n",
    "    trainable_variables = weights.values() + biases.values()\n",
    "\n",
    "    # Compute gradients.\n",
    "    gradients = g.gradient(loss, trainable_variables)\n",
    "    \n",
    "    # Update weights/biases following gradients.\n",
    "    optimizer.apply_gradients(zip(gradients, trainable_variables))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize weights & biases as histogram in Tensorboard.\n",
    "def summarize_weights(step):\n",
    "    for w in weights:\n",
    "        tf.summary.histogram(w.replace('_', '/'), weights[w], step=step)\n",
    "    for b in biases:\n",
    "        tf.summary.histogram(b.replace('_', '/'), biases[b], step=step)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a Summary Writer to log the metrics to Tensorboad.\n",
    "summary_writer = tf.summary.create_file_writer(logs_path)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "step: 100, loss: 568.735596, accuracy: 0.140625\n",
      "step: 200, loss: 413.169342, accuracy: 0.535156\n",
      "step: 300, loss: 250.977036, accuracy: 0.714844\n",
      "step: 400, loss: 173.749298, accuracy: 0.800781\n",
      "step: 500, loss: 156.936569, accuracy: 0.839844\n",
      "step: 600, loss: 137.818451, accuracy: 0.847656\n",
      "step: 700, loss: 93.407814, accuracy: 0.929688\n",
      "step: 800, loss: 90.832336, accuracy: 0.906250\n",
      "step: 900, loss: 86.932831, accuracy: 0.914062\n",
      "step: 1000, loss: 78.824707, accuracy: 0.906250\n",
      "step: 1100, loss: 94.388290, accuracy: 0.902344\n",
      "step: 1200, loss: 96.240608, accuracy: 0.894531\n",
      "step: 1300, loss: 96.657593, accuracy: 0.898438\n",
      "step: 1400, loss: 71.909309, accuracy: 0.914062\n",
      "step: 1500, loss: 67.343407, accuracy: 0.941406\n",
      "step: 1600, loss: 63.693596, accuracy: 0.941406\n",
      "step: 1700, loss: 60.081478, accuracy: 0.914062\n",
      "step: 1800, loss: 63.764942, accuracy: 0.921875\n",
      "step: 1900, loss: 58.722507, accuracy: 0.921875\n",
      "step: 2000, loss: 66.727455, accuracy: 0.917969\n",
      "step: 2100, loss: 70.566788, accuracy: 0.949219\n",
      "step: 2200, loss: 64.642334, accuracy: 0.925781\n",
      "step: 2300, loss: 54.872856, accuracy: 0.941406\n",
      "step: 2400, loss: 64.342377, accuracy: 0.925781\n",
      "step: 2500, loss: 74.306488, accuracy: 0.921875\n",
      "step: 2600, loss: 40.165890, accuracy: 0.949219\n",
      "step: 2700, loss: 64.992249, accuracy: 0.925781\n",
      "step: 2800, loss: 43.422794, accuracy: 0.957031\n",
      "step: 2900, loss: 46.625320, accuracy: 0.937500\n",
      "step: 3000, loss: 62.517433, accuracy: 0.914062\n"
     ]
    }
   ],
   "source": [
    "# Run training for the given number of steps.\n",
    "for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):\n",
    "    \n",
    "    # Start to trace the computation graph. The computation graph remains \n",
    "    # the same at each step, so we just need to export it once.\n",
    "    if step == 1:\n",
    "        tf.summary.trace_on(graph=True, profiler=True)\n",
    "    \n",
    "    # Run the optimization (computation graph).\n",
    "    run_optimization(batch_x, batch_y)\n",
    "    \n",
    "    # Export the computation graph to tens
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/4_Utils/tensorboard.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/4_Utils/`, this file is part of the TensorFlow 2.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 17
- Code cells: 11
- Markdown cells: 6



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v2/notebooks/4_Utils/tensorboard.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It provides utility functions for TensorFlow operations.

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

- [build_custom_layers.ipynb](build_custom_layers.ipynb_docs.md)
- [save_restore_model.ipynb](save_restore_model.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

dataset, training, gradient, accuracy, loss, mnist, sigmoid, tensorflow, softmax, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
