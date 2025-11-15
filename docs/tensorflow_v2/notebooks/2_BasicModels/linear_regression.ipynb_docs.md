# Documentation: tensorflow_v2/notebooks/2_BasicModels/linear_regression.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/2_BasicModels/linear_regression.ipynb`
- **File Size**: 18444 bytes
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
    "# Linear Regression Example\n",
    "\n",
    "Linear regression implementation with TensorFlow v2 library.\n",
    "\n",
    "This example is using a low-level approach to better understand all mechanics behind the training process.\n",
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
    "from __future__ import absolute_import, division, print_function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "rng = np.random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Parameters.\n",
    "learning_rate = 0.01\n",
    "training_steps = 1000\n",
    "display_step = 50"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Training Data.\n",
    "X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,\n",
    "              7.042,10.791,5.313,7.997,5.654,9.27,3.1])\n",
    "Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,\n",
    "              2.827,3.465,1.65,2.904,2.42,2.94,1.3])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Weight and Bias, initialized randomly.\n",
    "W = tf.Variable(rng.randn(), name=\"weight\")\n",
    "b = tf.Variable(rng.randn(), name=\"bias\")\n",
    "\n",
    "# Linear regression (Wx + b).\n",
    "def linear_regression(x):\n",
    "    return W * x + b\n",
    "\n",
    "# Mean square error.\n",
    "def mean_square(y_pred, y_true):\n",
    "    return tf.reduce_mean(tf.square(y_pred - y_true))\n",
    "\n",
    "# Stochastic Gradient Descent Optimizer.\n",
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
    "def run_optimization():\n",
    "    # Wrap computation inside a GradientTape for automatic differentiation.\n",
    "    with tf.GradientTape() as g:\n",
    "        pred = linear_regression(X)\n",
    "        loss = mean_square(pred, Y)\n",
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
      "step: 50, loss: 0.210631, W: 0.458940, b: -0.670898\n",
      "step: 100, loss: 0.195340, W: 0.446725, b: -0.584301\n",
      "step: 150, loss: 0.181797, W: 0.435230, b: -0.502807\n",
      "step: 200, loss: 0.169803, W: 0.424413, b: -0.426115\n",
      "step: 250, loss: 0.159181, W: 0.414232, b: -0.353942\n",
      "step: 300, loss: 0.149774, W: 0.404652, b: -0.286021\n",
      "step: 350, loss: 0.141443, W: 0.395636, b: -0.222102\n",
      "step: 400, loss: 0.134064, W: 0.387151, b: -0.161949\n",
      "step: 450, loss: 0.127530, W: 0.379167, b: -0.105341\n",
      "step: 500, loss: 0.121742, W: 0.371652, b: -0.052068\n",
      "step: 550, loss: 0.116617, W: 0.364581, b: -0.001933\n",
      "step: 600, loss: 0.112078, W: 0.357926, b: 0.045247\n",
      "step: 650, loss: 0.108058, W: 0.351663, b: 0.089647\n",
      "step: 700, loss: 0.104498, W: 0.345769, b: 0.131431\n",
      "step: 750, loss: 0.101345, W: 0.340223, b: 0.170753\n",
      "step: 800, loss: 0.098552, W: 0.335003, b: 0.207759\n",
      "step: 850, loss: 0.096079, W: 0.330091, b: 0.242583\n",
      "step: 900, loss: 0.093889, W: 0.325468, b: 0.275356\n",
      "step: 950, loss: 0.091949, W: 0.321118, b: 0.306198\n",
      "step: 1000, loss: 0.090231, W: 0.317024, b: 0.335223\n"
     ]
    }
   ],
   "source": [
    "# Run training for the given number of steps.\n",
    "for step in range(1, training_steps + 1):\n",
    "    # Run the optimization to update W and b values.\n",
    "    run_optimization()\n",
    "    \n",
    "    if step % display_step == 0:\n",
    "        pred = linear_regression(X)\n",
    "        loss = mean_square(pred, Y)\n",
    "        print(\"step: %i, loss: %f, W: %f, b: %f\" % (step, loss, W.numpy(), b.numpy()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/2_BasicModels/linear_regression.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/2_BasicModels/`, this file is part of the TensorFlow 2.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 10
- Code cells: 9
- Markdown cells: 1



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v2/notebooks/2_BasicModels/linear_regression.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It demonstrates basic machine learning models.

### Design Patterns

- Uses TensorFlow framework for machine learning operations


## Performance & Complexity

### Performance Characteristics

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

- [gradient_boosted_trees.ipynb](gradient_boosted_trees.ipynb_docs.md)
- [logistic_regression.ipynb](logistic_regression.ipynb_docs.md)
- [word2vec.ipynb](word2vec.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

rnn, training, regression, gradient, loss, tensorflow, optimizer, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
