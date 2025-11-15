# Documentation: notebooks/3_NeuralNetworks/dynamic_rnn.ipynb

## File Metadata

- **File Path**: `notebooks/3_NeuralNetworks/dynamic_rnn.ipynb`
- **File Size**: 15134 bytes
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
    "# Dynamic Recurrent Neural Network.\n",
    "\n",
    "TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs dynamic computation over sequences with variable length. This example is using a toy dataset to classify linear sequences. The generated sequences have variable length.\n",
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
    "- [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf), Sepp Hochreiter & Jurgen Schmidhuber, Neural Computation 9(8): 1735-1780, 1997."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from __future__ import print_function\n",
    "\n",
    "import tensorflow as tf\n",
    "import random"
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
    "# ====================\n",
    "#  TOY DATA GENERATOR\n",
    "# ====================\n",
    "\n",
    "class ToySequenceData(object):\n",
    "    \"\"\" Generate sequence of data with dynamic length.\n",
    "    This class generate samples for training:\n",
    "    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])\n",
    "    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])\n",
    "\n",
    "    NOTICE:\n",
    "    We have to pad each sequence to reach 'max_seq_len' for TensorFlow\n",
    "    consistency (we cannot feed a numpy array with inconsistent\n",
    "    dimensions). The dynamic calculation will then be perform thanks to\n",
    "    'seqlen' attribute that records every actual sequence length.\n",
    "    \"\"\"\n",
    "    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=3,\n",
    "                 max_value=1000):\n",
    "        self.data = []\n",
    "        self.labels = []\n",
    "        self.seqlen = []\n",
    "        for i in range(n_samples):\n",
    "            # Random sequence length\n",
    "            len = random.randint(min_seq_len, max_seq_len)\n",
    "            # Monitor sequence length for TensorFlow dynamic calculation\n",
    "            self.seqlen.append(len)\n",
    "            # Add a random or linear int sequence (50% prob)\n",
    "            if random.random() < .5:\n",
    "                # Generate a linear sequence\n",
    "                rand_start = random.randint(0, max_value - len)\n",
    "                s = [[float(i)/max_value] for i in\n",
    "                     range(rand_start, rand_start + len)]\n",
    "                # Pad sequence for dimension consistency\n",
    "                s += [[0.] for i in range(max_seq_len - len)]\n",
    "                self.data.append(s)\n",
    "                self.labels.append([1., 0.])\n",
    "            else:\n",
    "                # Generate a random sequence\n",
    "                s = [[float(random.randint(0, max_value))/max_value]\n",
    "                     for i in range(len)]\n",
    "                # Pad sequence for dimension consistency\n",
    "                s += [[0.] for i in range(max_seq_len - len)]\n",
    "                self.data.append(s)\n",
    "                self.labels.append([0., 1.])\n",
    "        self.batch_id = 0\n",
    "\n",
    "    def next(self, batch_size):\n",
    "        \"\"\" Return a batch of data. When dataset end is reached, start over.\n",
    "        \"\"\"\n",
    "        if self.batch_id == len(self.data):\n",
    "            self.batch_id = 0\n",
    "        batch_data = (self.data[self.batch_id:min(self.batch_id +\n",
    "                                                  batch_size, len(self.data))])\n",
    "        batch_labels = (self.labels[self.batch_id:min(self.batch_id +\n",
    "                                                  batch_size, len(self.data))])\n",
    "        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +\n",
    "                                                  batch_size, len(self.data))])\n",
    "        self.batch_id = min(self.batch_id + batch_size, len(self.data))\n",
    "        return batch_data, batch_labels, batch_seqlen"
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
    "# ==========\n",
    "#   MODEL\n",
    "# ==========\n",
    "\n",
    "# Parameters\n",
    "learning_rate = 0.01\n",
    "training_steps = 10000\n",
    "batch_size = 128\n",
    "display_step = 200\n",
    "\n",
    "# Network Parameters\n",
    "seq_max_len = 20 # Sequence max length\n",
    "n_hidden = 64 # hidden layer num of features\n",
    "n_classes = 2 # linear sequence or not\n",
    "\n",
    "trainset = ToySequenceData(n_samples=1000, max_seq_len=seq_max_len)\n",
    "testset = ToySequenceData(n_samples=500, max_seq_len=seq_max_len)\n",
    "\n",
    "# tf Graph input\n",
    "x = tf.placeholder(\"float\", [None, seq_max_len, 1])\n",
    "y = tf.placeholder(\"float\", [None, n_classes])\n",
    "# A placeholder for indicating each sequence length\n",
    "seqlen = tf.placeholder(tf.int32, [None])\n",
    "\n",
    "# Define weights\n",
    "weights = {\n",
    "    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))\n",
    "}\n",
    "biases = {\n",
    "    'out': tf.Variable(tf.random_normal([n_classes]))\n",
    "}"
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
    "def dynamicRNN(x, seqlen, weights, biases):\n",
    "\n",
    "    # Prepare data shape to match `rnn` function requirements\n",
    "    # Current data input shape: (batch_size, n_steps, n_input)\n",
    "    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)\n",
    "    \n",
    "    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)\n",
    "    x = tf.unstack(x, seq_max_len, 1)\n",
    "\n",
    "    # Define a lstm cell with tensorflow\n",
    "    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)\n",
    "\n",
    "    # Get lstm cell output, providing 'sequence_length' will perform dynamic\n",
    "    # calculation.\n",
    "    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,\n",
    "                                sequence_length=seqlen)\n",
    "\n",
    "    # When performing dynamic calculation, we must retrieve the last\n",
    "    # dynamically computed output, i.e., if a sequence length is 10, we need\n",
    "    # to retrieve the 10th output.\n",
    "    # However TensorFlow doesn't support advanced indexing yet, so we build\n",
    "    # a custom op that for each sample in batch size, get its length and\n",
    "    # get the corresponding relevant output.\n",
    "\n",
    "    # 'outputs' is a list of output at every timestep, we pack them in a Tensor\n",
    "    # and change back dimension to [batch_size, n_step, n_input]\n",
    "    outputs = tf.stack(outputs)\n",
    "    outputs = tf.transpose(outputs, [1, 0, 2])\n",
    "\n",
    "    # Hack to build the indexing and retrieve the right output.\n",
    "    batch_size = tf.shape(outputs)[0]\n",
    "    # Start indices for each sample\n",
    "    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)\n",
    "    # Indexing\n",
    "    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)\n",
    "\n",
    "    # Linear activation, using outputs computed above\n",
    "    return tf.matmul(outputs, weights['out']) + biases['out']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/aymeric.damien/anaconda2/lib/python2.7/site-packages/tensorflow/python/ops/gradients_impl.py:93: UserWarning: Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.\n",
      "  \"Converting sparse IndexedSlices to a dense Tensor of unknown shape. \"\n"
     ]
    }
   ],
   "source": [
    "pred = dynamicRNN(x, seqlen, weights, biases)\n",
    "\n",
    "# Define loss and optimizer\n",
    "cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)\n",
    "\n",
    "# Evaluate model\n",
    "correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))\n",
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
    "collapsed": false,
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Step 1, Minibatch Loss= 0.864517, Training Accuracy= 0.42188\n",
      "Step 200, Minibatch Loss= 0.686012, Training Accuracy= 0.43269\n",
      "Step 400, Minibatch Loss= 0.682970, Training Accuracy= 0.48077\n",
      "Step 600, Minibatch Loss= 0.679640, Training Accuracy= 0.50962\n",
      "Step 800, Minibatch Loss= 0.675208, Training Accuracy= 0.53846\n",
      "Step 1000, Minibatch Loss= 0.668636, Training Accuracy= 0.56731\n",
      "Step 1200, Minibatch Loss= 0.657525, Training Accuracy= 0.62500\n",
      "Step 1400, Minibatch Loss= 0.635423, Training Accuracy= 0.67308\n",
      "Step 1600, Minibatch Loss= 0.580433, 
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `notebooks/3_NeuralNetworks/dynamic_rnn.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `notebooks/` directory, specifically within `notebooks/3_NeuralNetworks/`, this file is part of the notebook-based tutorials.



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
jupyter notebook notebooks/3_NeuralNetworks/dynamic_rnn.ipynb
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
- [gan.ipynb](gan.ipynb_docs.md)
- [neural_network.ipynb](neural_network.ipynb_docs.md)
- [neural_network_eager_api.ipynb](neural_network_eager_api.ipynb_docs.md)
- [neural_network_raw.ipynb](neural_network_raw.ipynb_docs.md)
- [recurrent_network.ipynb](recurrent_network.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

activation, rnn, dataset, training, testing, gradient, neural network, lstm, accuracy, loss, softmax, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
