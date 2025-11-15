# Documentation: tensorflow_v1/notebooks/2_BasicModels/word2vec.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/2_BasicModels/word2vec.ipynb`
- **File Size**: 34258 bytes
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
    "# Word2Vec (Word Embedding)\n",
    "\n",
    "Implement Word2Vec algorithm to compute vector representations of words.\n",
    "This example is using a small chunk of Wikipedia articles to train from.\n",
    "\n",
    "More info: [Mikolov, Tomas et al. \"Efficient Estimation of Word Representations in Vector Space.\", 2013](https://arxiv.org/pdf/1301.3781.pdf)\n",
    "\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
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
    "from __future__ import division, print_function, absolute_import\n",
    "\n",
    "import collections\n",
    "import os\n",
    "import random\n",
    "import urllib\n",
    "import zipfile\n",
    "\n",
    "import numpy as np\n",
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
    "# Training Parameters\n",
    "learning_rate = 0.1\n",
    "batch_size = 128\n",
    "num_steps = 3000000\n",
    "display_step = 10000\n",
    "eval_step = 200000\n",
    "\n",
    "# Evaluation Parameters\n",
    "eval_words = ['five', 'of', 'going', 'hardware', 'american', 'britain']\n",
    "\n",
    "# Word2Vec Parameters\n",
    "embedding_size = 200 # Dimension of the embedding vector\n",
    "max_vocabulary_size = 50000 # Total number of different words in the vocabulary\n",
    "min_occurrence = 10 # Remove all words that does not appears at least n times\n",
    "skip_window = 3 # How many words to consider left and right\n",
    "num_skips = 2 # How many times to reuse an input to generate a label\n",
    "num_sampled = 64 # Number of negative examples to sample"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading the dataset... (It may take some time)\n",
      "Done!\n"
     ]
    }
   ],
   "source": [
    "# Download a small chunk of Wikipedia articles collection\n",
    "url = 'http://mattmahoney.net/dc/text8.zip'\n",
    "data_path = 'text8.zip'\n",
    "if not os.path.exists(data_path):\n",
    "    print(\"Downloading the dataset... (It may take some time)\")\n",
    "    filename, _ = urllib.urlretrieve(url, data_path)\n",
    "    print(\"Done!\")\n",
    "# Unzip the dataset file. Text has already been processed\n",
    "with zipfile.ZipFile(data_path) as f:\n",
    "    text_words = f.read(f.namelist()[0]).lower().split()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Words count: 17005207\n",
      "Unique words: 253854\n",
      "Vocabulary size: 50000\n",
      "Most common words: [('UNK', 418391), ('the', 1061396), ('of', 593677), ('and', 416629), ('one', 411764), ('in', 372201), ('a', 325873), ('to', 316376), ('zero', 264975), ('nine', 250430)]\n"
     ]
    }
   ],
   "source": [
    "# Build the dictionary and replace rare words with UNK token\n",
    "count = [('UNK', -1)]\n",
    "# Retrieve the most common words\n",
    "count.extend(collections.Counter(text_words).most_common(max_vocabulary_size - 1))\n",
    "# Remove samples with less than 'min_occurrence' occurrences\n",
    "for i in range(len(count) - 1, -1, -1):\n",
    "    if count[i][1] < min_occurrence:\n",
    "        count.pop(i)\n",
    "    else:\n",
    "        # The collection is ordered, so stop when 'min_occurrence' is reached\n",
    "        break\n",
    "# Compute the vocabulary size\n",
    "vocabulary_size = len(count)\n",
    "# Assign an id to each word\n",
    "word2id = dict()\n",
    "for i, (word, _)in enumerate(count):\n",
    "    word2id[word] = i\n",
    "\n",
    "data = list()\n",
    "unk_count = 0\n",
    "for word in text_words:\n",
    "    # Retrieve a word id, or assign it index 0 ('UNK') if not in dictionary\n",
    "    index = word2id.get(word, 0)\n",
    "    if index == 0:\n",
    "        unk_count += 1\n",
    "    data.append(index)\n",
    "count[0] = ('UNK', unk_count)\n",
    "id2word = dict(zip(word2id.values(), word2id.keys()))\n",
    "\n",
    "print(\"Words count:\", len(text_words))\n",
    "print(\"Unique words:\", len(set(text_words)))\n",
    "print(\"Vocabulary size:\", vocabulary_size)\n",
    "print(\"Most common words:\", count[:10])"
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
    "data_index = 0\n",
    "# Generate training batch for the skip-gram model\n",
    "def next_batch(batch_size, num_skips, skip_window):\n",
    "    global data_index\n",
    "    assert batch_size % num_skips == 0\n",
    "    assert num_skips <= 2 * skip_window\n",
    "    batch = np.ndarray(shape=(batch_size), dtype=np.int32)\n",
    "    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)\n",
    "    # get window size (words left and right + current one)\n",
    "    span = 2 * skip_window + 1\n",
    "    buffer = collections.deque(maxlen=span)\n",
    "    if data_index + span > len(data):\n",
    "        data_index = 0\n",
    "    buffer.extend(data[data_index:data_index + span])\n",
    "    data_index += span\n",
    "    for i in range(batch_size // num_skips):\n",
    "        context_words = [w for w in range(span) if w != skip_window]\n",
    "        words_to_use = random.sample(context_words, num_skips)\n",
    "        for j, context_word in enumerate(words_to_use):\n",
    "            batch[i * num_skips + j] = buffer[skip_window]\n",
    "            labels[i * num_skips + j, 0] = buffer[context_word]\n",
    "        if data_index == len(data):\n",
    "            buffer.extend(data[0:span])\n",
    "            data_index = span\n",
    "        else:\n",
    "            buffer.append(data[data_index])\n",
    "            data_index += 1\n",
    "    # Backtrack a little bit to avoid skipping words in the end of a batch\n",
    "    data_index = (data_index + len(data) - span) % len(data)\n",
    "    return batch, labels"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Input data\n",
    "X = tf.placeholder(tf.int32, shape=[None])\n",
    "# Input label\n",
    "Y = tf.placeholder(tf.int32, shape=[None, 1])\n",
    "\n",
    "# Ensure the following ops & var are assigned on CPU\n",
    "# (some ops are not compatible on GPU)\n",
    "with tf.device('/cpu:0'):\n",
    "    # Create the embedding variable (each row represent a word embedding vector)\n",
    "    embedding = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))\n",
    "    # Lookup the corresponding embedding vectors for each sample in X\n",
    "    X_embed = tf.nn.embedding_lookup(embedding, X)\n",
    "\n",
    "    # Construct the variables for the NCE loss\n",
    "    nce_weights = tf.Variable(tf.random_normal([vocabulary_size, embedding_size]))\n",
    "    nce_biases = tf.Variable(tf.zeros([vocabulary_size]))\n",
    "\n",
    "# Compute the average NCE loss for the batch\n",
    "loss_op = tf.reduce_mean(\n",
    "    tf.nn.nce_loss(weights=nce_weights,\n",
    "                   biases=nce_biases,\n",
    "                   labels=Y,\n",
    "                   inputs=X_embed,\n",
    "                   num_sampled=num_sampled,\n",
    "                   num_classes=vocabulary_size))\n",
    "\n",
    "# Define the optimizer\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate)\n",
    "train_op = optimizer.minimize(loss_op)\n",
    "\n",
    "# Evaluation\n",
    "# Compute the cosine similarity between input data embedding and every embedding vectors\n",
    "X_embed_norm = X_embed / tf.sqrt(tf.reduce_sum(tf.square(X_embed)))\n",
    "embedding_norm = embedding / tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keepdims=True))\n",
    "cosine_sim_op = tf.matmul(X_embed_norm, embedding_norm, transpose_b=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false,
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Step 1, Average Loss= 520.3188\n",
      "Evaluation...\n",
      "\"five\" nearest neighbors: brothers, swinging, dissemination, fruitful, trichloride, dll, timur, torre,\n",
      "\"of\" nearest neighbors: malting, vaginal, cecil, xiaoping, arrangers, hydras, exhibits, splits,\n",
      "\"going\" nearest neighbors: besht, xps, sdtv, mississippi, frequencies, tora, reciprocating, tursiops,\n",
      "\"hardware\" nearest neighbors: burgh, residences, mares, attested, whirlwind, isomerism, admiration, ties,\n",
      "\"american\" nearest neighbors: tensile, months, baffling, cricket, kodak, risky, nicomedia, jura,\n",
      "\"britain\" nearest neighbors: superstring, interpretations, genealogical, munition, boer, occasional, psychologists, turbofan,\n",
      "Step 10000, Average Loss= 202.2640\n",
      "Step 20000, Average Loss= 96.5149\n",
      "Step 30000, Average Loss= 67.2858\n",
      "Step 40000, Average Loss= 52.5055\n",
      "Step 50000, Average Loss= 42.6301\n",
      "Step 60000, Average Loss= 37.3644\n",
      "Step 70000, Average Loss= 33.1220\n",
      "Step 80000, Average Loss= 30.5835\n",
      "Step 90000, Average Loss= 28.2243\n",
      "Step 100000, Average Loss= 25.5532\n",
      "Step 110000, Average Loss= 24.0891\n",
      "Step 120000, Average Loss= 21.8576\n",
      "Step 130000, Average Loss= 21.2192\n",
      "Step 140000, Average Loss= 19.8834\n",
     
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/2_BasicModels/word2vec.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/notebooks/2_BasicModels/`, this file is part of the TensorFlow 1.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 8
- Code cells: 7
- Markdown cells: 1



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v1/notebooks/2_BasicModels/word2vec.ipynb
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

- [gradient_boosted_decision_tree.ipynb](gradient_boosted_decision_tree.ipynb_docs.md)
- [kmeans.ipynb](kmeans.ipynb_docs.md)
- [linear_regression.ipynb](linear_regression.ipynb_docs.md)
- [linear_regression_eager_api.ipynb](linear_regression_eager_api.ipynb_docs.md)
- [logistic_regression.ipynb](logistic_regression.ipynb_docs.md)
- [logistic_regression_eager_api.ipynb](logistic_regression_eager_api.ipynb_docs.md)
- [nearest_neighbor.ipynb](nearest_neighbor.ipynb_docs.md)
- [random_forest.ipynb](random_forest.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

dataset, training, testing, gradient, embedding, gan, loss, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
