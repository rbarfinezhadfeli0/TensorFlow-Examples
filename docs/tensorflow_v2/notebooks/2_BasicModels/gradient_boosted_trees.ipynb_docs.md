# Documentation: tensorflow_v2/notebooks/2_BasicModels/gradient_boosted_trees.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/2_BasicModels/gradient_boosted_trees.ipynb`
- **File Size**: 29572 bytes
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
    "# Gradient Boosted Decision Tree (GBDT)\n",
    "Implement a Gradient Boosted Decision Tree (GBDT) with TensorFlow. This example is using the Boston Housing Value dataset as training samples. The example supports both Classification (2 classes: value > $23000 or not) and Regression (raw home value as target).\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Boston Housing Dataset\n",
    "\n",
    "**Link:** https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html\n",
    "\n",
    "**Description:**\n",
    "\n",
    "The dataset contains information collected by the U.S Census Service concerning housing in the area of Boston Mass. It was obtained from the StatLib archive (http://lib.stat.cmu.edu/datasets/boston), and has been used extensively throughout the literature to benchmark algorithms. However, these comparisons were primarily done outside of Delve and are thus somewhat suspect. The dataset is small in size with only 506 cases.\n",
    "\n",
    "The data was originally published by Harrison, D. and Rubinfeld, D.L. `Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.`\n",
    "\n",
    "*For the full features list, please see the link above*"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from __future__ import print_function\n",
    "\n",
    "# Ignore all GPUs (current TF GBDT does not support GPU).\n",
    "import os\n",
    "os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"\"\n",
    "os.environ['TF_CPP_MIN_LOG_LEVEL'] = \"1\"\n",
    "\n",
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import copy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dataset parameters.\n",
    "num_classes = 2 # Total classes: greater or equal to $23,000, or not (See notes below).\n",
    "num_features = 13 # data features size.\n",
    "\n",
    "# Training parameters.\n",
    "max_steps = 2000\n",
    "batch_size = 256\n",
    "learning_rate = 1.0\n",
    "l1_regul = 0.0\n",
    "l2_regul = 0.1\n",
    "\n",
    "# GBDT parameters.\n",
    "num_batches_per_layer = 1000\n",
    "num_trees = 10\n",
    "max_depth = 4"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Prepare Boston Housing Dataset.\n",
    "from tensorflow.keras.datasets import boston_housing\n",
    "(x_train, y_train), (x_test, y_test) = boston_housing.load_data()\n",
    "\n",
    "# For classification purpose, we build 2 classes: price greater or lower than $23,000\n",
    "def to_binary_class(y):\n",
    "    for i, label in enumerate(y):\n",
    "        if label >= 23.0:\n",
    "            y[i] = 1\n",
    "        else:\n",
    "            y[i] = 0\n",
    "    return y\n",
    "\n",
    "y_train_binary = to_binary_class(copy.deepcopy(y_train))\n",
    "y_test_binary = to_binary_class(copy.deepcopy(y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### GBDT Classifier"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build the input function.\n",
    "train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(\n",
    "    x={'x': x_train}, y=y_train_binary,\n",
    "    batch_size=batch_size, num_epochs=None, shuffle=True)\n",
    "test_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(\n",
    "    x={'x': x_test}, y=y_test_binary,\n",
    "    batch_size=batch_size, num_epochs=1, shuffle=False)\n",
    "test_train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(\n",
    "    x={'x': x_train}, y=y_train_binary,\n",
    "    batch_size=batch_size, num_epochs=1, shuffle=False)\n",
    "# GBDT Models from TF Estimator requires 'feature_column' data format.\n",
    "feature_columns = [tf.feature_column.numeric_column(key='x', shape=(num_features,))]"
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
      "WARNING:tensorflow:Using temporary folder as model directory: /tmp/tmp5h6BoR\n",
      "INFO:tensorflow:Using config: {'_save_checkpoints_secs': 600, '_num_ps_replicas': 0, '_keep_checkpoint_max': 5, '_task_type': 'worker', '_global_id_in_cluster': 0, '_is_chief': True, '_cluster_spec': ClusterSpec({}), '_model_dir': '/tmp/tmp5h6BoR', '_protocol': None, '_save_checkpoints_steps': None, '_keep_checkpoint_every_n_hours': 10000, '_service': None, '_session_config': allow_soft_placement: true\n",
      "graph_options {\n",
      "  rewrite_options {\n",
      "    meta_optimizer_iterations: ONE\n",
      "  }\n",
      "}\n",
      ", '_tf_random_seed': None, '_save_summary_steps': 100, '_device_fn': None, '_session_creation_timeout_secs': 7200, '_experimental_distribute': None, '_num_worker_replicas': 1, '_task_id': 0, '_log_step_count_steps': 100, '_experimental_max_worker_delay_secs': None, '_evaluation_master': '', '_eval_distribute': None, '_train_distribute': None, '_master': ''}\n"
     ]
    }
   ],
   "source": [
    "gbdt_classifier = tf.estimator.BoostedTreesClassifier(\n",
    "    n_batches_per_layer=num_batches_per_layer,\n",
    "    feature_columns=feature_columns, \n",
    "    n_classes=num_classes,\n",
    "    learning_rate=learning_rate, \n",
    "    n_trees=num_trees,\n",
    "    max_depth=max_depth,\n",
    "    l1_regularization=l1_regul, \n",
    "    l2_regularization=l2_regul\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From /home/user/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow_core/python/ops/resource_variable_ops.py:1635: calling __init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "If using Keras pass *_constraint arguments to layers.\n",
      "WARNING:tensorflow:From /home/user/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow_core/python/training/training_util.py:236: initialized_value (from tensorflow.python.ops.variables) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Use Variable.read_value. Variables in 2.X are initialized automatically both in eager and graph (inside tf.defun) contexts.\n",
      "WARNING:tensorflow:From /home/user/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow_estimator/python/estimator/inputs/queues/feeding_queue_runner.py:62: __init__ (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "To construct input pipelines, use the `tf.data` module.\n",
      "WARNING:tensorflow:From /home/user/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow_estimator/python/estimator/inputs/queues/feeding_functions.py:500: add_queue_runner (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "To construct input pipelines, use the `tf.data` module.\n",
      "INFO:tensorflow:Calling model_fn.\n",
      "INFO:tensorflow:Done calling model_fn.\n",
      "INFO:tensorflow:Create CheckpointSaverHook.\n",
      "WARNING:tensorflow:Issue encountered when serializing resources.\n",
      "Type is unsupported, or the types of the items don't match field type in CollectionDef. Note this is a warning and probably safe to ignore.\n",
      "'_Resource' object has no attribute 'name'\n",
      "INFO:tensorflow:Graph was finalized.\n",
      "INFO:tensorflow:Running local_init_op.\n",
      "INFO:tensorflow:Done running local_init_op.\n",
      "WARNING:tensorflow:From /home/user/anaconda3/envs/py2/lib/python2.7/site-packages/tensorflow_core/python/training/monitored_session.py:906: start_queue_runners (from tensorflow.python.training.queue_runner_impl) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "To construct input pipelines, use the `tf.data` module.\n",
      "WARNING:tensorflow:Issue encountered when serializing resources.\n",
      "Type is unsupported, or the types of the items don't match field type in CollectionDef. Note this is a warning and probably safe to ignore.\n",
      "'_Resource' object has no attribute 'name'\n",
      "INFO:tensorflow:Saving checkpoints for 0 into /tmp/tmp5h6BoR/model.ckpt.\n",
      "WARNING:tensorflow:Issue encountered when serializing resources.\n",
      "Type is unsupported, or the types of the items don't match field type in CollectionDef. Note this is a warning and probably safe to ignore.\n",
      "'_Resource' object has no attribute 'name'\n",
      "INFO:tensorflow:loss = 0.6931475, step = 0\n",
      "WARNING:tensorflow:It seems that global step (tf.train.get_global_step) has not been increased. Current value (could be stable): 0 vs previous value: 0. You could increase the global step by passing tf.train.get_global_step() to Optimizer.apply_gradients or Optimizer.minimize.\n",
      "WARNING:tensorflow:It seems that global step (tf.train.get_global_step) has not been increased. Current value (could be stable): 0 vs previous value: 0. You could increase the global step by passing tf.train.get_global_step() to Optimizer.apply_gradients or Optimizer.minimize.\n",
      "WARNING:tensorflow:It seem
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/2_BasicModels/gradient_boosted_trees.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/2_BasicModels/`, this file is part of the TensorFlow 2.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 16
- Code cells: 12
- Markdown cells: 4



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v2/notebooks/2_BasicModels/gradient_boosted_trees.ipynb
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

- [linear_regression.ipynb](linear_regression.ipynb_docs.md)
- [logistic_regression.ipynb](logistic_regression.ipynb_docs.md)
- [word2vec.ipynb](word2vec.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

dataset, classification, training, regression, gradient, accuracy, loss, tensorflow, optimizer

---

*This documentation was automatically generated for comprehensive repository understanding.*
