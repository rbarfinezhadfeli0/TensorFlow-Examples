# Documentation: tensorflow_v2/notebooks/5_DataManagement/load_data.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/5_DataManagement/load_data.ipynb`
- **File Size**: 19052 bytes
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
    "# Load and parse data with TensorFlow 2.0 (tf.data)\n",
    "\n",
    "A TensorFlow 2.0 example to build input pipelines for loading data efficiently.\n",
    "\n",
    "\n",
    "- Numpy Arrays\n",
    "- Images\n",
    "- CSV file\n",
    "- Custom data from a Generator\n",
    "\n",
    "For more information about creating and loading TensorFlow's `TFRecords` data format, see: [tfrecords.ipynb](tfrecords.ipynb)\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from __future__ import absolute_import, division, print_function\n",
    "\n",
    "import numpy as np\n",
    "import random\n",
    "import requests\n",
    "import string\n",
    "import tarfile\n",
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load Numpy Arrays\n",
    "\n",
    "Build a data pipeline over numpy arrays."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a toy dataset (even and odd numbers, with respective labels of 0 and 1).\n",
    "evens = np.arange(0, 100, step=2, dtype=np.int32)\n",
    "evens_label = np.zeros(50, dtype=np.int32)\n",
    "odds = np.arange(1, 100, step=2, dtype=np.int32)\n",
    "odds_label = np.ones(50, dtype=np.int32)\n",
    "# Concatenate arrays\n",
    "features = np.concatenate([evens, odds])\n",
    "labels = np.concatenate([evens_label, odds_label])\n",
    "\n",
    "# Load a numpy array using tf data api with `from_tensor_slices`.\n",
    "data = tf.data.Dataset.from_tensor_slices((features, labels))\n",
    "# Refill data indefinitely.  \n",
    "data = data.repeat()\n",
    "# Shuffle data.\n",
    "data = data.shuffle(buffer_size=100)\n",
    "# Batch data (aggregate records together).\n",
    "data = data.batch(batch_size=4)\n",
    "# Prefetch batch (pre-load batch for faster consumption).\n",
    "data = data.prefetch(buffer_size=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tf.Tensor([ 9 94 29 85], shape=(4,), dtype=int32) tf.Tensor([1 0 1 1], shape=(4,), dtype=int32)\n",
      "tf.Tensor([68 57 88 41], shape=(4,), dtype=int32) tf.Tensor([0 1 0 1], shape=(4,), dtype=int32)\n",
      "tf.Tensor([51 19 18 56], shape=(4,), dtype=int32) tf.Tensor([1 1 0 0], shape=(4,), dtype=int32)\n",
      "tf.Tensor([70 84 99 32], shape=(4,), dtype=int32) tf.Tensor([0 0 1 0], shape=(4,), dtype=int32)\n",
      "tf.Tensor([40  0 25 28], shape=(4,), dtype=int32) tf.Tensor([0 0 1 0], shape=(4,), dtype=int32)\n"
     ]
    }
   ],
   "source": [
    "for batch_x, batch_y in data.take(5):\n",
    "    print(batch_x, batch_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tf.Tensor([ 9 94 29 85], shape=(4,), dtype=int32) tf.Tensor([1 0 1 1], shape=(4,), dtype=int32)\n",
      "tf.Tensor([68 57 88 41], shape=(4,), dtype=int32) tf.Tensor([0 1 0 1], shape=(4,), dtype=int32)\n",
      "tf.Tensor([51 19 18 56], shape=(4,), dtype=int32) tf.Tensor([1 1 0 0], shape=(4,), dtype=int32)\n",
      "tf.Tensor([70 84 99 32], shape=(4,), dtype=int32) tf.Tensor([0 0 1 0], shape=(4,), dtype=int32)\n",
      "tf.Tensor([40  0 25 28], shape=(4,), dtype=int32) tf.Tensor([0 0 1 0], shape=(4,), dtype=int32)\n",
      "tf.Tensor([20 38 22 79], shape=(4,), dtype=int32) tf.Tensor([0 0 0 1], shape=(4,), dtype=int32)\n",
      "tf.Tensor([20 22 96 27], shape=(4,), dtype=int32) tf.Tensor([0 0 0 1], shape=(4,), dtype=int32)\n",
      "tf.Tensor([34 58 86 67], shape=(4,), dtype=int32) tf.Tensor([0 0 0 1], shape=(4,), dtype=int32)\n",
      "tf.Tensor([ 2 98 24 21], shape=(4,), dtype=int32) tf.Tensor([0 0 0 1], shape=(4,), dtype=int32)\n",
      "tf.Tensor([16 45 18 35], shape=(4,), dtype=int32) tf.Tensor([0 1 0 1], shape=(4,), dtype=int32)\n"
     ]
    }
   ],
   "source": [
    "# Note: If you are planning on calling multiple time,\n",
    "# you can user the iterator way:\n",
    "ite_data = iter(data)\n",
    "for i in range(5):\n",
    "    batch_x, batch_y = next(ite_data)\n",
    "    print(batch_x, batch_y)\n",
    "\n",
    "for i in range(5):\n",
    "    batch_x, batch_y = next(ite_data)\n",
    "    print(batch_x, batch_y)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load CSV files\n",
    "\n",
    "Build a data pipeline from features stored in a CSV file. For this example, Titanic dataset will be used as a toy dataset stored in CSV format."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Titanic Dataset\n",
    "\n",
    "\n",
    "\n",
    "survived|pclass|name|sex|age|sibsp|parch|ticket|fare\n",
    "--------|------|----|---|---|-----|-----|------|----\n",
    "1|1|\"Allen, Miss. Elisabeth Walton\"|female|29|0|0|24160|211.3375\n",
    "1|1|\"Allison, Master. Hudson Trevor\"|male|0.9167|1|2|113781|151.5500\n",
    "0|1|\"Allison, Miss. Helen Loraine\"|female|2|1|2|113781|151.5500\n",
    "0|1|\"Allison, Mr. Hudson Joshua Creighton\"|male|30|1|2|113781|151.5500\n",
    "...|...|...|...|...|...|...|...|..."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download Titanic dataset (in csv format).\n",
    "d = requests.get(\"https://raw.githubusercontent.com/tflearn/tflearn.github.io/master/resources/titanic_dataset.csv\")\n",
    "with open(\"titanic_dataset.csv\", \"wb\") as f:\n",
    "    f.write(d.content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Titanic dataset.\n",
    "# Original features: survived,pclass,name,sex,age,sibsp,parch,ticket,fare\n",
    "# Select specific columns: survived,pclass,name,sex,age,fare\n",
    "column_to_use = [0, 1, 2, 3, 4, 8]\n",
    "record_defaults = [tf.int32, tf.int32, tf.string, tf.string, tf.float32, tf.float32]\n",
    "\n",
    "# Load the whole dataset file, and slice each line.\n",
    "data = tf.data.experimental.CsvDataset(\"titanic_dataset.csv\", record_defaults, header=True, select_cols=column_to_use)\n",
    "# Refill data indefinitely.\n",
    "data = data.repeat()\n",
    "# Shuffle data.\n",
    "data = data.shuffle(buffer_size=1000)\n",
    "# Batch data (aggregate records together).\n",
    "data = data.batch(batch_size=2)\n",
    "# Prefetch batch (pre-load batch for faster consumption).\n",
    "data = data.prefetch(buffer_size=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1]\n",
      "[2 2]\n",
      "['Richards, Master. George Sibley' 'Rugg, Miss. Emily']\n",
      "['male' 'female']\n",
      "[ 0.8333 21.    ]\n",
      "[18.75 10.5 ]\n"
     ]
    }
   ],
   "source": [
    "for survived, pclass, name, sex, age, fare in data.take(1):\n",
    "    print(survived.numpy())\n",
    "    print(pclass.numpy())\n",
    "    print(name.numpy())\n",
    "    print(sex.numpy())\n",
    "    print(age.numpy())\n",
    "    print(fare.numpy())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load Images\n",
    "\n",
    "Build a data pipeline by loading images from disk. For this example, Oxford Flowers dataset will be used."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download Oxford 17 flowers dataset\n",
    "d = requests.get(\"http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz\")\n",
    "with open(\"17flowers.tgz\", \"wb\") as f:\n",
    "    f.write(d.content)\n",
    "# Extract archive.\n",
    "with tarfile.open(\"17flowers.tgz\") as t:\n",
    "    t.extractall()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "with open('jpg/dataset.csv', 'w') as f:\n",
    "    c = 0\n",
    "    for i in range(1360):\n",
    "        f.write(\"jpg/image_%04i.jpg,%i\\n\" % (i+1, c))\n",
    "        if (i+1) % 80 == 0:\n",
    "            c += 1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Images\n",
    "with open(\"jpg/dataset.csv\") as f:\n",
    "    dataset_file = f.read().splitlines()\n",
    "\n",
    "# Load the whole dataset file, and slice each line.\n",
    "data = tf.data.Dataset.from_tensor_slices(dataset_file)\n",
    "# Refill data indefinitely.\n",
    "data = data.repeat()\n",
    "# Shuffle data.\n",
    "data = data.shuffle(buffer_size=1000)\n",
    "\n",
    "# Load and pre-process images.\n",
    "def load_image(path):\n",
    "    # Read image from path.\n",
    "    image = tf.io.read_file(path)\n",
    "    # Decode the jpeg image to array [0, 255].\n",
    "    image = tf.image.decode_jpeg(image)\n",
    "    # Resize images to a common size of 256x256.\n",
    "    image = tf.image.resize(image, [256, 256])\n",
    "    # Rescale values to [-1, 1].\n",
    "    image = 1. - image / 127.5\n",
    "    return image\n",
    "# Decode each line from the dataset file.\n",
    "def parse_records(line):\n",
    "    # File is in csv format: \"image_path,label_id\".\n",
    "    # TensorFlow requires a default value, but it will never be used.\n",
    "    image_path, image_label = tf.io.decode_csv(line, [\"\", 0])\n",
    "    # Apply the function to load images.\n",
    "    image = load_image(image_path)\n",
   
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/5_DataManagement/load_data.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/5_DataManagement/`, this file is part of the TensorFlow 2.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 20
- Code cells: 14
- Markdown cells: 6



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v2/notebooks/5_DataManagement/load_data.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It demonstrates data loading and preprocessing techniques.

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

- [image_transformation.ipynb](image_transformation.ipynb_docs.md)
- [tfrecords.ipynb](tfrecords.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

dataset, tensorflow

---

*This documentation was automatically generated for comprehensive repository understanding.*
