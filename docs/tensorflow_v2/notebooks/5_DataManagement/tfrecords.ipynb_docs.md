# Documentation: tensorflow_v2/notebooks/5_DataManagement/tfrecords.ipynb

## File Metadata

- **File Path**: `tensorflow_v2/notebooks/5_DataManagement/tfrecords.ipynb`
- **File Size**: 8372 bytes
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
    "# Create and Load TFRecords\n",
    "\n",
    "A simple TensorFlow 2.0 example to parse a dataset into TFRecord format, and then read that dataset.\n",
    "\n",
    "In this example, the Titanic Dataset (in CSV format) will be used as a toy dataset, for parsing all the dataset features into TFRecord format, and then building an input pipeline that can be used for training models.\n",
    "\n",
    "- Author: Aymeric Damien\n",
    "- Project: https://github.com/aymericdamien/TensorFlow-Examples/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Titanic Dataset\n",
    "\n",
    "The titanic dataset is a popular dataset for ML that provides a list of all passengers onboard the Titanic, along with various features such as their age, sex, class (1st, 2nd, 3rd)... And if the passenger survived the disaster or not.\n",
    "\n",
    "It can be used to see that even though some luck was involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class...\n",
    "\n",
    "#### Overview\n",
    "survived|pclass|name|sex|age|sibsp|parch|ticket|fare\n",
    "--------|------|----|---|---|-----|-----|------|----\n",
    "1|1|\"Allen, Miss. Elisabeth Walton\"|female|29|0|0|24160|211.3375\n",
    "1|1|\"Allison, Master. Hudson Trevor\"|male|0.9167|1|2|113781|151.5500\n",
    "0|1|\"Allison, Miss. Helen Loraine\"|female|2|1|2|113781|151.5500\n",
    "0|1|\"Allison, Mr. Hudson Joshua Creighton\"|male|30|1|2|113781|151.5500\n",
    "...|...|...|...|...|...|...|...|...\n",
    "\n",
    "\n",
    "#### Variable Descriptions\n",
    "```\n",
    "survived        Survived\n",
    "                (0 = No; 1 = Yes)\n",
    "pclass          Passenger Class\n",
    "                (1 = 1st; 2 = 2nd; 3 = 3rd)\n",
    "name            Name\n",
    "sex             Sex\n",
    "age             Age\n",
    "sibsp           Number of Siblings/Spouses Aboard\n",
    "parch           Number of Parents/Children Aboard\n",
    "ticket          Ticket Number\n",
    "fare            Passenger Fare\n",
    "```"
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
    "import csv\n",
    "import requests\n",
    "import tensorflow as tf"
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
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Create TFRecords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Generate Integer Features.\n",
    "def build_int64_feature(data):\n",
    "    return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))\n",
    "\n",
    "# Generate Float Features.\n",
    "def build_float_feature(data):\n",
    "    return tf.train.Feature(float_list=tf.train.FloatList(value=[data]))\n",
    "\n",
    "# Generate String Features.\n",
    "def build_string_feature(data):\n",
    "    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))\n",
    "\n",
    "# Generate a TF `Example`, parsing all features of the dataset.\n",
    "def convert_to_tfexample(survived, pclass, name, sex, age, sibsp, parch, ticket, fare):\n",
    "    return tf.train.Example(\n",
    "        features=tf.train.Features(\n",
    "            feature={\n",
    "                'survived': build_int64_feature(survived),\n",
    "                'pclass': build_int64_feature(pclass),\n",
    "                'name': build_string_feature(name),\n",
    "                'sex': build_string_feature(sex),\n",
    "                'age': build_float_feature(age),\n",
    "                'sibsp': build_int64_feature(sibsp),\n",
    "                'parch': build_int64_feature(parch),\n",
    "                'ticket': build_string_feature(ticket),\n",
    "                'fare': build_float_feature(fare),\n",
    "            })\n",
    "    )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Open dataset file.\n",
    "with open(\"titanic_dataset.csv\") as f:\n",
    "    # Output TFRecord file.\n",
    "    with tf.io.TFRecordWriter(\"titanic_dataset.tfrecord\") as w:\n",
    "        # Generate a TF Example for all row in our dataset.\n",
    "        # CSV reader will read and parse all rows.\n",
    "        reader = csv.reader(f, skipinitialspace=True)\n",
    "        for i, record in enumerate(reader):\n",
    "            # Skip header.\n",
    "            if i == 0:\n",
    "                continue\n",
    "            survived, pclass, name, sex, age, sibsp, parch, ticket, fare = record\n",
    "            # Parse each csv row to TF Example using the above functions.\n",
    "            example = convert_to_tfexample(int(survived), int(pclass), name, sex, float(age), int(sibsp), int(parch), ticket, float(fare))\n",
    "            # Serialize each TF Example to string, and write to TFRecord file.\n",
    "            w.write(example.SerializeToString())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Load TFRecords"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Build features template, with types.\n",
    "features = {\n",
    "    'survived': tf.io.FixedLenFeature([], tf.int64),\n",
    "    'pclass': tf.io.FixedLenFeature([], tf.int64),\n",
    "    'name': tf.io.FixedLenFeature([], tf.string),\n",
    "    'sex': tf.io.FixedLenFeature([], tf.string),\n",
    "    'age': tf.io.FixedLenFeature([], tf.float32),\n",
    "    'sibsp': tf.io.FixedLenFeature([], tf.int64),\n",
    "    'parch': tf.io.FixedLenFeature([], tf.int64),\n",
    "    'ticket': tf.io.FixedLenFeature([], tf.string),\n",
    "    'fare': tf.io.FixedLenFeature([], tf.float32),\n",
    "}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load TFRecord data.\n",
    "filenames = [\"titanic_dataset.tfrecord\"]\n",
    "data = tf.data.TFRecordDataset(filenames)\n",
    "\n",
    "# Parse features, using the above template.\n",
    "def parse_record(record):\n",
    "    return tf.io.parse_single_example(record, features=features)\n",
    "# Apply the parsing to each record from the dataset.\n",
    "data = data.map(parse_record)\n",
    "\n",
    "# Refill data indefinitely.\n",
    "data = data.repeat()\n",
    "# Shuffle data.\n",
    "data = data.shuffle(buffer_size=1000)\n",
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
      "[0 1 0 0]\n",
      "['Gallagher, Mr. Martin' 'Fortune, Miss. Mabel Helen'\n",
      " 'Andersson, Mr. Johan Samuel' 'Jensen, Mr. Niels Peder']\n",
      "[  7.7417 263.       7.775    7.8542]\n"
     ]
    }
   ],
   "source": [
    "# Dequeue data and display.\n",
    "for record in data.take(1):\n",
    "    print(record['survived'].numpy())\n",
    "    print(record['name'].numpy())\n",
    "    print(record['fare'].numpy())"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2",
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
   "version": "2.7.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

```

## High-Level Overview

This file is located at `tensorflow_v2/notebooks/5_DataManagement/tfrecords.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v2/` directory, specifically within `tensorflow_v2/notebooks/5_DataManagement/`, this file is part of the TensorFlow 2.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 11
- Code cells: 7
- Markdown cells: 4



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v2/notebooks/5_DataManagement/tfrecords.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It demonstrates data loading and preprocessing techniques.

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

- [image_transformation.ipynb](image_transformation.ipynb_docs.md)
- [load_data.ipynb](load_data.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

dataset, training, tensorflow

---

*This documentation was automatically generated for comprehensive repository understanding.*
