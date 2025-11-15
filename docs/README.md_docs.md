# Documentation: README.md

## File Metadata

- **File Path**: `README.md`
- **File Size**: 22407 bytes
- **File Type**: .md
- **Purpose**: Repository documentation and guide

## Original Source

```markdown
# TensorFlow Examples

This tutorial was designed for easily diving into TensorFlow, through examples. For readability, it includes both notebooks and source codes with explanation, for both TF v1 & v2.

It is suitable for beginners who want to find clear and concise examples about TensorFlow. Besides the traditional 'raw' TensorFlow implementations, you can also find the latest TensorFlow API practices (such as `layers`, `estimator`, `dataset`, ...).

**Update (05/16/2020):** Moving all default examples to TF2. For TF v1 examples: [check here](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1).

## Tutorial index

#### 0 - Prerequisite
- [Introduction to Machine Learning](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/0_Prerequisite/ml_introduction.ipynb).
- [Introduction to MNIST Dataset](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).

#### 1 - Introduction
- **Hello World** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/helloworld.ipynb)). Very simple example to learn how to print "hello world" using TensorFlow 2.0+.
- **Basic Operations** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/1_Introduction/basic_operations.ipynb)). A simple example that cover TensorFlow 2.0+ basic operations.

#### 2 - Basic Models
- **Linear Regression** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/linear_regression.ipynb)). Implement a Linear Regression with TensorFlow 2.0+.
- **Logistic Regression** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/logistic_regression.ipynb)). Implement a Logistic Regression with TensorFlow 2.0+.
- **Word2Vec (Word Embedding)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/word2vec.ipynb)). Build a Word Embedding Model (Word2Vec) from Wikipedia data, with TensorFlow 2.0+.
- **GBDT (Gradient Boosted Decision Trees)** ([notebooks](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/2_BasicModels/gradient_boosted_trees.ipynb)). Implement a Gradient Boosted Decision Trees with TensorFlow 2.0+ to predict house value using Boston Housing dataset.

#### 3 - Neural Networks
##### Supervised

- **Simple Neural Network** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network.ipynb)). Use TensorFlow 2.0 'layers' and 'model' API to build a simple neural network to classify MNIST digits dataset.
- **Simple Neural Network (low-level)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/neural_network_raw.ipynb)). Raw implementation of a simple neural network to classify MNIST digits dataset.
- **Convolutional Neural Network** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network.ipynb)). Use TensorFlow 2.0+ 'layers' and 'model' API to build a convolutional neural network to classify MNIST digits dataset.
- **Convolutional Neural Network (low-level)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/convolutional_network_raw.ipynb)). Raw implementation of a convolutional neural network to classify MNIST digits dataset.
- **Recurrent Neural Network (LSTM)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/recurrent_network.ipynb)). Build a recurrent neural network (LSTM) to classify MNIST digits dataset, using TensorFlow 2.0 'layers' and 'model' API.
- **Bi-directional Recurrent Neural Network (LSTM)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/bidirectional_rnn.ipynb)). Build a bi-directional recurrent neural network (LSTM) to classify MNIST digits dataset, using TensorFlow 2.0+ 'layers' and 'model' API.
- **Dynamic Recurrent Neural Network (LSTM)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/dynamic_rnn.ipynb)). Build a recurrent neural network (LSTM) that performs dynamic calculation to classify sequences of variable length, using TensorFlow 2.0+ 'layers' and 'model' API.

##### Unsupervised
- **Auto-Encoder** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/autoencoder.ipynb)). Build an auto-encoder to encode an image to a lower dimension and re-construct it.
- **DCGAN (Deep Convolutional Generative Adversarial Networks)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/3_NeuralNetworks/dcgan.ipynb)). Build a Deep Convolutional Generative Adversarial Network (DCGAN) to generate images from noise.

#### 4 - Utilities
- **Save and Restore a model** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/4_Utils/save_restore_model.ipynb)). Save and Restore a model with TensorFlow 2.0+.
- **Build Custom Layers & Modules** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/4_Utils/build_custom_layers.ipynb)). Learn how to build your own layers / modules and integrate them into TensorFlow 2.0+ Models.
- **Tensorboard** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/4_Utils/tensorboard.ipynb)). Track and visualize neural network computation graph, metrics, weights and more using TensorFlow 2.0+ tensorboard.

#### 5 - Data Management
- **Load and Parse data** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/load_data.ipynb)). Build efficient data pipeline with TensorFlow 2.0 (Numpy arrays, Images, CSV files, custom data, ...).
- **Build and Load TFRecords** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/tfrecords.ipynb)). Convert data into TFRecords format, and load them with TensorFlow 2.0+.
- **Image Transformation (i.e. Image Augmentation)** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/5_DataManagement/image_transformation.ipynb)). Apply various image augmentation techniques with TensorFlow 2.0+, to generate distorted images for training.

#### 6 - Hardware
- **Multi-GPU Training** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v2/notebooks/6_Hardware/multigpu_training.ipynb)). Train a convolutional neural network with multiple GPUs on CIFAR-10 dataset.

## TensorFlow v1

The tutorial index for TF v1 is available here: [TensorFlow v1.15 Examples](tensorflow_v1). Or see below for a list of the examples.

## Dataset
Some examples require MNIST dataset for training and testing. Don't worry, this dataset will automatically be downloaded when running examples.
MNIST is a database of handwritten digits, for a quick description of that dataset, you can check [this notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/0_Prerequisite/mnist_dataset_intro.ipynb).

Official Website: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/).

## Installation

To download all the examples, simply clone this repository:
```
git clone https://github.com/aymericdamien/TensorFlow-Examples
```

To run them, you also need the latest version of TensorFlow. To install it:
```
pip install tensorflow
```

or (with GPU support):
```
pip install tensorflow_gpu
```

For more details about TensorFlow installation, you can check [TensorFlow Installation Guide](https://www.tensorflow.org/install/)


## TensorFlow v1 Examples - Index

The tutorial index for TF v1 is available here: [TensorFlow v1.15 Examples](tensorflow_v1).

#### 0 - Prerequisite
- [Introduction to Machine Learning](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/tensorflow_v1/0_Prerequisite/ml_introduction.ipynb).
- [Introduction to MNIST Dataset](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/tensorflow_v1/0_Prerequisite/mnist_dataset_intro.ipynb).

#### 1 - Introduction
- **Hello World** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/1_Introduction/helloworld.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/1_Introduction/helloworld.py)). Very simple example to learn how to print "hello world" using TensorFlow.
- **Basic Operations** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/tensorflow_v1/1_Introduction/basic_operations.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-examples/Examples/blob/master/tensorflow_v1/1_Introduction/basic_operations.py)). A simple example that cover TensorFlow basic operations.
- **TensorFlow Eager API basics** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/1_Introduction/basic_eager_api.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/examples/1_Introduction/basic_eager_api.py)). Get started with TensorFlow's Eager API.

#### 2 - Basic Models
- **Linear Regression** ([notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/tensorflow_v1/notebooks/2_BasicModels/linear_regression.ipynb)) ([code](https://github.com/aymericdamien/TensorFlow-Examples/blob
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `README.md` within the TensorFlow Examples repository.

### Purpose

This file provides documentation in Markdown format.

### Context



## Detailed Walkthrough

### Document Structure

This document contains the following sections:

- TensorFlow Examples
  - Tutorial index
      - 0 - Prerequisite
      - 1 - Introduction
      - 2 - Basic Models
      - 3 - Neural Networks
        - Supervised
        - Unsupervised
      - 4 - Utilities
      - 5 - Data Management
      - 6 - Hardware
  - TensorFlow v1
  - Dataset
  - Installation
  - TensorFlow v1 Examples - Index
      - 0 - Prerequisite
      - 1 - Introduction
      - 2 - Basic Models
      - 3 - Neural Networks
        - Supervised
        - Unsupervised
      - 4 - Utilities
      - 5 - Data Management
      - 6 - Multi GPU
  - More Examples
    - Tutorials
    - Examples



## Inline Code Examples

### Example Usage



## Design & Architecture

### Architectural Context

### Design Patterns

- Uses TensorFlow framework for machine learning operations


## Performance & Complexity

### Performance Characteristics

- **GPU Acceleration**: This code is designed to leverage GPU hardware for accelerated computation
- **Scalability**: Implements multi-GPU training for handling large-scale models and datasets
- **Computational Complexity**: Neural network operations can be computationally intensive
- **Memory Usage**: Deep learning models require significant memory for parameters and activations

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

### Usage Notes

- Ensure TensorFlow is installed: `pip install tensorflow`
- Some examples may require additional dependencies
- GPU support is optional but recommended for large models



## Related Files

### Related Files in Repository

Files in the same directory:

- [LICENSE](LICENSE_docs.md)
- [generate_docs.py](generate_docs.py_docs.md)
- [input_data.py](input_data.py_docs.md)


See the [folder index](./index.md) for a complete list of related files.



## Keywords

rnn, training, 4 - Utilities, gradient, neural network, Supervised, convolution, Dataset, Examples, 3 - Neural Networks, Tutorial index, mnist, TensorFlow v1, 6 - Hardware, TensorFlow Examples, cifar, 1 - Introduction, 0 - Prerequisite, regression, Installation, embedding, Tutorials, lstm, loss, TensorFlow v1 Examples - Index, More Examples, Unsupervised, 2 - Basic Models, 6 - Multi GPU, autoencoder, dataset, testing, gan, tensorflow, cnn, 5 - Data Management

---

*This documentation was automatically generated for comprehensive repository understanding.*
