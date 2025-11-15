# Documentation: tensorflow_v1/notebooks/5_DataManagement/image_transformation.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/5_DataManagement/image_transformation.ipynb`
- **File Size**: 2458254 bytes
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
    "# Image Transformation (i.e. Image Augmentation)\n",
    "\n",
    "Learn how to apply various image augmentation techniques with TensorFlow. The transformations are meant to be applied for each image sample when training only, and each transformation will be performed with random parameters.\n",
    "\n",
    "**Transformations:**\n",
    "- Random flip left-right\n",
    "- Random contrast, brightness, saturation and hue\n",
    "- Random distortion and crop\n",
    "\n",
    "For more information about loading data, see: [load_data.ipynb](load_data.ipynb)\n",
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
    "from IPython.display import Image as IImage, display\n",
    "import numpy as np\n",
    "import PIL\n",
    "from PIL import Image\n",
    "import random\n",
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
    "# Download an image.\n",
    "d = requests.get(\"https://www.paristoolkit.com/Images/xeffel_view.jpg.pagespeed.ic.8XcZNqpzSj.jpg\")\n",
    "with open(\"image.jpeg\", \"wb\") as f:\n",
    "    f.write(d.content)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load image to numpy array.\n",
    "img = PIL.Image.open('image.jpeg')\n",
    "img.load()\n",
    "img_array = np.array(img)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/5_DataManagement/image_transformation.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/notebooks/5_DataManagement/`, this file is part of the TensorFlow 1.x examples collection.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 22
- Code cells: 21
- Markdown cells: 1



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook tensorflow_v1/notebooks/5_DataManagement/image_transformation.ipynb
```

Then execute cells sequentially to see TensorFlow in action.



## Design & Architecture

### Architectural Context

This file is part of the TensorFlow Examples educational repository structure. It demonstrates data loading and preprocessing techniques.

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

- [build_an_image_dataset.ipynb](build_an_image_dataset.ipynb_docs.md)
- [load_data.ipynb](load_data.ipynb_docs.md)
- [tensorflow_dataset_api.ipynb](tensorflow_dataset_api.ipynb_docs.md)
- [tfrecords.ipynb](tfrecords.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

rnn, dataset, training, relu, gan, loss, tensorflow, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
