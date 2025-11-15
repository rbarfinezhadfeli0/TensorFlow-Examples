# Documentation: helloworld.ipynb

## File Metadata
- **Path**: `tensorflow_v2/notebooks/1_Introduction/helloworld.ipynb`
- **Size**: 1,541 bytes
- **Extension**: .ipynb
- **Type**: Jupyter Notebook

---

## Original Source

```
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Hello World\n",
    "\n",
    "A very simple \"hello world\" using TensorFlow v2 tensors.\n",
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
    "import tensorflow as tf"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tf.Tensor(hello world, shape=(), dtype=string)\n"
     ]
    }
   ],
   "source": [
    "# Create a Tensor.\n",
    "hello = tf.constant(\"hello world\")\n",
    "print(hello)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "hello world\n"
     ]
    }
   ],
   "source": [
    "# To access a Tensor value, call numpy().\n",
    "print(hello.numpy())"
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

---

## High-Level Overview

This is a Jupyter Notebook.

### Notebook Structure

- **Total Cells**: 4
- **Code Cells**: 3
- **Markdown Cells**: 1

### Description

# Hello World

A very simple "hello world" using TensorFlow v2 tensors.

- Author: Aymeric Damien
- Project: https://github.com/aymericdamien/TensorFlow-Examples/



---

## Related Files

*(Links to related files will be added during cross-reference phase)*
