# Documentation: notebooks/2_BasicModels/linear_regression_eager_api.ipynb

## File Metadata

- **File Path**: `notebooks/2_BasicModels/linear_regression_eager_api.ipynb`
- **File Size**: 27131 bytes
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
    "# Linear Regression with Eager API\n",
    "\n",
    "A linear regression implemented using TensorFlow's Eager API.\n",
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
    "from __future__ import absolute_import, division, print_function\n",
    "\n",
    "import matplotlib.pyplot as plt\n",
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
    "# Set Eager API\n",
    "tf.enable_eager_execution()\n",
    "tfe = tf.contrib.eager"
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
    "# Training Data\n",
    "train_X = [3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,\n",
    "           7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1]\n",
    "train_Y = [1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,\n",
    "           2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3]\n",
    "n_samples = len(train_X)\n",
    "\n",
    "# Parameters\n",
    "learning_rate = 0.01\n",
    "display_step = 100\n",
    "num_steps = 1000"
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
    "# Weight and Bias\n",
    "W = tfe.Variable(np.random.randn())\n",
    "b = tfe.Variable(np.random.randn())\n",
    "\n",
    "# Linear regression (Wx + b)\n",
    "def linear_regression(inputs):\n",
    "    return inputs * W + b\n",
    "\n",
    "# Mean square error\n",
    "def mean_square_fn(model_fn, inputs, labels):\n",
    "    return tf.reduce_sum(tf.pow(model_fn(inputs) - labels, 2)) / (2 * n_samples)"
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
    "# SGD Optimizer\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\n",
    "\n",
    "# Compute gradients\n",
    "grad = tfe.implicit_gradients(mean_square_fn)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Initial cost= 31.307329178 W= -0.7870768 b= -0.2507985\n",
      "Epoch: 0001 cost= 9.502781868 W= -0.26173288 b= -0.17560114\n",
      "Epoch: 0100 cost= 0.114994615 W= 0.36224815 b= 0.014603348\n",
      "Epoch: 0200 cost= 0.106785327 W= 0.34959725 b= 0.104292504\n",
      "Epoch: 0300 cost= 0.100346453 W= 0.33839324 b= 0.1837239\n",
      "Epoch: 0400 cost= 0.095296182 W= 0.32847065 b= 0.25407064\n",
      "Epoch: 0500 cost= 0.091335081 W= 0.3196829 b= 0.3163719\n",
      "Epoch: 0600 cost= 0.088228233 W= 0.31190023 b= 0.37154746\n",
      "Epoch: 0700 cost= 0.085791394 W= 0.30500764 b= 0.42041263\n",
      "Epoch: 0800 cost= 0.083880097 W= 0.2989034 b= 0.46368918\n",
      "Epoch: 0900 cost= 0.082380980 W= 0.2934973 b= 0.50201607\n",
      "Epoch: 1000 cost= 0.081205189 W= 0.28870946 b= 0.5359594\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAgkAAAFkCAYAAACq4KjhAAAABHNCSVQICAgIfAhkiAAAAAlwSFlz\nAAAPYQAAD2EBqD+naQAAIABJREFUeJzt3Xl8VNX9//HXZzASEwKIoFgEE0FpWqs2cWEHEZFaQCy1\nLUpVKtaliF8Ut0IVa+KumFa0WrVQbekifiu48VWpG6T4I9SVuFQ2t4IsQliNzPn9cbNNMoHMemcm\n7+fjkcfDe+bO3M8IZN5zzrnnmHMOERERkcYCfhcgIiIiqUkhQURERMJSSBAREZGwFBJEREQkLIUE\nERERCUshQURERMJSSBAREZGwFBJEREQkLIUEERERCUshQURERMKKKSSY2bVmFjSzu/dx3hAzqzCz\nXWb2gZmdF8t1RUREJPGiDglmdgLwc+DNfZyXDzwFvAgcC5QBD5nZqdFeW0RERBIvqpBgZu2Ax4CJ\nwJf7OP0SYKVz7mrn3PvOuVnA48CUaK4tIiIiyRFtT8IsYIFzblELzu0DvNCobSHQN8pri4iISBLs\nF+kTzOwnwHHA8S18SldgXaO2dUB7M2vrnNsd5hoHAacBq4FdkdYoIiLSimUD+cBC59zGWF4oopBg\nZocB9wDDnHPVsVx4H04D/pTA1xcREcl05wB/juUFIu1JKAa6AMvNzGra2gCDzGwS0NY55xo957/A\nIY3aDgG2hutFqLEa4LHHHqOwsDDCElPPlClTmDlzpt9lxI3eT+rKpPcCej+pLJPeC2TW+6msrGT8\n+PFQ81kai0hDwgvAdxq1zQYqgVvDBASAcuB7jdqG17Q3ZxdAYWEhRUVFEZaYejp06JAR76OW3k/q\nyqT3Ano/qSyT3gtk3vupEfNwfUQhwTm3HVjRsM3MtgMbnXOVNcc3A92cc7VrIfwO+IWZ3QY8ApwC\n/BA4PcbaRUREJIHiseJi496DQ4HudQ86txr4PjAMeAPv1scLnHON73gQERGRFBLx3Q2NOeeGNjqe\nEOacV/DmM4iIiEia0N4NSTBu3Di/S4grvZ/UlUnvBfR+UlkmvRfIvPcTLxZ+rqG/zKwIqKioqMjE\niSQiIiIJs3z5coqLiwGKnXPLY3mtmIcbREQy0dq1a9mwYYPfZYg00blzZ3r06JGUaykkiIg0snbt\nWgoLC9mxY4ffpYg0kZOTQ2VlZVKCgkKCiEgjGzZsYMeOHRmzoJtkjtqFkjZs2KCQICLip0xZ0E0k\nWrq7QURERMJSSBAREZGwFBJEREQkLIUEERERCUshQUREojZjxgwCgeg+SmbPnk0gEGDt2rVxrqre\nmjVrCAQC/PGPf4zq+cmoMZUpJIiItEIrVqxg/PjxHHbYYWRnZ9OtWzfGjx/PihUr9v3kBsws6pBg\nZphZVM9NllhqnDt3LmVlZXGuKLkUEkREWpknnniCoqIi/vnPf/Kzn/2M+++/n4kTJ/LSSy9RVFTE\nk08+2eLX+tWvfhX1olPnnnsuO3fuTNrqgcn25z//Oe1DgtZJEBGJA+dcwr4Vx/O1V65cybnnnkuv\nXr145ZVX6NSpU91jl19+OQMGDOCnP/0pb731Fvn5+c2+zo4dO8jJySEQCLD//vtHVYuZRf1cSQ71\nJIiIRKmqqoobJk9mWEEBY7p3Z1hBATdMnkxVVVXKvvbtt9/Ozp07efDBB0MCAkCnTp144IEH2LZt\nG7fffntde+28g8rKSs4++2w6derEwIEDQx5raNeuXUyePJkuXbrQvn17xowZw2effUYgEODXv/51\n3Xnhxvvz8/MZPXo0ixcv5qSTTuKAAw6gZ8+ePProoyHX2Lx5M1OnTuWYY44hLy+PDh06cPrpp/PW\nW29F/f9mxYoVDB06lJycHLp3705paSnBYLDJefPnz2fkyJF069aN7OxsevXqRUlJSci5J598Mk8/\n/XTdnIhAIMARRxwBQHV1Nddffz3HH388HTt2pF27dgwaNIiXXnop6toTRT0JIiJRqKqqYmzfvlxR\nWcmMYBADHLBw1izGLlrEvPJy8vLyUu61n3rqKfLz8+nXr1/YxwcOHEh+fj5PP/10XVttL8ZZZ53F\nUUcdxS233ELtDsLhxuzPO+88Hn/8cc4991xOOukkXn75Zb7//e83OS/cc82MDz/8kLPOOosLLriA\n888/n0ceeYQJEyZw/PHH1y2TvXLlSubPn89ZZ51FQUEB69at44EHHmDIkCGsWLGCrl27RvT/Zd26\ndQwZMoRgMMgvf/lLcnJyePDBB8nOzm5y7uzZs8nLy+PKK6+kXbt2LFq0iOuvv56qqipuu+02AKZP\nn86WLVv49NNPueeee3DO0a5dOwC2bt3KI488wrhx4/j5z39OVVUVDz/8MCNGjOD111/nmGOOiaj2\nhHLOpdwPUAS4iooKJyKSbBUVFW5fv4Ouv+wy92wg4Bw0+XkmEHA3TJ4c9fUT9dpbtmxxZubOPPPM\nvZ53xhlnuEAg4LZt2+acc27GjBnOzNz48eObnDtjxgwXCATqjpcvX+7MzF155ZUh502YMMEFAgF3\n44031rXNnj3bBQIBt2bNmrq2/Px8FwgE3OLFi+vavvjiC5edne2uuuqquravvvqqSS1r1qxx2dnZ\nrqSkpK5t9erVzszcnDlz9vqe/+d//scFAgG3bNmyurYNGza4jh07Nqlx165dTZ5/8cUXu3bt2oXU\nNXLkSFdQUNDk3GAw6Kqrq0PatmzZ4rp27eomTpy41zpb8nez9hygyMX4eazhBhGRKCxesIDTwnRF\nA4wIBlk8f37KvXbtUMW+eiFqH9+6dWtdm5lx0UUX7fMazz33HGbGJZdcEtJ+2WWX1fU+7Mu3vvWt\nkJ6Ozp0707t3b1auXFnXlpWVVfffwWCQTZs2kZOTQ+/evVm+fHmLrtPQs88+S58+fSguLq5rO+ig\ngzjnnHOanNu2bdu6/962bRsbN25kwIAB7Nixg/fee2+f1zIz9tvP68h3zrF582a++uorjj/++Khq\nTySFBBGRCDnnyK2uprmphAbkVFe3+EMxWa9d++G/r3kNzYWJgoKCfV6jdgy+8bm9evVqcZ3h7nY4\n8MAD2bx5c92xc46ZM2dy1FFH0bZtWzp37szBBx/M22+/zZYtW1p8rYZ1H3nkkU3ae/fu3aRtxYoV\nnHnmmXTs2JH27dvTpUsXfvrTnwK0+Npz5szh2GOPJTs7m4MOOoiDDz6Yp59+OqraE0lzEkREImRm\nbM/KwkHYD3MHbM/KiuqOhES+dvv27Tn00EP3Obnvrbfeolu3bnVj6LUOOOCAiK8ZjTZt2oRtbxiM\nSktLuf7665k4cSIlJSV06tSJQCDA5ZdfHnayYbxs2bKFQYMG0bFjR0pKSjjiiCPIzs6moqKCa6+9\ntkXXfuyxx5gwYQI/+MEPuPrqqzn44INp06YNN998c0hvSSpQSBARiUL/UaNYOGsWI8J8KDwXCDBg\n9OiUfO2RI0fy0EMPsWTJkrCTF1999VVWr17dZLigpQ4//HCCwSCrVq2iZ8+ede0ffvhh1DWHM2/e\nPIYOHcqDDz4Y0v7ll1/SpUuXiF/v8MMPD1tj4+GDl156ic2bN/Pkk0/Sv3//uvaPPvqoyXObC3Lz\n5s2jZ8+ePP744yHt119/fcR1J5qGG0REojC1tJS7Cwt5NhCg9vutA54NBJhZWMiVJSUp+dpXXXUV\n2dnZXHTRRWzatCnksU2bNnHxxReTm5vL1KlTo3r90047Decc9913X0j7b3/727iuI9GmTZsmQy5/\n//vf+fTTT6N6vdNPP51//etfLFu2rK7tiy++4M9//nPY6zbsMfjqq6+avF+A3NzcsMMH4XpKli5d\nSnl5eVS1J5J6EkREopCXl8e88nLumj6du+fPJ6e6mh1ZWfQfPZp5JSVR36KY6Nfu1asXc+bMYfz4\n8XznO9/hggsuoKCggFWrVvHII4+wceNG/vKXv7Ro/kE4RUVFjB07lnvuuYcNGzbQp08fXn755bpv\n6fEKCiNHjuSmm27iZz/7Gf369ePtt9/mT3/6U0jvRSSuvvpqHn30UU477TQuv/xycnJy+P3vf09+\nfn7I8Ey/fv048MADOffcc5k8eTLgDR+Ee1/FxcX87W9/48orr+SEE06gXbt2jBw5kpEjR/LEE08w\nZswYvv/977Ny5UoeeOABvv3tb7Nt27bo/ockSqy3RyTiB90CKSI+asltZo0Fg8GE1ZOI137nnXfc\nOeec47p16+batm3rvvGNb7jx48e7d999t8m5tbc5bty4Mexjbdq0CWnbuXOnu+yyy1znzp1dXl6e\nGzNmjPvggw+cmbnbb7+97rxwt0AWFBS40aNHN7nOkCFD3NChQ+uOd+/e7a666irXrVs3l5ub6wYN\nGuSWLl3qTj755JDzVq9e7QKBwD5vgaz9f3LyySe7nJwc1717d3fzzTe7Rx55pEmN5eXlrl+/fi43\nN9cddthh7rrrrnPPP/+8CwQC7uWXX647b/v27W78+PGuU6dOLhAIhNwOeeutt7qCggJ3wAEHuOLi\nYvfMM8+4888/3x1xxBF7rTHZt0Cai2KGbKKZWRFQUVFRQVFRkd/liEgrs3z5coqLi9HvoPh54403\nKCoq4k9/+hPjxo3zu5y01ZK/m7XnAMXOuZjuqdScBBERiatdu3Y1abvnnnto06YNgwYN8qEiiZbm\nJIiISFzdfvvtVFRUcPLJJ7PffvvxzDPPsHDhQi666CK6devmd3kSAYUEERGJq379+vHCCy9QUlLC\ntm3b6NGjBzfeeCO//OUv/S5NIqSQICIicTVs2DCGDRvmdxkSB5qTICIiImEpJIiIiEhYCgkiIiIS\nlkKCiIiIhKWQICIiImEpJIiIiEhYCgkiIiISlkKCiIg0cdhhh/Hzn//c1xo++ugjAoFAk+2aG3vx\nxRcJBAIsWbKkrm38+PEceeSRiS4x4ykkiIi0InPmzCEQCIT9abgiYiAQCNn++N133+XGG2/kk08+\nafKas2bN4tFHH01K/c1pvFWzmREI6CMuVlpxUUSklTEzbrrpJvLz80Pajz766Lr//uijj2jTpk3d\n8TvvvMONN97IqaeeymGHHRbyvHvvvZfu3bvz05/+NKF1R2L27Nmk4i7H6UYhQUSkFRoxYsRet8HO\nysoKOXbONfm2nsoaBhyJnvpiRESkiYZzEh5++GHOPvtsAAYMGEAgEKBNmzYsWbKE7t278/777/PC\nCy/UDVsMHz687nW+/PJLJk+eTI8ePcjOzuaoo47izjvvbHK9zZs3c+6559KxY0c6derEBRdcwNat\nW6Ouv/GchNr5Db/5zW944IEH6NmzJwcccAB9+vTh3//+d5PnV1ZWMnbsWA466CBycnI48cQTeeaZ\nZ6KuJ11F1JNgZhcDlwD5NU3vAr92zj3XzPmDgX82anbAoc659ZGVKiIi8bJlyxY2btwY0nbQQQfV\n/XfDXoOTTz6ZX/ziF9x3333ccMMNdR++vXv35t577+XSSy/loIMO4rrrrsM5x6GHHgrAjh07GDhw\nIOvXr+fiiy/msMMO47XXXuPqq69m/fr13H777YDXSzFq1CiWLl3KpZdeSu/evZk3bx4TJkyIuvfC\nzMI+d86cOezYsYNLL70U5xy33XYbY8eO5T//+U/dHIa3336bgQMHcvjhh3PdddeRk5PDX//6V0aP\nHs0//vEPRo4cGVVN6SjS4YaPgWuADwEDzgeeNLPjnHOVzTzHAUcBVXUNCggiIr5xznHKKaeEtJkZ\ne/bsCXv+EUccwYABA7jvvvs49dRT6devX91jZ5xxBtdeey1du3Zl3LhxIc+7/fbbWbt2LW+++Wbd\n/IcLL7yQQw45hLKyMq644gq6du3KE088wZIlS7jnnnuYPHkyABdffDGDBg2K47v2fPrpp/znP/+h\nXbt2APTs2ZMf/vCHvPDCC3U9IJdddhm9evVi6dKldcMWl156KX369OHaa69VSGiOc+7pRk3TzewS\noA/QXEgA+MI5F32/kYhICtuxA957L7HX+OY3IScnPq9lZtx3330Jv0Xw8ccfZ8iQIeTl5YX0Wgwb\nNow777yTV199lbPOOotnnnmGtm3bhtxyGQgEmDRpUshtjfFw9tln1wUEgIEDB+KcY+XKlQBs2LCB\nV155hVtvvZUvv/yy7jznHKeddholJSV88cUXdOnSJa51paqoJy6aWQD4EZADlO/tVOANM8sG3gFm\nOOfi+6cuIuKj996D4uLEXqOiAvYyzzBiJ5xwwl4nLsbDhx9+SGVlZdgPVDNj/XqvU3nt2rV069aN\n7OzskHN69+4d95q6d+8ecnzggQcC3pyI2poBrrvuOq699tpm61ZIaIaZHY0XCrLxhhDOdM41l6E/\nBy4ClgFtgQuBl8zsROfcG9GVLCKSWr75Te9DPNHXSDfOOUaMGMGVV14Z9vFEhI
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `notebooks/2_BasicModels/linear_regression_eager_api.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `notebooks/` directory, specifically within `notebooks/2_BasicModels/`, this file is part of the notebook-based tutorials.



## Detailed Walkthrough

### Notebook Structure

This Jupyter notebook contains interactive code cells demonstrating TensorFlow concepts.

- Total cells: 7
- Code cells: 6
- Markdown cells: 1



## Inline Code Examples

### Example Usage

Open this notebook in Jupyter:

```bash
jupyter notebook notebooks/2_BasicModels/linear_regression_eager_api.ipynb
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

- **Graph Mode**: This uses eager execution; graph mode (TF 1.x style) is an alternative
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
- [logistic_regression.ipynb](logistic_regression.ipynb_docs.md)
- [logistic_regression_eager_api.ipynb](logistic_regression_eager_api.ipynb_docs.md)
- [nearest_neighbor.ipynb](nearest_neighbor.ipynb_docs.md)
- [random_forest.ipynb](random_forest.ipynb_docs.md)
- [word2vec.ipynb](word2vec.ipynb_docs.md)

Related implementations:

- Check `examples/` directory for Python script versions

See the [folder index](./index.md) for a complete list of related files.



## Keywords

rnn, training, regression, gradient, gan, tensorflow, optimizer, cnn

---

*This documentation was automatically generated for comprehensive repository understanding.*
