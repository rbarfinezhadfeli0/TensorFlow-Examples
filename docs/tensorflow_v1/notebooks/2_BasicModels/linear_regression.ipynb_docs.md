# Documentation: tensorflow_v1/notebooks/2_BasicModels/linear_regression.ipynb

## File Metadata

- **File Path**: `tensorflow_v1/notebooks/2_BasicModels/linear_regression.ipynb`
- **File Size**: 62033 bytes
- **File Type**: .ipynb
- **Purpose**: Jupyter notebook with interactive code examples

## Original Source

```
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "collapsed": false
   },
   "source": [
    "# Linear Regression Example\n",
    "\n",
    "A linear regression learning algorithm example using TensorFlow library.\n",
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
    "import tensorflow as tf\n",
    "import numpy\n",
    "import matplotlib.pyplot as plt\n",
    "rng = numpy.random"
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
    "# Parameters\n",
    "learning_rate = 0.01\n",
    "training_epochs = 1000\n",
    "display_step = 50"
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
    "train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,\n",
    "                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])\n",
    "train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,\n",
    "                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])\n",
    "n_samples = train_X.shape[0]"
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
    "# tf Graph Input\n",
    "X = tf.placeholder(\"float\")\n",
    "Y = tf.placeholder(\"float\")\n",
    "\n",
    "# Set model weights\n",
    "W = tf.Variable(rng.randn(), name=\"weight\")\n",
    "b = tf.Variable(rng.randn(), name=\"bias\")"
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
    "# Construct a linear model\n",
    "pred = tf.add(tf.multiply(X, W), b)"
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
    "# Mean squared error\n",
    "cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)\n",
    "# Gradient descent\n",
    "optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# Initialize the variables (i.e. assign their default value)\n",
    "init = tf.global_variables_initializer()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 0050 cost= 0.195095107 W= 0.441748 b= -0.580876\n",
      "Epoch: 0100 cost= 0.181448311 W= 0.430319 b= -0.498661\n",
      "Epoch: 0150 cost= 0.169377610 W= 0.419571 b= -0.421336\n",
      "Epoch: 0200 cost= 0.158700854 W= 0.409461 b= -0.348611\n",
      "Epoch: 0250 cost= 0.149257123 W= 0.399953 b= -0.28021\n",
      "Epoch: 0300 cost= 0.140904188 W= 0.391011 b= -0.215878\n",
      "Epoch: 0350 cost= 0.133515999 W= 0.3826 b= -0.155372\n",
      "Epoch: 0400 cost= 0.126981199 W= 0.374689 b= -0.0984639\n",
      "Epoch: 0450 cost= 0.121201262 W= 0.367249 b= -0.0449408\n",
      "Epoch: 0500 cost= 0.116088994 W= 0.360252 b= 0.00539905\n",
      "Epoch: 0550 cost= 0.111567356 W= 0.35367 b= 0.052745\n",
      "Epoch: 0600 cost= 0.107568085 W= 0.34748 b= 0.0972751\n",
      "Epoch: 0650 cost= 0.104030922 W= 0.341659 b= 0.139157\n",
      "Epoch: 0700 cost= 0.100902475 W= 0.336183 b= 0.178547\n",
      "Epoch: 0750 cost= 0.098135538 W= 0.331033 b= 0.215595\n",
      "Epoch: 0800 cost= 0.095688373 W= 0.32619 b= 0.25044\n",
      "Epoch: 0850 cost= 0.093524046 W= 0.321634 b= 0.283212\n",
      "Epoch: 0900 cost= 0.091609895 W= 0.317349 b= 0.314035\n",
      "Epoch: 0950 cost= 0.089917004 W= 0.31332 b= 0.343025\n",
      "Epoch: 1000 cost= 0.088419855 W= 0.30953 b= 0.370291\n",
      "Optimization Finished!\n",
      "Training cost= 0.0884199 W= 0.30953 b= 0.370291 \n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n45zP53Nbt251zjlXWlrqzMydffbZzfYtLS11Pp+vYXvVqlXOzNz06dPD9rvgggucz+dz1113XUPb\n/fff73w+n/vggw8a2vLy8pzP53PLly9vaPvkk09cdna2+9nPftbQ9vXXXzeL5YMPPnDZ2dmurKys\noW39+vXOzNyCBQt2e8//7//9P+fz+dwrr7zS0LZ582bXo0ePZjF++eWXzY6/7LLLXNeuXcPiGjNm\njMvPz2+2byAQcHV1dWFttbW1rnfv3u7iiy/ebZxt+dms3wcodO38PNZwg4hIDJYvXsyoCF3RAKMD\nAZYvWpRy564fqmitF6L+/S+++KKhzcy47LLLWr3GM888g5nxk5/8JKz9iiuuaOh9aM3BBx8c1tPR\ns2dPBg8ezNq1axvasrKyGv4cCAT49NNPycnJYfDgwaxatapN12lsyZIlHHPMMRQVFTW07bvvvpx9\n9tnN9u3cuXPDn7du3cqWLVsYNmwY27dvZ82aNa1ey8zYY49gR75zjs8++4yvv/6aI488MqbYE0lJ\ngohIlJxzdKmro6WphAbk1NW1+UMxWeeu//BvbV5DS8lEfn5+q9eoH4Nvuu/AgQPbHGekpx323ntv\nPvvss4Zt5xzz5s1j0KBBdO7cmZ49e9KrVy/eeOMNamtr23ytxnEfdNBBzdoHDx7crG316tWcfvrp\n9OjRg27durHffvtxzjnnALT52gsWLOCwww4jOzubfffdl169evHUU0/FFHsiaU6CiEiUzIxtWVk4\niPhh7oBtWVkxPZGQyHN369aN/fffv9XJfa+//jp9+vRpGEOvt9dee0V9zXrRxNupU6eI7Y0To/Ly\ncq699louuugiysrK2GefffD5fEyZMiXiZMNYY2yajNXW1jJixAh69OhBWVkZ/fv3Jzs7m+rqaq66\n6qo2Xfuhhx7iggsu4Ic//CFXXnklvXr1olOnTtxwww1hvSWpQEmCiEgMho4dy9L58xkd4UPhGZ+P\nYcXFKXnuMWPGcM8997BixYqIkxdfeukl1q9f32y4oK0OPPBAAoEA69atY8CAAQ3t77zzTswxR7Jw\n4UJOPPFEfve734W1f/755+y3335Rn+/AAw+MGOPbb78dtv3888/z2Wef8eSTTzJ06NCG9vfff7/Z\nsS0lRgsXLmTAgAE89thjYe3XXntt1HEnmoYbRERiMKO8nLkFBSzx+aj/rumAJT4f8woKmF5WlpLn\n/tnPfkZ2djY//vGP+fTTT8Pe+/TTT7nsssvo0qULM2bMiOn8o0aNwjnX8GREvd/85jdxXUeiU6dO\nzb7l/+lPf2Ljxo0xne/UU0/l73//O6+88kpD2yeffEJlZWXE6zbuMfj666+b3S9Aly5dIg4fdOrU\nqdn/i5UrV1JVVRVT7ImkngQRkRjk5uaysKqKW2bNYu6iReTU1bE9K4uhxcUsLCuL+RHFRJ974MCB\nLFiwgIkTJ/I///M/XHTRReTn57Nu3Truu+8+tmzZwh/+8Ic2zT+IpLCwkPHjx3PrrbeyefNmjjnm\nGF544QXeffddILphh90ZM2YM119/PRdeeCHHHnssb7zxBg8//HBY70U0rrzySh588EFGjRrFlClT\nyMnJ4Xe/+x0HHnhg2PDMsccey9577825557L5MmTgeDwQaT7Kioq4tFHH2X69Ol873vfo2vXrowZ\nM4YxY8bw+OOPc9ppp/GDH/yAtWvXctddd3HIIYewdevW2P6HJEp7H49IxAs9AikiHmrLY2ZNBQKB\nhMWTiHO/+eab7uyzz3Z9+vRxnTt3dt/+9rfdxIkT3VtvvdVs3/rHHLds2RLxvU6dOoW17dixw11x\nxRWuZ8+erlu3bm78+PHu3XffdWbm5syZ07BfpEcg8/PzXXFxcbPrHH/88e7EE09s2P7qq6/cz372\nM9enTx/XpUsXN2LECLdy5Up3wgknhO23fv165/P5Wn0Esv7/yQknnOBycnJc37593Q033ODuu+++\nZjFWVVW5Y4891nXp0sUdcMAB7uqrr3bPPvus8/l87oUXXmjYb9u2bW7ixIlun332cT6fL+xxyF/9\n6lcuPz/f7bXXXq6oqMg9/fTT7vzzz3f9+/ffbYzJfgTSXAwzZBPNzAqB6urqagoLC70OR0Q6mFWr\nVlFUVIR+B8XPq6++SmFhIQ8//DAlJSVeh5O22vKzWb8PUOSca9czlZqTICIicfXVV181a7v11lvp\n1KkTI0aM8CAiiZXmJIiISFzNmTOH6upqjj/+ePbYYw+efvppli5dyo9//GP69OnjdXgSBSUJIiIS\nV0OGDOHZZ5+lrKyMrVu30q9fP6677jquueYar0OTKClJEBGRuBo5ciQjR470OgyJA81JEBERkYiU\nJIiIiEhEShJEREQkIiUJIiIiEpGSBBEREYlISYKIiIhEpCRBREREIlKSICIizRxwwAFceumlnsbw\n/vvv4/P5eOSRR3a731//+ld8Ph8rVqxoaJs4cSIHHXRQokPMeEoSREQ6kAULFuDz+SK+Gq+I6PP5\nwsofv/XWW1x33XV89NFHzc45f/58HnzwwaTE35KmpZrNDJ9PH3HtpRUXRUQ6GDPj+uuvJy8vL6z9\n0EMPbfjz+++/T6dOnRq233zzTa677jpOPvlkDjjggLDjbr/9dvr27cs555yT0Lijcf/995OKVY7T\njZIEEZEOaPTo0bstg52VlRW27Zxr9m09lTVOcCR26osREZFmGs9JuPfeeznrrLMAGDZsGD6fj06d\nOrFixQr69u3L22+/zXPPPdcwbHHKKac0nOfzzz9n8uTJ9OvXj+zsbAYNGsTNN9/c7HqfffYZ5557\nLj169GCfffbhoosu4osvvog5/qZzEurnN9x2223cddddDBgwgL322otjjjmGf/3rX82Or6mpYfz4\n8ey7777k5ORw1FFH8fTTT8ccT7qKqifBzC4DfgLkhZreAn7pnHumhf3PA34POKA+Bf3SOZcTU7Qi\nIhIXtbW1bNmyJaxt3333bfhz416DE044gZ/+9KfccccdzJ49u+HDd/Dgwdx+++1cfvnl7Lvvvlx9\n9dU459h///0B2L59O8OHD+fjjz/msssu44ADDuDll1/myiuv5OOPP2bOnDlAsJdi7NixrFy5kssv\nv5zBgwezcOFCLrjggph7L8ws4rELFixg+/btXH755Tjn
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `tensorflow_v1/notebooks/2_BasicModels/linear_regression.ipynb` within the TensorFlow Examples repository.

### Purpose

This file is a Jupyter notebook containing interactive code cells, explanatory text, and visualizations for learning TensorFlow.

### Context

Located in the `tensorflow_v1/` directory, specifically within `tensorflow_v1/notebooks/2_BasicModels/`, this file is part of the TensorFlow 1.x examples collection.



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
jupyter notebook tensorflow_v1/notebooks/2_BasicModels/linear_regression.ipynb
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
- [linear_regression_eager_api.ipynb](linear_regression_eager_api.ipynb_docs.md)
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
