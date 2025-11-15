# Documentation: notebooks/2_BasicModels/gradient_boosted_decision_tree.ipynb

## File Metadata

- **File Path**: `notebooks/2_BasicModels/gradient_boosted_decision_tree.ipynb`
- **File Size**: 32353 bytes
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
    "# Gradient Boosted Decision Tree\n",
    "\n",
    "Implement a Gradient Boosted Decision tree (GBDT) with TensorFlow to classify\n",
    "handwritten digit images. This example is using the MNIST database of\n",
    "handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).\n",
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
    "from __future__ import print_function\n",
    "\n",
    "import tensorflow as tf\n",
    "from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier\n",
    "from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner\n",
    "\n",
    "# Ignore all GPUs (current TF GBDT does not support GPU).\n",
    "import os\n",
    "os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Extracting /tmp/data/train-images-idx3-ubyte.gz\n",
      "Extracting /tmp/data/train-labels-idx1-ubyte.gz\n",
      "Extracting /tmp/data/t10k-images-idx3-ubyte.gz\n",
      "Extracting /tmp/data/t10k-labels-idx1-ubyte.gz\n"
     ]
    }
   ],
   "source": [
    "# Import MNIST data\n",
    "# Set verbosity to display errors only (Remove this line for showing warnings)\n",
    "tf.logging.set_verbosity(tf.logging.ERROR)\n",
    "from tensorflow.examples.tutorials.mnist import input_data\n",
    "mnist = input_data.read_data_sets(\"/tmp/data/\", one_hot=False,\n",
    "                                  source_url='http://yann.lecun.com/exdb/mnist/')"
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
    "# Parameters\n",
    "batch_size = 4096 # The number of samples per batch\n",
    "num_classes = 10 # The 10 digits\n",
    "num_features = 784 # Each image is 28x28 pixels\n",
    "max_steps = 10000\n",
    "\n",
    "# GBDT Parameters\n",
    "learning_rate = 0.1\n",
    "l1_regul = 0.\n",
    "l2_regul = 1.\n",
    "examples_per_layer = 1000\n",
    "num_trees = 10\n",
    "max_depth = 16"
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
    "# Fill GBDT parameters into the config proto\n",
    "learner_config = gbdt_learner.LearnerConfig()\n",
    "learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate\n",
    "learner_config.regularization.l1 = l1_regul\n",
    "learner_config.regularization.l2 = l2_regul / examples_per_layer\n",
    "learner_config.constraints.max_tree_depth = max_depth\n",
    "growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER\n",
    "learner_config.growing_mode = growing_mode\n",
    "run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)\n",
    "learner_config.multi_class_strategy = (\n",
    "    gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN)\\\n",
    "\n",
    "# Create a TensorFlor GBDT Estimator\n",
    "gbdt_model = GradientBoostedDecisionTreeClassifier(\n",
    "    model_dir=None, # No save directory specified\n",
    "    learner_config=learner_config,\n",
    "    n_classes=num_classes,\n",
    "    examples_per_layer=examples_per_layer,\n",
    "    num_trees=num_trees,\n",
    "    center_bias=False,\n",
    "    config=run_config)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {
    "collapsed": false,
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "INFO:tensorflow:Active Feature Columns: ['images_0', 'images_1', 'images_2', 'images_3', 'images_4', 'images_5', 'images_6', 'images_7', 'images_8', 'images_9', 'images_10', 'images_11', 'images_12', 'images_13', 'images_14', 'images_15', 'images_16', 'images_17', 'images_18', 'images_19', 'images_20', 'images_21', 'images_22', 'images_23', 'images_24', 'images_25', 'images_26', 'images_27', 'images_28', 'images_29', 'images_30', 'images_31', 'images_32', 'images_33', 'images_34', 'images_35', 'images_36', 'images_37', 'images_38', 'images_39', 'images_40', 'images_41', 'images_42', 'images_43', 'images_44', 'images_45', 'images_46', 'images_47', 'images_48', 'images_49', 'images_50', 'images_51', 'images_52', 'images_53', 'images_54', 'images_55', 'images_56', 'images_57', 'images_58', 'images_59', 'images_60', 'images_61', 'images_62', 'images_63', 'images_64', 'images_65', 'images_66', 'images_67', 'images_68', 'images_69', 'images_70', 'images_71', 'images_72', 'images_73', 'images_74', 'images_75', 'images_76', 'images_77', 'images_78', 'images_79', 'images_80', 'images_81', 'images_82', 'images_83', 'images_84', 'images_85', 'images_86', 'images_87', 'images_88', 'images_89', 'images_90', 'images_91', 'images_92', 'images_93', 'images_94', 'images_95', 'images_96', 'images_97', 'images_98', 'images_99', 'images_100', 'images_101', 'images_102', 'images_103', 'images_104', 'images_105', 'images_106', 'images_107', 'images_108', 'images_109', 'images_110', 'images_111', 'images_112', 'images_113', 'images_114', 'images_115', 'images_116', 'images_117', 'images_118', 'images_119', 'images_120', 'images_121', 'images_122', 'images_123', 'images_124', 'images_125', 'images_126', 'images_127', 'images_128', 'images_129', 'images_130', 'images_131', 'images_132', 'images_133', 'images_134', 'images_135', 'images_136', 'images_137', 'images_138', 'images_139', 'images_140', 'images_141', 'images_142', 'images_143', 'images_144', 'images_145', 'images_146', 'images_147', 'images_148', 'images_149', 'images_150', 'images_151', 'images_152', 'images_153', 'images_154', 'images_155', 'images_156', 'images_157', 'images_158', 'images_159', 'images_160', 'images_161', 'images_162', 'images_163', 'images_164', 'images_165', 'images_166', 'images_167', 'images_168', 'images_169', 'images_170', 'images_171', 'images_172', 'images_173', 'images_174', 'images_175', 'images_176', 'images_177', 'images_178', 'images_179', 'images_180', 'images_181', 'images_182', 'images_183', 'images_184', 'images_185', 'images_186', 'images_187', 'images_188', 'images_189', 'images_190', 'images_191', 'images_192', 'images_193', 'images_194', 'images_195', 'images_196', 'images_197', 'images_198', 'images_199', 'images_200', 'images_201', 'images_202', 'images_203', 'images_204', 'images_205', 'images_206', 'images_207', 'images_208', 'images_209', 'images_210', 'images_211', 'images_212', 'images_213', 'images_214', 'images_215', 'images_216', 'images_217', 'images_218', 'images_219', 'images_220', 'images_221', 'images_222', 'images_223', 'images_224', 'images_225', 'images_226', 'images_227', 'images_228', 'images_229', 'images_230', 'images_231', 'images_232', 'images_233', 'images_234', 'images_235', 'images_236', 'images_237', 'images_238', 'images_239', 'images_240', 'images_241', 'images_242', 'images_243', 'images_244', 'images_245', 'images_246', 'images_247', 'images_248', 'images_249', 'images_250', 'images_251', 'images_252', 'images_253', 'images_254', 'images_255', 'images_256', 'images_257', 'images_258', 'images_259', 'images_260', 'images_261', 'images_262', 'images_263', 'images_264', 'images_265', 'images_266', 'images_267', 'images_268', 'images_269', 'images_270', 'images_271', 'images_272', 'images_273', 'images_274', 'images_275', 'images_276', 'images_277', 'images_278', 'images_279', 'images_280', 'images_281', 'images_282', 'images_283', 'images_284', 'images_285', 'images_286', 'images_287', 'images_288', 'images_289', 'images_290', 'images_291', 'images_292', 'images_293', 'images_294', 'images_295', 'images_296', 'images_297', 'images_298', 'images_299', 'images_300', 'images_301', 'images_302', 'images_303', 'images_304', 'images_305', 'images_306', 'images_307', 'images_308', 'images_309', 'images_310', 'images_311', 'images_312', 'images_313', 'images_314', 'images_315', 'images_316', 'images_317', 'images_318', 'images_319', 'images_320', 'images_321', 'images_322', 'images_323', 'images_324', 'images_325', 'images_326', 'images_327', 'images_328', 'images_329', 'images_330', 'images_331', 'images_332', 'images_333', 'images_334', 'images_335', 'images_336', 'images_337', 'images_338', 'images_339', 'images_340', 'images_341', 'images_342', 'images_343', 'images_344', 'images_345', 'images_346', 'images_347', 'images_348', 'images_349', 'images_350', 'images_351', 'images_352', 'images_353', 'images_354', 'images_355', 'images_356', 'images_357', 'images_358', 'images_359', 'images_360', 'images_361', 'images_362', 'images_363', 'images_364', 'images_365', 'images_366', 'images_367', 'images_368', 'images_369', 'images_370', 'images_371', 'images_372', 'images_373', 'images_374', 'images_375', 'images_376', 'images_377', 'images_378', 'images_379', 'images_380', 'images_381', 'images_382', 'images_383', 'images_384', 'images_385', 'images_386', 'images_387', 'images_388', 'images_389', 'images_390', 'images_391', 'images_392', 'images_393', 'images_394', 'images_395', 'images_396', 'images_397', 'images_398', 'images_399', 'images_400', 'images_401', 'images_402', 'images_403', 'images_404', 'images_405', 'images_406', 'images_407', 'images_408', 'images_409', 'images_410', 'images_411', 'images_412', 'images_413', 'images_414', 'images_415', 'images_416', 'images_417', 'images_418', 'images_419', 'images_420', 'images_421', 'images_422', 'images_423', 'images_424', 'images_425', 'images_426', 'images_427', 'images_428', 'images_429', 'images_430', 'images_431', 'images_432', 'images_433', 'images_434', 'images_435', 'images_436', 'images_
... [Content truncated for brevity] ...
```

## High-Level Overview

This file is located at `notebooks/2_BasicModels/gradient_boosted_decision_tree.ipynb` within the TensorFlow Examples repository.

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
jupyter notebook notebooks/2_BasicModels/gradient_boosted_decision_tree.ipynb
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

- [kmeans.ipynb](kmeans.ipynb_docs.md)
- [linear_regression.ipynb](linear_regression.ipynb_docs.md)
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

training, testing, gradient, accuracy, loss, mnist, tensorflow

---

*This documentation was automatically generated for comprehensive repository understanding.*
