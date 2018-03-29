# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Boosted Trees estimators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import boosted_trees as canned_boosted_trees


class _BoostedTreesEstimator(estimator.Estimator):
  """An Estimator for Tensorflow Boosted Trees models."""

  def __init__(self,
               feature_columns,
               n_batches_per_layer,
               head,
               model_dir=None,
               weight_column=None,
               n_trees=100,
               max_depth=6,
               learning_rate=0.1,
               l1_regularization=0.,
               l2_regularization=0.,
               tree_complexity=0.,
               config=None):
    """Initializes a `BoostedTreesEstimator` instance.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      n_batches_per_layer: the number of batches to collect statistics per
        layer.
      head: the `Head` instance defined for Estimator.
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator
        to continue training a previously saved model.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
        weights. It is used to downweight or boost examples during training. It
        will be multiplied by the loss of the example. If it is a string, it is
        used as a key to fetch weight tensor from the `features`. If it is a
        `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
        then weight_column.normalizer_fn is applied on it to get weight tensor.
      n_trees: number trees to be created.
      max_depth: maximum depth of the tree to grow.
      learning_rate: shrinkage parameter to be used when a tree added to the
        model.
      l1_regularization: regularization multiplier applied to the absolute
        weights of the tree leafs.
      l2_regularization: regularization multiplier applied to the square weights
        of the tree leafs.
      tree_complexity: regularization factor to penalize trees with more leaves.
      config: `RunConfig` object to configure the runtime settings.
    """
    # TODO(youngheek): param validations.

    # HParams for the model.
    tree_hparams = canned_boosted_trees.TreeHParams(
        n_trees, max_depth, learning_rate, l1_regularization, l2_regularization,
        tree_complexity)

    def _model_fn(features, labels, mode, config):
      return canned_boosted_trees._bt_model_fn(  # pylint: disable=protected-access
          features, labels, mode, head, feature_columns, tree_hparams,
          n_batches_per_layer, config)

    super(_BoostedTreesEstimator, self).__init__(
        model_fn=_model_fn, model_dir=model_dir, config=config)


def boosted_trees_classifier_train_in_memory(
    train_input_fn,
    feature_columns,
    model_dir=None,
    n_classes=canned_boosted_trees._HOLD_FOR_MULTI_CLASS_SUPPORT,
    weight_column=None,
    label_vocabulary=None,
    n_trees=100,
    max_depth=6,
    learning_rate=0.1,
    l1_regularization=0.,
    l2_regularization=0.,
    tree_complexity=0.,
    config=None,
    train_hooks=None):
  """Trains a boosted tree classifier with in memory dataset.

  Example:

  ```python
  bucketized_feature_1 = bucketized_column(
    numeric_column('feature_1'), BUCKET_BOUNDARIES_1)
  bucketized_feature_2 = bucketized_column(
    numeric_column('feature_2'), BUCKET_BOUNDARIES_2)

  def input_fn_train():
    dataset = create-dataset-from-training-data
    # Don't use repeat or cache, since it is assumed to be one epoch
    # This is either tf.data.Dataset, or a tuple of feature dict and label.
    return dataset

  classifier = boosted_trees_classifier_train_in_memory(
      train_input_fn,
      feature_columns=[bucketized_feature_1, bucketized_feature_2],
      n_trees=100,
      ... <some other params>
  )

  def input_fn_eval():
    ...
    return dataset

  metrics = classifier.evaluate(input_fn=input_fn_eval, steps=10)
  ```

  Args:
    train_input_fn: the input function returns a dataset containing a single
      epoch of *unbatched* features and labels.
    feature_columns: An iterable containing all the feature columns used by
      the model. All items in the set should be instances of classes derived
      from `FeatureColumn`.
    model_dir: Directory to save model parameters, graph and etc. This can
      also be used to load checkpoints from the directory into a estimator
      to continue training a previously saved model.
    n_classes: number of label classes. Default is binary classification.
      Multiclass support is not yet implemented.
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to downweight or boost examples during training. It
      will be multiplied by the loss of the example. If it is a string, it is
      used as a key to fetch weight tensor from the `features`. If it is a
      `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
      then weight_column.normalizer_fn is applied on it to get weight tensor.
    label_vocabulary: A list of strings represents possible label values. If
      given, labels must be string type and have any value in
      `label_vocabulary`. If it is not given, that means labels are
      already encoded as integer or float within [0, 1] for `n_classes=2` and
      encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
      Also there will be errors if vocabulary is not provided and labels are
      string.
    n_trees: number trees to be created.
    max_depth: maximum depth of the tree to grow.
    learning_rate: shrinkage parameter to be used when a tree added to the
      model.
    l1_regularization: regularization multiplier applied to the absolute
      weights of the tree leafs.
    l2_regularization: regularization multiplier applied to the square weights
      of the tree leafs.
    tree_complexity: regularization factor to penalize trees with more leaves.
    config: `RunConfig` object to configure the runtime settings.
    train_hooks: a list of Hook instances to be passed to estimator.train().

  Returns:
    a `BoostedTreesClassifier` instance created with the given arguments and
      trained with the data loaded up on memory from the input_fn.

  Raises:
    ValueError: when wrong arguments are given or unsupported functionalities
       are requested.
  """
  # pylint: disable=protected-access
  # TODO(nponomareva): Support multi-class cases.
  if n_classes == canned_boosted_trees._HOLD_FOR_MULTI_CLASS_SUPPORT:
    n_classes = 2
  head, closed_form = (
      canned_boosted_trees._create_classification_head_and_closed_form(
          n_classes, weight_column, label_vocabulary=label_vocabulary))

  # HParams for the model.
  tree_hparams = canned_boosted_trees.TreeHParams(
      n_trees, max_depth, learning_rate, l1_regularization, l2_regularization,
      tree_complexity)

  def _model_fn(features, labels, mode, config):
    return canned_boosted_trees._bt_model_fn(
        features,
        labels,
        mode,
        head,
        feature_columns,
        tree_hparams,
        n_batches_per_layer=1,
        config=config,
        closed_form_grad_and_hess_fn=closed_form,
        train_in_memory=True)

  in_memory_classifier = estimator.Estimator(
      model_fn=_model_fn, model_dir=model_dir, config=config)

  in_memory_classifier.train(input_fn=train_input_fn, hooks=train_hooks)

  return in_memory_classifier
  # pylint: enable=protected-access


def boosted_trees_regressor_train_in_memory(
    train_input_fn,
    feature_columns,
    model_dir=None,
    label_dimension=canned_boosted_trees._HOLD_FOR_MULTI_DIM_SUPPORT,
    weight_column=None,
    n_trees=100,
    max_depth=6,
    learning_rate=0.1,
    l1_regularization=0.,
    l2_regularization=0.,
    tree_complexity=0.,
    config=None,
    train_hooks=None):
  """Trains a boosted tree regressor with in memory dataset.

  Example:

  ```python
  bucketized_feature_1 = bucketized_column(
    numeric_column('feature_1'), BUCKET_BOUNDARIES_1)
  bucketized_feature_2 = bucketized_column(
    numeric_column('feature_2'), BUCKET_BOUNDARIES_2)

  def input_fn_train():
    dataset = create-dataset-from-training-data
    # Don't use repeat or cache, since it is assumed to be one epoch
    # This is either tf.data.Dataset, or a tuple of feature dict and label.
    return dataset

  regressor = boosted_trees_regressor_train_in_memory(
      train_input_fn,
      feature_columns=[bucketized_feature_1, bucketized_feature_2],
      n_trees=100,
      ... <some other params>
  )

  def input_fn_eval():
    ...
    return dataset

  metrics = regressor.evaluate(input_fn=input_fn_eval, steps=10)
  ```

  Args:
    train_input_fn: the input function returns a dataset containing a single
      epoch of *unbatched* features and labels.
    feature_columns: An iterable containing all the feature columns used by
      the model. All items in the set should be instances of classes derived
      from `FeatureColumn`.
    model_dir: Directory to save model parameters, graph and etc. This can
      also be used to load checkpoints from the directory into a estimator
      to continue training a previously saved model.
    label_dimension: Number of regression targets per example.
      Multi-dimensional support is not yet implemented.
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
      weights. It is used to downweight or boost examples during training. It
      will be multiplied by the loss of the example. If it is a string, it is
      used as a key to fetch weight tensor from the `features`. If it is a
      `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
      then weight_column.normalizer_fn is applied on it to get weight tensor.
    n_trees: number trees to be created.
    max_depth: maximum depth of the tree to grow.
    learning_rate: shrinkage parameter to be used when a tree added to the
      model.
    l1_regularization: regularization multiplier applied to the absolute
      weights of the tree leafs.
    l2_regularization: regularization multiplier applied to the square weights
      of the tree leafs.
    tree_complexity: regularization factor to penalize trees with more leaves.
    config: `RunConfig` object to configure the runtime settings.
    train_hooks: a list of Hook instances to be passed to estimator.train().

  Returns:
    a `BoostedTreesClassifier` instance created with the given arguments and
      trained with the data loaded up on memory from the input_fn.

  Raises:
    ValueError: when wrong arguments are given or unsupported functionalities
       are requested.
  """
  # pylint: disable=protected-access
  # TODO(nponomareva): Extend it to multi-dimension cases.
  if label_dimension == canned_boosted_trees._HOLD_FOR_MULTI_DIM_SUPPORT:
    label_dimension = 1
  head = canned_boosted_trees._create_regression_head(label_dimension,
                                                      weight_column)

  # HParams for the model.
  tree_hparams = canned_boosted_trees.TreeHParams(
      n_trees, max_depth, learning_rate, l1_regularization, l2_regularization,
      tree_complexity)

  def _model_fn(features, labels, mode, config):
    return canned_boosted_trees._bt_model_fn(
        features,
        labels,
        mode,
        head,
        feature_columns,
        tree_hparams,
        n_batches_per_layer=1,
        config=config,
        train_in_memory=True)

  in_memory_regressor = estimator.Estimator(
      model_fn=_model_fn, model_dir=model_dir, config=config)

  in_memory_regressor.train(input_fn=train_input_fn, hooks=train_hooks)

  return in_memory_regressor
  # pylint: enable=protected-access
