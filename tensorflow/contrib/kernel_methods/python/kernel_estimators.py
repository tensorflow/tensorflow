# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Estimators that combine explicit kernel mappings with linear models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.contrib import layers
from tensorflow.contrib.kernel_methods.python.mappers import dense_kernel_mapper as dkm
from tensorflow.contrib.learn.python.learn.estimators import estimator
from tensorflow.contrib.learn.python.learn.estimators import head as head_lib
from tensorflow.contrib.learn.python.learn.estimators import linear
from tensorflow.contrib.learn.python.learn.estimators import prediction_key
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import tf_logging as logging

_FEATURE_COLUMNS = "feature_columns"
_KERNEL_MAPPERS = "kernel_mappers"
_OPTIMIZER = "optimizer"


def _check_valid_kernel_mappers(kernel_mappers):
  """Checks that the input kernel_mappers are valid."""
  if kernel_mappers is None:
    return True
  for kernel_mappers_list in six.itervalues(kernel_mappers):
    for kernel_mapper in kernel_mappers_list:
      if not isinstance(kernel_mapper, dkm.DenseKernelMapper):
        return False
  return True


def _check_valid_head(head):
  """Returns true if the provided head is supported."""
  if head is None:
    return False
  # pylint: disable=protected-access
  return isinstance(head, head_lib._BinaryLogisticHead) or isinstance(
      head, head_lib._MultiClassHead)
  # pylint: enable=protected-access


def _update_features_and_columns(features, feature_columns,
                                 kernel_mappers_dict):
  """Updates features and feature_columns based on provided kernel mappers.

  Currently supports the update of RealValuedColumns only.

  Args:
    features: Initial features dict. The key is a `string` (feature column name)
      and the value is a tensor.
    feature_columns: Initial iterable containing all the feature columns to be
      consumed (possibly after being updated) by the model. All items should be
      instances of classes derived from `FeatureColumn`.
    kernel_mappers_dict: A dict from feature column (type: _FeatureColumn) to
      objects inheriting from KernelMapper class.

  Returns:
    updated features and feature_columns based on provided kernel_mappers_dict.
  """
  if kernel_mappers_dict is None:
    return features, feature_columns

  # First construct new columns and features affected by kernel_mappers_dict.
  mapped_features = dict()
  mapped_columns = set()
  for feature_column in kernel_mappers_dict:
    column_name = feature_column.name
    # Currently only mappings over RealValuedColumns are supported.
    if not isinstance(feature_column, layers.feature_column._RealValuedColumn):  # pylint: disable=protected-access
      logging.warning(
          "Updates are currently supported on RealValuedColumns only. Metadata "
          "for FeatureColumn {} will not be updated.".format(column_name))
      continue
    mapped_column_name = column_name + "_MAPPED"
    # Construct new feature columns based on provided kernel_mappers.
    column_kernel_mappers = kernel_mappers_dict[feature_column]
    new_dim = sum([mapper.output_dim for mapper in column_kernel_mappers])
    mapped_columns.add(
        layers.feature_column.real_valued_column(mapped_column_name, new_dim))

    # Get mapped features by concatenating mapped tensors (one mapped tensor
    # per kernel mappers from the list of kernel mappers corresponding to each
    # feature column).
    output_tensors = []
    for kernel_mapper in column_kernel_mappers:
      output_tensors.append(kernel_mapper.map(features[column_name]))
    tensor = array_ops.concat(output_tensors, 1)
    mapped_features[mapped_column_name] = tensor

  # Finally update features dict and feature_columns.
  features = features.copy()
  features.update(mapped_features)
  feature_columns = set(feature_columns)
  feature_columns.update(mapped_columns)

  return features, feature_columns


def _kernel_model_fn(features, labels, mode, params, config=None):
  """model_fn for the Estimator using kernel methods.

  Args:
    features: `Tensor` or dict of `Tensor` (depends on data passed to `fit`).
    labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of
      dtype `int32` or `int64` in the range `[0, n_classes)`.
    mode: Defines whether this is training, evaluation or prediction. See
      `ModeKeys`.
    params: A dict of hyperparameters.
      The following hyperparameters are expected:
      * head: A `Head` instance.
      * feature_columns: An iterable containing all the feature columns used by
          the model.
      * optimizer: string, `Optimizer` object, or callable that defines the
          optimizer to use for training. If `None`, will use a FTRL optimizer.
      * kernel_mappers: Dictionary of kernel mappers to be applied to the input
          features before training.
    config: `RunConfig` object to configure the runtime settings.

  Returns:
    A `ModelFnOps` instance.

  Raises:
    ValueError: If mode is not any of the `ModeKeys`.
  """
  feature_columns = params[_FEATURE_COLUMNS]
  kernel_mappers = params[_KERNEL_MAPPERS]

  updated_features, updated_columns = _update_features_and_columns(
      features, feature_columns, kernel_mappers)
  params[_FEATURE_COLUMNS] = updated_columns

  return linear._linear_model_fn(  # pylint: disable=protected-access
      updated_features, labels, mode, params, config)


class _KernelEstimator(estimator.Estimator):
  """Generic kernel-based linear estimator."""

  def __init__(self,
               feature_columns=None,
               model_dir=None,
               weight_column_name=None,
               head=None,
               optimizer=None,
               kernel_mappers=None,
               config=None):
    """Constructs a `_KernelEstimator` object."""
    if not feature_columns and not kernel_mappers:
      raise ValueError(
          "You should set at least one of feature_columns, kernel_mappers.")
    if not _check_valid_kernel_mappers(kernel_mappers):
      raise ValueError("Invalid kernel mappers.")

    if not _check_valid_head(head):
      raise ValueError(
          "head type: {} is not supported. Supported head types: "
          "_BinaryLogisticHead, _MultiClassHead.".format(type(head)))

    params = {
        "head": head,
        _FEATURE_COLUMNS: feature_columns or [],
        _OPTIMIZER: optimizer,
        _KERNEL_MAPPERS: kernel_mappers
    }
    super(_KernelEstimator, self).__init__(
        model_fn=_kernel_model_fn,
        model_dir=model_dir,
        config=config,
        params=params)


class KernelLinearClassifier(_KernelEstimator):
  """Linear classifier using kernel methods as feature preprocessing.

  It trains a linear model after possibly mapping initial input features into
  a mapped space using explicit kernel mappings. Due to the kernel mappings,
  training a linear classifier in the mapped (output) space can detect
  non-linearities in the input space.

  The user can provide a list of kernel mappers to be applied to all or a subset
  of existing feature_columns. This way, the user can effectively provide 2
  types of feature columns:
  - those passed as elements of feature_columns in the classifier's constructor
  - those appearing as a key of the kernel_mappers dict.
  If a column appears in feature_columns only, no mapping is applied to it. If
  it appears as a key in kernel_mappers, the corresponding kernel mappers are
  applied to it. Note that it is possible that a column appears in both places.
  Currently kernel_mappers are supported for _RealValuedColumns only.

  Example usage:
  ```
  real_column_a = real_valued_column(name='real_column_a',...)
  sparse_column_b = sparse_column_with_hash_bucket(...)
  kernel_mappers = {real_column_a : [RandomFourierFeatureMapper(...)]}
  optimizer = ...

  # real_column_a is used as a feature in both its initial and its transformed
  # (mapped) form. sparse_column_b is not affected by kernel mappers.
  kernel_classifier = KernelLinearClassifier(
      feature_columns=[real_column_a, sparse_column_b],
      model_dir=...,
      optimizer=optimizer,
      kernel_mappers=kernel_mappers)

  # real_column_a is used as a feature in its transformed (mapped) form only.
  # sparse_column_b is not affected by kernel mappers.
  kernel_classifier = KernelLinearClassifier(
      feature_columns=[sparse_column_b],
      model_dir=...,
      optimizer=optimizer,
      kernel_mappers=kernel_mappers)

  # Input builders
  def train_input_fn: # returns x, y
    ...
  def eval_input_fn: # returns x, y
    ...

  kernel_classifier.fit(input_fn=train_input_fn)
  kernel_classifier.evaluate(input_fn=eval_input_fn)
  kernel_classifier.predict(...)
  ```

  Input of `fit` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:
  * if `weight_column_name` is not `None`, a feature with
    `key=weight_column_name` whose value is a `Tensor`.
  * for each `column` in `feature_columns`:
    - if `column` is a `SparseColumn`, a feature with `key=column.name`
      whose `value` is a `SparseTensor`.
    - if `column` is a `WeightedSparseColumn`, two features: the first with
      `key` the id column name, the second with `key` the weight column name.
      Both features' `value` must be a `SparseTensor`.
    - if `column` is a `RealValuedColumn`, a feature with `key=column.name`
      whose `value` is a `Tensor`.
  """

  def __init__(self,
               feature_columns=None,
               model_dir=None,
               n_classes=2,
               weight_column_name=None,
               optimizer=None,
               kernel_mappers=None,
               config=None):
    """Construct a `KernelLinearClassifier` estimator object.

    Args:
      feature_columns: An iterable containing all the feature columns used by
        the model. All items in the set should be instances of classes derived
        from `FeatureColumn`.
      model_dir: Directory to save model parameters, graph etc. This can also be
        used to load checkpoints from the directory into an estimator to
        continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
        Note that class labels are integers representing the class index (i.e.
        values from 0 to n_classes-1). For arbitrary label values (e.g. string
        labels), convert to class indices first.
      weight_column_name: A string defining feature column name representing
        weights. It is used to down weight or boost examples during training. It
        will be multiplied by the loss of the example.
      optimizer: The optimizer used to train the model. If specified, it should
        be an instance of `tf.Optimizer`. If `None`, the Ftrl optimizer is used
        by default.
      kernel_mappers: Dictionary of kernel mappers to be applied to the input
        features before training a (linear) model. Keys are feature columns and
        values are lists of mappers to be applied to the corresponding feature
        column. Currently only _RealValuedColumns are supported and therefore
        all mappers should conform to the `DenseKernelMapper` interface (see
        ./mappers/dense_kernel_mapper.py).
      config: `RunConfig` object to configure the runtime settings.

    Returns:
      A `KernelLinearClassifier` estimator.

    Raises:
      ValueError: if n_classes < 2.
      ValueError: if neither feature_columns nor kernel_mappers are provided.
      ValueError: if mappers provided as kernel_mappers values are invalid.
    """
    super(KernelLinearClassifier, self).__init__(
        feature_columns=feature_columns,
        model_dir=model_dir,
        weight_column_name=weight_column_name,
        head=head_lib.multi_class_head(
            n_classes=n_classes, weight_column_name=weight_column_name),
        kernel_mappers=kernel_mappers,
        config=config)

  def predict_classes(self, input_fn=None):
    """Runs inference to determine the predicted class per instance.

    Args:
      input_fn: The input function providing features.

    Returns:
      A generator of predicted classes for the features provided by input_fn.
      Each predicted class is represented by its class index (i.e. integer from
      0 to n_classes-1)
    """
    key = prediction_key.PredictionKey.CLASSES
    predictions = super(KernelLinearClassifier, self).predict(
        input_fn=input_fn, outputs=[key])
    return (pred[key] for pred in predictions)

  def predict_proba(self, input_fn=None):
    """Runs inference to determine the class probability predictions.

    Args:
      input_fn: The input function providing features.

    Returns:
      A generator of predicted class probabilities for the features provided by
        input_fn.
    """
    key = prediction_key.PredictionKey.PROBABILITIES
    predictions = super(KernelLinearClassifier, self).predict(
        input_fn=input_fn, outputs=[key])
    return (pred[key] for pred in predictions)
