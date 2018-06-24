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
"""Baseline estimators.

Baseline estimators are bias-only estimators that can be used for debugging
and as simple baselines.

Example:

```
# Build BaselineClassifier
classifier = BaselineClassifier(n_classes=3)

# Input builders
def input_fn_train: # returns x, y (where y represents label's class index).
  pass

def input_fn_eval: # returns x, y (where y represents label's class index).
  pass

# Fit model.
classifier.train(input_fn=input_fn_train)

# Evaluate cross entropy between the test and train labels.
loss = classifier.evaluate(input_fn=input_fn_eval)["loss"]

# predict outputs the probability distribution of the classes as seen in
# training.
predictions = classifier.predict(new_samples)
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.estimator import estimator
from tensorflow.python.estimator.canned import head as head_lib
from tensorflow.python.estimator.canned import optimizers
from tensorflow.python.feature_column import feature_column as feature_column_lib
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.losses import losses
from tensorflow.python.training import training_util
from tensorflow.python.util.tf_export import estimator_export

# The default learning rate of 0.3 is a historical artifact of the initial
# implementation, but seems a reasonable choice.
_LEARNING_RATE = 0.3


def _get_weight_column_key(weight_column):
  if weight_column is None:
    return None
  if isinstance(weight_column, six.string_types):
    return weight_column
  if not isinstance(weight_column, feature_column_lib._NumericColumn):  # pylint: disable=protected-access
    raise TypeError('Weight column must be either a string or _NumericColumn.'
                    ' Given type: {}.'.format(type(weight_column)))
  return weight_column.key()


def _baseline_logit_fn_builder(num_outputs, weight_column=None):
  """Function builder for a baseline logit_fn.

  Args:
    num_outputs: Number of outputs for the model.
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
       weights. It will be multiplied by the loss of the example.
  Returns:
    A logit_fn (see below).
  """

  def baseline_logit_fn(features):
    """Baseline model logit_fn.

    The baseline model simply learns a bias, so the output logits are a
    `Variable` with one weight for each output that learns the bias for the
    corresponding output.

    Args:
      features: The first item returned from the `input_fn` passed to `train`,
        `evaluate`, and `predict`. This should be a single `Tensor` or dict with
        `Tensor` values.
    Returns:
      A `Tensor` representing the logits.
    """
    size_checks = []
    batch_size = None

    weight_column_key = _get_weight_column_key(weight_column)

    # The first dimension is assumed to be a batch size and must be consistent
    # among all of the features.
    for key, feature in features.items():
      # Skip weight_column to ensure we don't add size checks to it.
      # These would introduce a dependency on the weight at serving time.
      if key == weight_column_key:
        continue
      first_dim = array_ops.shape(feature)[0]
      if batch_size is None:
        batch_size = first_dim
      else:
        size_checks.append(check_ops.assert_equal(batch_size, first_dim))

    with ops.control_dependencies(size_checks):
      with variable_scope.variable_scope('baseline'):
        bias = variable_scope.get_variable('bias', shape=[num_outputs],
                                           initializer=init_ops.Zeros)
        return math_ops.multiply(bias, array_ops.ones([batch_size,
                                                       num_outputs]))

  return baseline_logit_fn


def _baseline_model_fn(features, labels, mode, head, optimizer,
                       weight_column=None, config=None):
  """Model_fn for baseline models.

  Args:
    features: `Tensor` or dict of `Tensor` (depends on data passed to `train`).
    labels: `Tensor` of labels that are compatible with the `Head` instance.
    mode: Defines whether this is training, evaluation or prediction.
      See `ModeKeys`.
    head: A `Head` instance.
    optimizer: String, `tf.Optimizer` object, or callable that creates the
      optimizer to use for training. If not specified, will use `FtrlOptimizer`
      with a default learning rate of 0.3.
    weight_column: A string or a `_NumericColumn` created by
      `tf.feature_column.numeric_column` defining feature column representing
       weights. It will be multiplied by the loss of the example.
    config: `RunConfig` object to configure the runtime settings.

  Raises:
    KeyError: If weight column is specified but not present.
    ValueError: If features is an empty dictionary.

  Returns:
    An `EstimatorSpec` instance.
  """
  del config  # Unused.

  logit_fn = _baseline_logit_fn_builder(head.logits_dimension, weight_column)
  logits = logit_fn(features)

  def train_op_fn(loss):
    opt = optimizers.get_optimizer_instance(
        optimizer, learning_rate=_LEARNING_RATE)
    return opt.minimize(loss, global_step=training_util.get_global_step())

  return head.create_estimator_spec(
      features=features,
      mode=mode,
      logits=logits,
      labels=labels,
      train_op_fn=train_op_fn)


@estimator_export('estimator.BaselineClassifier')
class BaselineClassifier(estimator.Estimator):
  """A classifier that can establish a simple baseline.

  This classifier ignores feature values and will learn to predict the average
  value of each label. For single-label problems, this will predict the
  probability distribution of the classes as seen in the labels. For multi-label
  problems, this will predict the fraction of examples that are positive for
  each class.

  Example:

  ```python

  # Build BaselineClassifier
  classifier = BaselineClassifier(n_classes=3)

  # Input builders
  def input_fn_train: # returns x, y (where y represents label's class index).
    pass

  def input_fn_eval: # returns x, y (where y represents label's class index).
    pass

  # Fit model.
  classifier.train(input_fn=input_fn_train)

  # Evaluate cross entropy between the test and train labels.
  loss = classifier.evaluate(input_fn=input_fn_eval)["loss"]

  # predict outputs the probability distribution of the classes as seen in
  # training.
  predictions = classifier.predict(new_samples)

  ```

  Input of `train` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column` is not `None`, a feature with
     `key=weight_column` whose value is a `Tensor`.
  """

  def __init__(self,
               model_dir=None,
               n_classes=2,
               weight_column=None,
               label_vocabulary=None,
               optimizer='Ftrl',
               config=None,
               loss_reduction=losses.Reduction.SUM):
    """Initializes a BaselineClassifier instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      n_classes: number of label classes. Default is binary classification.
        It must be greater than 1. Note: Class labels are integers representing
        the class index (i.e. values from 0 to n_classes-1). For arbitrary
        label values (e.g. string labels), convert to class indices first.
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
         weights. It will be multiplied by the loss of the example.
      label_vocabulary: Optional list of strings with size `[n_classes]`
        defining the label vocabulary. Only supported for `n_classes` > 2.
      optimizer: String, `tf.Optimizer` object, or callable that creates the
        optimizer to use for training. If not specified, will use
        `FtrlOptimizer` with a default learning rate of 0.3.
      config: `RunConfig` object to configure the runtime settings.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM`.
    Returns:
      A `BaselineClassifier` estimator.

    Raises:
      ValueError: If `n_classes` < 2.
    """
    if n_classes == 2:
      head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(  # pylint: disable=protected-access
          weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction)
    else:
      head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(  # pylint: disable=protected-access
          n_classes, weight_column=weight_column,
          label_vocabulary=label_vocabulary,
          loss_reduction=loss_reduction)
    def _model_fn(features, labels, mode, config):
      return _baseline_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          optimizer=optimizer,
          weight_column=weight_column,
          config=config)
    super(BaselineClassifier, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config)


@estimator_export('estimator.BaselineRegressor')
class BaselineRegressor(estimator.Estimator):
  """A regressor that can establish a simple baseline.

  This regressor ignores feature values and will learn to predict the average
  value of each label.

  Example:

  ```python

  # Build BaselineRegressor
  regressor = BaselineRegressor()

  # Input builders
  def input_fn_train: # returns x, y (where y is the label).
    pass

  def input_fn_eval: # returns x, y (where y is the label).
    pass

  # Fit model.
  regressor.train(input_fn=input_fn_train)

  # Evaluate squared-loss between the test and train targets.
  loss = regressor.evaluate(input_fn=input_fn_eval)["loss"]

  # predict outputs the mean value seen during training.
  predictions = regressor.predict(new_samples)
  ```

  Input of `train` and `evaluate` should have following features,
    otherwise there will be a `KeyError`:

  * if `weight_column` is not `None`, a feature with
     `key=weight_column` whose value is a `Tensor`.
  """

  def __init__(self,
               model_dir=None,
               label_dimension=1,
               weight_column=None,
               optimizer='Ftrl',
               config=None,
               loss_reduction=losses.Reduction.SUM):
    """Initializes a BaselineRegressor instance.

    Args:
      model_dir: Directory to save model parameters, graph and etc. This can
        also be used to load checkpoints from the directory into a estimator to
        continue training a previously saved model.
      label_dimension: Number of regression targets per example. This is the
        size of the last dimension of the labels and logits `Tensor` objects
        (typically, these have shape `[batch_size, label_dimension]`).
      weight_column: A string or a `_NumericColumn` created by
        `tf.feature_column.numeric_column` defining feature column representing
         weights. It will be multiplied by the loss of the example.
      optimizer: String, `tf.Optimizer` object, or callable that creates the
        optimizer to use for training. If not specified, will use
        `FtrlOptimizer` with a default learning rate of 0.3.
      config: `RunConfig` object to configure the runtime settings.
      loss_reduction: One of `tf.losses.Reduction` except `NONE`. Describes how
        to reduce training loss over batch. Defaults to `SUM`.
    Returns:
      A `BaselineRegressor` estimator.
    """

    head = head_lib._regression_head(  # pylint: disable=protected-access
        label_dimension=label_dimension,
        weight_column=weight_column,
        loss_reduction=loss_reduction)
    def _model_fn(features, labels, mode, config):
      return _baseline_model_fn(
          features=features,
          labels=labels,
          mode=mode,
          head=head,
          optimizer=optimizer,
          config=config)
    super(BaselineRegressor, self).__init__(
        model_fn=_model_fn,
        model_dir=model_dir,
        config=config)
