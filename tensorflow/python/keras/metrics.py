# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=unused-import
"""Built-in metrics.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import sys
import types
import numpy as np
import six

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.losses import categorical_hinge
from tensorflow.python.keras.losses import cosine_proximity
from tensorflow.python.keras.losses import hinge
from tensorflow.python.keras.losses import kullback_leibler_divergence
from tensorflow.python.keras.losses import logcosh
from tensorflow.python.keras.losses import mean_absolute_error
from tensorflow.python.keras.losses import mean_absolute_percentage_error
from tensorflow.python.keras.losses import mean_squared_error
from tensorflow.python.keras.losses import mean_squared_logarithmic_error
from tensorflow.python.keras.losses import poisson
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras.losses import squared_hinge
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.generic_utils import to_list
from tensorflow.python.keras.utils.losses_utils import squeeze_or_expand_dimensions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


@six.add_metaclass(abc.ABCMeta)
class Metric(Layer):
  """Encapsulates metric logic and state.

  Usage:

  ```python
  m = SomeMetric(...)
  for input in ...:
    m.update_state(input)
  print('Final result: ', m.result().numpy())
  ```

  Usage with tf.keras API:

  ```python
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(64, activation='relu'))
  model.add(tf.keras.layers.Dense(10, activation='softmax'))

  model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=[tf.keras.metrics.CategoricalAccuracy()])

  data = np.random.random((1000, 32))
  labels = np.random.random((1000, 10))

  dataset = tf.data.Dataset.from_tensor_slices((data, labels))
  dataset = dataset.batch(32)
  dataset = dataset.repeat()

  model.fit(dataset, epochs=10, steps_per_epoch=30)
  ```

  To be implemented by subclasses:
  * `__init__()`: All state variables should be created in this method by
    calling `self.add_weight()` like: `self.var = self.add_weight(...)`
  * `update_state()`: Has all updates to the state variables like:
    self.var.assign_add(...).
  * `result()`: Computes and returns a value for the metric
    from the state variables.

  Example subclass implementation:

  ```
  class BinaryTruePositives(Metric):
    def __init__(self, name='binary_true_positives', dtype=None):
      super(BinaryTruePositives, self).__init__(name=name, dtype=dtype)
      self.true_positives = self.add_weight(
          'true_positives', initializer=init_ops.zeros_initializer)

    def update_state(self, y_true, y_pred, sample_weight=None):
      y_true = math_ops.cast(y_true, dtypes.bool)
      y_pred = math_ops.cast(y_pred, dtypes.bool)
      y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
          y_pred, y_true, sample_weight)

      values = math_ops.logical_and(
          math_ops.equal(y_true, True), math_ops.equal(y_pred, True))
      values = math_ops.cast(values, self._dtype)
      if sample_weight is not None:
        sample_weight = math_ops.cast(sample_weight, self._dtype)
        values = math_ops.multiply(values, sample_weight)
      state_ops.assign_add(self.true_positives, math_ops.reduce_sum(values))

    def result(self):
      return array_ops.identity(self.true_positives)
  ```
  """

  def __init__(self, name=None, dtype=None):
    super(Metric, self).__init__(name=name, dtype=dtype)
    self.stateful = True  # All metric layers are stateful.
    self.built = True
    self._dtype = K.floatx() if dtype is None else dtypes.as_dtype(dtype).name

  def __new__(cls, *args, **kwargs):
    obj = super(Metric, cls).__new__(cls)

    if sys.version_info < (3,):
      # Wrap methods in `weakmethod` function to remove binding and create a
      # weak reference. This is to remove reference cycle that is created here.
      # This is not an issue in python versions > 3.
      if context.executing_eagerly():
        obj.update_state = metrics_utils.weakmethod(obj.update_state)
      obj.update_state = metrics_utils.weakmethod(
          types.MethodType(
              metrics_utils.update_state_wrapper(obj.update_state), obj))
      result = metrics_utils.weakmethod(obj.result)
      obj.result = metrics_utils.weakmethod(
          types.MethodType(metrics_utils.result_wrapper(result), obj))
    else:
      obj.update_state = types.MethodType(
          metrics_utils.update_state_wrapper(obj.update_state), obj)
      obj.result = types.MethodType(
          metrics_utils.result_wrapper(obj.result), obj)

    return obj

  def __call__(self, *args, **kwargs):
    """Accumulates statistics and then computes metric result value.

    Args:
      *args:
      **kwargs: A mini-batch of inputs to the Metric,
        passed on to `update_state()`.

    Returns:
      The metric value tensor.
    """
    update_op = self.update_state(*args, **kwargs)
    with ops.control_dependencies([update_op]):
      result_t = self.result()

      # We are adding the metric object as metadata on the result tensor.
      # This is required when we want to use a metric with `add_metric` API on
      # a Model/Layer in graph mode. This metric instance will later be used
      # to reset variable state after each epoch of training.
      # Example:
      #   model = Model()
      #   model.add_metric(Mean()(values), name='mean')
      if not context.executing_eagerly():
        result_t._metric_obj = self  # pylint: disable=protected-access
      return result_t

  def reset_states(self):
    """Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.
    """
    for v in self.variables:
      K.set_value(v, 0)

  @abc.abstractmethod
  def update_state(self, *args, **kwargs):
    """Accumulates statistics for the metric.

    Note: This function is executed as a graph function in graph mode.
    This means:
      a) Operations on the same resource are executed in textual order.
         This should make it easier to do things like add the updated
         value of a variable to another, for example.
      b) You don't need to worry about collecting the update ops to execute.
         All update ops added to the graph by this function will be executed.
      As a result, code should generally work the same way with graph or
      eager execution.
    and adds the update op to the metric layer.

    Args:
      *args:
      **kwargs: A mini-batch of inputs to the Metric.
    """
    NotImplementedError('Must be implemented in subclasses.')

  @abc.abstractmethod
  def result(self):
    """Computes and returns the metric value tensor.

    Result computation is an idempotent operation that simply calculates the
    metric value using the state variables.
    """
    NotImplementedError('Must be implemented in subclasses.')

  @classmethod
  def from_config(cls, config):
    if 'trainable' in config:
      config.pop('trainable')
    return cls(**config)

  ### For use by subclasses ###
  @doc_controls.for_subclass_implementers
  def add_weight(self,
                 name,
                 shape=(),
                 aggregation=tf_variables.VariableAggregation.SUM,
                 synchronization=tf_variables.VariableSynchronization.ON_READ,
                 initializer=None):
    """Adds state variable. Only for use by subclasses."""
    return super(Metric, self).add_weight(
        name=name,
        shape=shape,
        dtype=self._dtype,
        trainable=False,
        initializer=initializer,
        collections=[],
        synchronization=synchronization,
        aggregation=aggregation)

  ### End: For use by subclasses ###


@keras_export('keras.metrics.Mean')
class Mean(Metric):
  """Computes the (weighted) mean of the given values.

  For example, if values is [1, 3, 5, 7] then the mean is 4.
  If the weights were specified as [1, 1, 0, 0] then the mean would be 2.

  This metric creates two variables, `total` and `count` that are used to
  compute the average of `values`. This average is ultimately returned as `mean`
  which is an idempotent operation that simply divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.Mean()
  m.update_state([1, 3, 5, 7])
  print('Final result: ', m.result().numpy())  # Final result: 4.0
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.add_metric(tf.keras.metrics.Mean(name='mean_1')(outputs))
  model.compile('sgd', loss='mse')
  ```
  """

  def __init__(self, name='mean', dtype=None):
    """Creates a `Mean` instance.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(Mean, self).__init__(name=name, dtype=dtype)
    # Create new state variables
    self.total = self.add_weight(
        'total', initializer=init_ops.zeros_initializer)
    self.count = self.add_weight(
        'count', initializer=init_ops.zeros_initializer)

  def update_state(self, values, sample_weight=None):
    """Accumulates statistics for computing the mean.

    For example, if `values` is [1, 3, 5, 7] then the mean is 4. If
    the `sample_weight` is specified as [1, 1, 0, 0] then the mean would be 2.

    Args:
      values: Per-example value.
      sample_weight: Optional weighting of each example. Defaults to 1.

    Returns:
      Update op.
    """
    values = math_ops.cast(values, self._dtype)
    if sample_weight is None:
      num_values = math_ops.cast(array_ops.size(values), self._dtype)
    else:
      sample_weight = math_ops.cast(sample_weight, self._dtype)

      # Update dimensions of weights to match with values if possible.
      values, _, sample_weight = squeeze_or_expand_dimensions(
          values, None, sample_weight)
      try:
        # Broadcast weights if possible.
        sample_weight = weights_broadcast_ops.broadcast_weights(
            sample_weight, values)
      except ValueError:
        # Reduce values to same ndim as weight array
        ndim = K.ndim(values)
        weight_ndim = K.ndim(sample_weight)
        values = math_ops.reduce_mean(
            values, axis=list(range(weight_ndim, ndim)))

      num_values = math_ops.reduce_sum(sample_weight)
      values = math_ops.multiply(values, sample_weight)
    values = math_ops.reduce_sum(values)

    # Update state variables. Count should be updated only when total is
    # updated.
    update_total_op = state_ops.assign_add(self.total, values)
    with ops.control_dependencies([update_total_op]):
      return state_ops.assign_add(self.count, num_values)

  def result(self):
    return math_ops.div_no_nan(self.total, self.count)


class MeanMetricWrapper(Mean):
  """Wraps a stateless metric function with the Mean metric."""

  def __init__(self, fn, name=None, dtype=None, **kwargs):
    """Creates a `MeanMetricWrapper` instance.

    Args:
      fn: The metric function to wrap, with signature
        `fn(y_true, y_pred, **kwargs)`.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
    self._fn = fn
    self._fn_kwargs = kwargs

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates metric statistics.

    `y_true` and `y_pred` should have the same shape.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be
        a `Tensor` whose rank is either 0, or the same rank as `y_true`,
        and must be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
        y_pred, y_true, sample_weight)

    matches = self._fn(y_true, y_pred, **self._fn_kwargs)
    return super(MeanMetricWrapper, self).update_state(
        matches, sample_weight=sample_weight)

  def get_config(self):
    config = {'fn': self._fn}
    config.update(self._fn_kwargs)
    base_config = super(MeanMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.metrics.Accuracy')
class Accuracy(MeanMetricWrapper):
  """Calculates how often predictions matches labels.

  For example, if `y_true` is [1, 2, 3, 4] and `y_pred` is [0, 2, 3, 4]
  then the accuracy is 3/4 or .75.  If the weights were specified as
  [1, 1, 0, 0] then the accuracy would be 1/2 or .5.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `binary accuracy`: an idempotent operation that simply
  divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.Accuracy()
  m.update_state([1, 2, 3, 4], [0, 2, 3, 4])
  print('Final result: ', m.result().numpy())  # Final result: 0.75
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.Accuracy()])
  ```
  """

  def __init__(self, name='accuracy', dtype=None):
    super(Accuracy, self).__init__(accuracy, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(Accuracy, cls).from_config(config)


@keras_export('keras.metrics.BinaryAccuracy')
class BinaryAccuracy(MeanMetricWrapper):
  """Calculates how often predictions matches labels.

  For example, if `y_true` is [1, 1, 0, 0] and `y_pred` is [0.98, 1, 0, 0.6]
  then the binary accuracy is 3/4 or .75.  If the weights were specified as
  [1, 0, 0, 1] then the binary accuracy would be 1/2 or .5.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `binary accuracy`: an idempotent operation that simply
  divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.BinaryAccuracy()
  m.update_state([1, 1, 0, 0], [0.98, 1, 0, 0.6])
  print('Final result: ', m.result().numpy())  # Final result: 0.75
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.BinaryAccuracy()])
  ```
  """

  def __init__(self, name='binary_accuracy', dtype=None, threshold=0.5):
    """Creates a `BinaryAccuracy` instance.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
      threshold: (Optional) Float representing the threshold for deciding
      whether prediction values are 1 or 0.
    """
    super(BinaryAccuracy, self).__init__(
        binary_accuracy, name, dtype=dtype, threshold=threshold)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(BinaryAccuracy, cls).from_config(config)


@keras_export('keras.metrics.CategoricalAccuracy')
class CategoricalAccuracy(MeanMetricWrapper):
  """Calculates how often predictions matches labels.

  For example, if `y_true` is [[0, 0, 1], [0, 1, 0]] and `y_pred` is
  [[0.1, 0.9, 0.8], [0.05, 0.95, 0]] then the categorical accuracy is 1/2 or .5.
  If the weights were specified as [0.7, 0.3] then the categorical accuracy
  would be .3.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `categorical accuracy`: an idempotent operation that
  simply divides `total` by `count`.

  `y_pred` and `y_true` should be passed in as vectors of probabilities, rather
  than as labels. If necessary, use `tf.one_hot` to expand `y_true` as a vector.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.CategoricalAccuracy()
  m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
  print('Final result: ', m.result().numpy())  # Final result: 0.5
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile(
    'sgd',
    loss='mse',
    metrics=[tf.keras.metrics.CategoricalAccuracy()])
  ```
  """

  def __init__(self, name='categorical_accuracy', dtype=None):
    """Creates a `CategoricalAccuracy` instance.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(CategoricalAccuracy, self).__init__(
        categorical_accuracy, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(CategoricalAccuracy, cls).from_config(config)


@keras_export('keras.metrics.SparseCategoricalAccuracy')
class SparseCategoricalAccuracy(MeanMetricWrapper):
  """Calculates how often predictions matches integer labels.

  For example, if `y_true` is [[2], [1]] and `y_pred` is
  [[0.1, 0.9, 0.8], [0.05, 0.95, 0]] then the categorical accuracy is 1/2 or .5.
  If the weights were specified as [0.7, 0.3] then the categorical accuracy
  would be .3.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `sparse categorical accuracy`: an idempotent operation
  that simply divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.SparseCategoricalAccuracy()
  m.update_state([[2], [1]], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
  print('Final result: ', m.result().numpy())  # Final result: 0.5
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
  ```
  """

  def __init__(self, name='sparse_categorical_accuracy', dtype=None):
    super(SparseCategoricalAccuracy, self).__init__(
        sparse_categorical_accuracy, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(SparseCategoricalAccuracy, cls).from_config(config)


class TopKCategoricalAccuracy(MeanMetricWrapper):
  """Computes how often targets are in the top `K` predictions.

  Usage:

  ```python
  m = tf.keras.metrics.TopKCategoricalAccuracy()
  m.update_state([[0, 0, 1], [0, 1, 0]], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
  print('Final result: ', m.result().numpy())  # Final result: 1.0
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', metrics=[tf.keras.metrics.TopKCategoricalAccuracy()])
  ```
  """

  def __init__(self, k=5, name='top_k_categorical_accuracy', dtype=None):
    """Creates a `TopKCategoricalAccuracy` instance.

    Args:
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to 5.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(TopKCategoricalAccuracy, self).__init__(
        top_k_categorical_accuracy, name, dtype=dtype, k=k)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(TopKCategoricalAccuracy, cls).from_config(config)


class SparseTopKCategoricalAccuracy(MeanMetricWrapper):
  """Computes how often integer targets are in the top `K` predictions.

  Usage:

  ```python
  m = tf.keras.metrics.SparseTopKCategoricalAccuracy()
  m.update_state([2, 1], [[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
  print('Final result: ', m.result().numpy())  # Final result: 1.0
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile(
    'sgd',
    metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy()])
  ```
  """

  def __init__(self, k=5, name='sparse_top_k_categorical_accuracy', dtype=None):
    """Creates a `SparseTopKCategoricalAccuracy` instance.

    Args:
      k: (Optional) Number of top elements to look at for computing accuracy.
        Defaults to 5.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(SparseTopKCategoricalAccuracy, self).__init__(
        sparse_top_k_categorical_accuracy, name, dtype=dtype, k=k)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(SparseTopKCategoricalAccuracy, cls).from_config(config)


class _ConfusionMatrixConditionCount(Metric):
  """Calculates the number of the given confusion matrix condition."""

  def __init__(self,
               confusion_matrix_cond,
               thresholds=None,
               name=None,
               dtype=None):
    """Creates a `_ConfusionMatrixConditionCount` instance.

    Args:
      confusion_matrix_cond: One of `metrics_utils.ConfusionMatrix` conditions.
      thresholds: (Optional) Defaults to 0.5. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(_ConfusionMatrixConditionCount, self).__init__(name=name, dtype=dtype)
    self._confusion_matrix_cond = confusion_matrix_cond
    self.init_thresholds = thresholds
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=0.5)
    self.accumulator = self.add_weight(
        'accumulator',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates the given confusion matrix condition statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    return metrics_utils.update_confusion_matrix_variables(
        {self._confusion_matrix_cond: self.accumulator}, y_true, y_pred,
        self.thresholds, sample_weight)

  def result(self):
    if len(self.thresholds) == 1:
      result = self.accumulator[0]
    else:
      result = self.accumulator
    return ops.convert_to_tensor(result)

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))

  def get_config(self):
    config = {'thresholds': self.init_thresholds}
    base_config = super(_ConfusionMatrixConditionCount, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.metrics.FalsePositives')
class FalsePositives(_ConfusionMatrixConditionCount):
  """Calculates the number of false positives.

  For example, if `y_true` is [0, 1, 0, 0] and `y_pred` is [0, 0, 1, 1]
  then the false positives value is 2.  If the weights were specified as
  [0, 0, 1, 0] then the false positives value would be 1.

  If `sample_weight` is given, calculates the sum of the weights of
  false positives. This metric creates one local variable, `accumulator`
  that is used to keep track of the number of false positives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.FalsePositives()
  m.update_state([0, 1, 0, 0], [0, 0, 1, 1])
  print('Final result: ', m.result().numpy())  # Final result: 2
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.FalsePositives()])
  ```
  """

  def __init__(self, thresholds=None, name=None, dtype=None):
    """Creates a `FalsePositives` instance.

    Args:
      thresholds: (Optional) Defaults to 0.5. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(FalsePositives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_POSITIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)


@keras_export('keras.metrics.FalseNegatives')
class FalseNegatives(_ConfusionMatrixConditionCount):
  """Calculates the number of false negatives.

  For example, if `y_true` is [0, 1, 1, 1] and `y_pred` is [0, 1, 0, 0]
  then the false negatives value is 2.  If the weights were specified as
  [0, 0, 1, 0] then the false negatives value would be 1.

  If `sample_weight` is given, calculates the sum of the weights of
  false negatives. This metric creates one local variable, `accumulator`
  that is used to keep track of the number of false negatives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.FalseNegatives()
  m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
  print('Final result: ', m.result().numpy())  # Final result: 2
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.FalseNegatives()])
  ```
  """

  def __init__(self, thresholds=None, name=None, dtype=None):
    """Creates a `FalseNegatives` instance.

    Args:
      thresholds: (Optional) Defaults to 0.5. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(FalseNegatives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.FALSE_NEGATIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)


@keras_export('keras.metrics.TrueNegatives')
class TrueNegatives(_ConfusionMatrixConditionCount):
  """Calculates the number of true negatives.

  For example, if `y_true` is [0, 1, 0, 0] and `y_pred` is [1, 1, 0, 0]
  then the true negatives value is 2.  If the weights were specified as
  [0, 0, 1, 0] then the true negatives value would be 1.

  If `sample_weight` is given, calculates the sum of the weights of
  true negatives. This metric creates one local variable, `accumulator`
  that is used to keep track of the number of true negatives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.TrueNegatives()
  m.update_state([0, 1, 0, 0], [1, 1, 0, 0])
  print('Final result: ', m.result().numpy())  # Final result: 2
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.TrueNegatives()])
  ```
  """

  def __init__(self, thresholds=None, name=None, dtype=None):
    """Creates a `TrueNegatives` instance.

    Args:
      thresholds: (Optional) Defaults to 0.5. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(TrueNegatives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_NEGATIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)


@keras_export('keras.metrics.TruePositives')
class TruePositives(_ConfusionMatrixConditionCount):
  """Calculates the number of true positives.

  For example, if `y_true` is [0, 1, 1, 1] and `y_pred` is [1, 0, 1, 1]
  then the true positives value is 2.  If the weights were specified as
  [0, 0, 1, 0] then the true positives value would be 1.

  If `sample_weight` is given, calculates the sum of the weights of
  true positives. This metric creates one local variable, `true_positives`
  that is used to keep track of the number of true positives.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.TruePositives()
  m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
  print('Final result: ', m.result().numpy())  # Final result: 2
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.TruePositives()])
  ```
  """

  def __init__(self, thresholds=None, name=None, dtype=None):
    """Creates a `TruePositives` instance.

    Args:
      thresholds: (Optional) Defaults to 0.5. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(TruePositives, self).__init__(
        confusion_matrix_cond=metrics_utils.ConfusionMatrix.TRUE_POSITIVES,
        thresholds=thresholds,
        name=name,
        dtype=dtype)


@keras_export('keras.metrics.Precision')
class Precision(Metric):
  """Computes the precision of the predictions with respect to the labels.

  For example, if `y_true` is [0, 1, 1, 1] and `y_pred` is [1, 0, 1, 1]
  then the precision value is 2/(2+1) ie. 0.66. If the weights were specified as
  [0, 0, 1, 0] then the precision value would be 1.

  The metric creates two local variables, `true_positives` and `false_positives`
  that are used to compute the precision. This value is ultimately returned as
  `precision`, an idempotent operation that simply divides `true_positives`
  by the sum of `true_positives` and `false_positives`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.Precision()
  m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
  print('Final result: ', m.result().numpy())  # Final result: 0.66
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.Precision()])
  ```
  """

  def __init__(self, thresholds=None, name=None, dtype=None):
    """Creates a `Precision` instance.

    Args:
      thresholds: (Optional) Defaults to 0.5. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(Precision, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=0.5)
    self.tp = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.fp = self.add_weight(
        'false_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates true positive and false positive statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    return metrics_utils.update_confusion_matrix_variables({
        metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.tp,
        metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.fp
    }, y_true, y_pred, self.thresholds, sample_weight)

  def result(self):
    result = math_ops.div_no_nan(self.tp, self.tp + self.fp)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))

  def get_config(self):
    config = {'thresholds': self.init_thresholds}
    base_config = super(Precision, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.metrics.Recall')
class Recall(Metric):
  """Computes the recall of the predictions with respect to the labels.

  For example, if `y_true` is [0, 1, 1, 1] and `y_pred` is [1, 0, 1, 1]
  then the recall value is 2/(2+1) ie. 0.66. If the weights were specified as
  [0, 0, 1, 0] then the recall value would be 1.

  This metric creates two local variables, `true_positives` and
  `false_negatives`, that are used to compute the recall. This value is
  ultimately returned as `recall`, an idempotent operation that simply divides
  `true_positives` by the sum of `true_positives` and `false_negatives`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  Usage:

  ```python
  m = tf.keras.metrics.Recall()
  m.update_state([0, 1, 1, 1], [1, 0, 1, 1])
  print('Final result: ', m.result().numpy())  # Final result: 0.66
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.Recall()])
  ```
  """

  def __init__(self, thresholds=None, name=None, dtype=None):
    """Creates a `Recall` instance.

    Args:
      thresholds: (Optional) Defaults to 0.5. A float value or a python
        list/tuple of float threshold values in [0, 1]. A threshold is compared
        with prediction values to determine the truth value of predictions
        (i.e., above the threshold is `true`, below is `false`). One metric
        value is generated for each threshold value.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(Recall, self).__init__(name=name, dtype=dtype)
    self.init_thresholds = thresholds
    self.thresholds = metrics_utils.parse_init_thresholds(
        thresholds, default_threshold=0.5)
    self.tp = self.add_weight(
        'true_positives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)
    self.fn = self.add_weight(
        'false_negatives',
        shape=(len(self.thresholds),),
        initializer=init_ops.zeros_initializer)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates true positive and false negative statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    return metrics_utils.update_confusion_matrix_variables({
        metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.tp,
        metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.fn
    }, y_true, y_pred, self.thresholds, sample_weight)

  def result(self):
    result = math_ops.div_no_nan(self.tp, self.tp + self.fn)
    return result[0] if len(self.thresholds) == 1 else result

  def reset_states(self):
    num_thresholds = len(to_list(self.thresholds))
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))

  def get_config(self):
    config = {'thresholds': self.init_thresholds}
    base_config = super(Recall, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@six.add_metaclass(abc.ABCMeta)
class SensitivitySpecificityBase(Metric):
  """Abstract base class for computing sensitivity and specificity.

  For additional information about specificity and sensitivity, see the
  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity
  """

  def __init__(self, value, num_thresholds=200, name=None, dtype=None):
    super(SensitivitySpecificityBase, self).__init__(name=name, dtype=dtype)
    if num_thresholds <= 0:
      raise ValueError('`num_thresholds` must be > 0.')
    self.value = value
    self.tp = self.add_weight(
        'true_positives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.tn = self.add_weight(
        'true_negatives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.fp = self.add_weight(
        'false_positives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)
    self.fn = self.add_weight(
        'false_negatives',
        shape=(num_thresholds,),
        initializer=init_ops.zeros_initializer)

    # Compute `num_thresholds` thresholds in [0, 1]
    if num_thresholds == 1:
      self.thresholds = [0.5]
    else:
      thresholds = [(i + 1) * 1.0 / (num_thresholds - 1)
                    for i in range(num_thresholds - 2)]
      self.thresholds = [0.0] + thresholds + [1.0]

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates confusion matrix statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    return metrics_utils.update_confusion_matrix_variables({
        metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.tp,
        metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.tn,
        metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.fp,
        metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.fn,
    }, y_true, y_pred, self.thresholds, sample_weight)

  def reset_states(self):
    num_thresholds = len(self.thresholds)
    for v in self.variables:
      K.set_value(v, np.zeros((num_thresholds,)))


@keras_export('keras.metrics.SensitivityAtSpecificity')
class SensitivityAtSpecificity(SensitivitySpecificityBase):
  """Computes the sensitivity at a given specificity.

  `Sensitivity` measures the proportion of actual positives that are correctly
  identified as such (tp / (tp + fn)).
  `Specificity` measures the proportion of actual negatives that are correctly
  identified as such (tn / (tn + fp)).

  This metric creates four local variables, `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` that are used to compute the
  sensitivity at the given specificity. The threshold for the given specificity
  value is computed and used to evaluate the corresponding sensitivity.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  For additional information about specificity and sensitivity, see the
  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

  Usage:

  ```python
  m = tf.keras.metrics.SensitivityAtSpecificity(0.4, num_thresholds=1)
  m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
  print('Final result: ', m.result().numpy())  # Final result: 0.5
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SensitivityAtSpecificity()])
  ```
  """

  def __init__(self, specificity, num_thresholds=200, name=None, dtype=None):
    """Creates a `SensitivityAtSpecificity` instance.

    Args:
      specificity: A scalar value in range `[0, 1]`.
      num_thresholds: (Optional) Defaults to 200. The number of thresholds to
        use for matching the given specificity.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    if specificity < 0 or specificity > 1:
      raise ValueError('`specificity` must be in the range [0, 1].')
    self.specificity = specificity
    self.num_thresholds = num_thresholds
    super(SensitivityAtSpecificity, self).__init__(
        specificity, num_thresholds=num_thresholds, name=name, dtype=dtype)

  def result(self):
    # Calculate specificities at all the thresholds.
    specificities = math_ops.div_no_nan(self.tn, self.tn + self.fp)

    # Find the index of the threshold where the specificity is closest to the
    # given specificity.
    min_index = math_ops.argmin(
        math_ops.abs(specificities - self.value), axis=0)
    min_index = math_ops.cast(min_index, dtypes.int32)

    # Compute sensitivity at that index.
    return math_ops.div_no_nan(self.tp[min_index],
                               self.tp[min_index] + self.fn[min_index])

  def get_config(self):
    config = {
        'num_thresholds': self.num_thresholds,
        'specificity': self.specificity
    }
    base_config = super(SensitivityAtSpecificity, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.metrics.SpecificityAtSensitivity')
class SpecificityAtSensitivity(SensitivitySpecificityBase):
  """Computes the specificity at a given sensitivity.

  `Sensitivity` measures the proportion of actual positives that are correctly
  identified as such (tp / (tp + fn)).
  `Specificity` measures the proportion of actual negatives that are correctly
  identified as such (tn / (tn + fp)).

  This metric creates four local variables, `true_positives`, `true_negatives`,
  `false_positives` and `false_negatives` that are used to compute the
  specificity at the given sensitivity. The threshold for the given sensitivity
  value is computed and used to evaluate the corresponding specificity.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.

  For additional information about specificity and sensitivity, see the
  following: https://en.wikipedia.org/wiki/Sensitivity_and_specificity

  Usage:

  ```python
  m = tf.keras.metrics.SpecificityAtSensitivity(0.8, num_thresholds=1)
  m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
  print('Final result: ', m.result().numpy())  # Final result: 1.0
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.keras.metrics.SpecificityAtSensitivity()])
  ```
  """

  def __init__(self, sensitivity, num_thresholds=200, name=None, dtype=None):
    """Creates a `SpecificityAtSensitivity` instance.

    Args:
      sensitivity: A scalar value in range `[0, 1]`.
      num_thresholds: (Optional) Defaults to 200. The number of thresholds to
        use for matching the given specificity.
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    if sensitivity < 0 or sensitivity > 1:
      raise ValueError('`sensitivity` must be in the range [0, 1].')
    self.sensitivity = sensitivity
    self.num_thresholds = num_thresholds
    super(SpecificityAtSensitivity, self).__init__(
        sensitivity, num_thresholds=num_thresholds, name=name, dtype=dtype)

  def result(self):
    # Calculate sensitivities at all the thresholds.
    sensitivities = math_ops.div_no_nan(self.tp, self.tp + self.fn)

    # Find the index of the threshold where the sensitivity is closest to the
    # given specificity.
    min_index = math_ops.argmin(
        math_ops.abs(sensitivities - self.value), axis=0)
    min_index = math_ops.cast(min_index, dtypes.int32)

    # Compute specificity at that index.
    return math_ops.div_no_nan(self.tn[min_index],
                               self.tn[min_index] + self.fp[min_index])

  def get_config(self):
    config = {
        'num_thresholds': self.num_thresholds,
        'sensitivity': self.sensitivity
    }
    base_config = super(SpecificityAtSensitivity, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.metrics.CosineProximity')
class CosineProximity(MeanMetricWrapper):
  """Computes the cosine distance between the labels and predictions.

  For example, if `y_true` is [0, 1, 1], and `y_pred` is [1, 0, 1], the cosine
  proximity is -0.5.

  This metric keeps the average cosine distance between `predictions` and
  `labels` over a stream of data.

  Usage:
  ```python
  m = tf.metrics.CosineProximity()
  m.update_state([0, 1, 1], [1, 0, 1])
  print('Final result: ', m.result().numpy())  # Final result: -0.5
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile(
      'sgd',
      loss='mse',
      metrics=[tf.metrics.CosineProximity()])
  ```
  """

  def __init__(self, name='cosine_proximity', dtype=None):
    super(CosineProximity, self).__init__(cosine, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(CosineProximity, cls).from_config(config)


@keras_export('keras.metrics.MeanAbsoluteError')
class MeanAbsoluteError(MeanMetricWrapper):
  """Computes the mean absolute error between the labels and predictions.

  For example, if `y_true` is [0., 0., 1., 1.], and `y_pred` is [1., 1., 1., 0.]
  the mean absolute error is 3/4 (0.75).

  Usage:
  ```python
  m = tf.metrics.MeanAbsoluteError()
  m.update_state([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Final result: ', m.result().numpy())  # Final result: 0.75
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', metrics=[tf.keras.metrics.MeanAbsoluteError()])
  ```
  """

  def __init__(self, name='mean_absolute_error', dtype=None):
    super(MeanAbsoluteError, self).__init__(
        mean_absolute_error, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(MeanAbsoluteError, cls).from_config(config)


@keras_export('keras.metrics.MeanAbsolutePercentageError')
class MeanAbsolutePercentageError(MeanMetricWrapper):
  """Computes the mean absolute percentage error between `y_true` and `y_pred`.

  For example, if `y_true` is [0., 0., 1., 1.], and `y_pred` is [1., 1., 1., 0.]
  the mean absolute percentage error is 5e+08.

  Usage:

  ```python
  m = tf.keras.metrics.MeanAbsolutePercentageError()
  m.update_state([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Final result: ', m.result().numpy())  # Final result: 5e+08
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
  ```
  """

  def __init__(self, name='mean_absolute_percentage_error', dtype=None):
    super(MeanAbsolutePercentageError, self).__init__(
        mean_absolute_percentage_error, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(MeanAbsolutePercentageError, cls).from_config(config)


@keras_export('keras.metrics.MeanSquaredError')
class MeanSquaredError(MeanMetricWrapper):
  """Computes the mean squared error between `y_true` and `y_pred`.

  For example, if `y_true` is [0., 0., 1., 1.], and `y_pred` is [1., 1., 1., 0.]
  the mean squared error is 3/4 (0.75).

  Usage:

  ```python
  m = tf.keras.metrics.MeanSquaredError()
  m.update_state([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Final result: ', m.result().numpy())  # Final result: 0.75
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', metrics=[tf.keras.metrics.MeanSquaredError()])
  ```
  """

  def __init__(self, name='mean_squared_error', dtype=None):
    super(MeanSquaredError, self).__init__(
        mean_squared_error, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(MeanSquaredError, cls).from_config(config)


@keras_export('keras.metrics.MeanSquaredLogarithmicError')
class MeanSquaredLogarithmicError(MeanMetricWrapper):
  """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

  For example, if `y_true` is [0., 0., 1., 1.], and `y_pred` is [1., 1., 1., 0.]
  the mean squared logarithmic error is 0.36034.

  Usage:

  ```python
  m = tf.keras.metrics.MeanSquaredLogarithmicError()
  m.update_state([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Final result: ', m.result().numpy())  # Final result: 0.36034
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', metrics=[tf.keras.metrics.MeanSquaredLogarithmicError()])
  ```
  """

  def __init__(self, name='mean_squared_logarithmic_error', dtype=None):
    super(MeanSquaredLogarithmicError, self).__init__(
        mean_squared_logarithmic_error, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(MeanSquaredLogarithmicError, cls).from_config(config)


@keras_export('keras.metrics.Hinge')
class Hinge(MeanMetricWrapper):
  """Computes the hinge metric between `y_true` and `y_pred`.

  For example, if `y_true` is [0., 1., 1.], and `y_pred` is [1., 0., 1.]
  the hinge metric value is 0.66.

  Usage:

  ```python
  m = tf.keras.metrics.Hinge()
  m.update_state([0., 1., 1.], [1., 0., 1.])
  print('Final result: ', m.result().numpy())  # Final result: 0.66
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', metrics=[tf.keras.metrics.Hinge()])
  ```
  """

  def __init__(self, name='hinge', dtype=None):
    super(Hinge, self).__init__(hinge, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(Hinge, cls).from_config(config)


@keras_export('keras.metrics.SquaredHinge')
class SquaredHinge(MeanMetricWrapper):
  """Computes the squared hinge metric between `y_true` and `y_pred`.

  For example, if `y_true` is [0., 1., 1.], and `y_pred` is [1., 0., 1.]
  the squared hinge metric value is 0.66.

  Usage:

  ```python
  m = tf.keras.metrics.SquaredHinge()
  m.update_state([0., 1., 1.], [1., 0., 1.])
  print('Final result: ', m.result().numpy())  # Final result: 0.66
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', metrics=[tf.keras.metrics.SquaredHinge()])
  ```
  """

  def __init__(self, name='squared_hinge', dtype=None):
    super(SquaredHinge, self).__init__(squared_hinge, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(SquaredHinge, cls).from_config(config)


@keras_export('keras.metrics.CategoricalHinge')
class CategoricalHinge(MeanMetricWrapper):
  """Computes the categorical hinge metric between `y_true` and `y_pred`.

  For example, if `y_true` is [0., 1., 1.], and `y_pred` is [1., 0., 1.]
  the categorical hinge metric value is 1.0.

  Usage:

  ```python
  m = tf.keras.metrics.CategoricalHinge()
  m.update_state([0., 1., 1.], [1., 0., 1.])
  print('Final result: ', m.result().numpy())  # Final result: 1.0
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', metrics=[tf.keras.metrics.CategoricalHinge()])
  ```
  """

  def __init__(self, name='categorical_hinge', dtype=None):
    super(CategoricalHinge, self).__init__(categorical_hinge, name, dtype=dtype)

  @classmethod
  def from_config(cls, config):
    if 'fn' in config:
      config.pop('fn')
    return super(CategoricalHinge, cls).from_config(config)


class RootMeanSquaredError(Mean):
  """Computes root mean squared error metric between `y_true` and `y_pred`.

  Usage:

  ```python
  m = tf.keras.metrics.RootMeanSquaredError()
  m.update_state([2., 4., 6.], [1., 3., 2.])
  print('Final result: ', m.result().numpy())  # Final result: 2.449
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', metrics=[tf.keras.metrics.RootMeanSquaredError()])
  ```
  """

  def __init__(self, name='root_mean_squared_error', dtype=None):
    super(RootMeanSquaredError, self).__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    """Accumulates root mean squared error statistics.

    Args:
      y_true: The ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional weighting of each example. Defaults to 1. Can be a
        `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
        be broadcastable to `y_true`.

    Returns:
      Update op.
    """
    y_true = math_ops.cast(y_true, self._dtype)
    y_pred = math_ops.cast(y_pred, self._dtype)
    y_pred, y_true, sample_weight = squeeze_or_expand_dimensions(
        y_pred, y_true, sample_weight)
    error_sq = math_ops.square(y_pred - y_true)
    return super(RootMeanSquaredError, self).update_state(
        error_sq, sample_weight=sample_weight)

  def result(self):
    return math_ops.sqrt(math_ops.div_no_nan(self.total, self.count))


def accuracy(y_true, y_pred):
  y_pred.get_shape().assert_is_compatible_with(y_true.get_shape())
  if y_true.dtype != y_pred.dtype:
    y_pred = math_ops.cast(y_pred, y_true.dtype)
  return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())


@keras_export('keras.metrics.binary_accuracy')
def binary_accuracy(y_true, y_pred, threshold=0.5):
  threshold = math_ops.cast(threshold, y_pred.dtype)
  y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
  return K.mean(math_ops.equal(y_true, y_pred), axis=-1)


@keras_export('keras.metrics.categorical_accuracy')
def categorical_accuracy(y_true, y_pred):
  return math_ops.cast(
      math_ops.equal(
          math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)),
      K.floatx())


@keras_export('keras.metrics.sparse_categorical_accuracy')
def sparse_categorical_accuracy(y_true, y_pred):
  y_pred_rank = ops.convert_to_tensor(y_pred).get_shape().ndims
  y_true_rank = ops.convert_to_tensor(y_true).get_shape().ndims
  # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
  if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
      K.int_shape(y_true)) == len(K.int_shape(y_pred))):
    y_true = array_ops.squeeze(y_true, [-1])
  y_pred = math_ops.argmax(y_pred, axis=-1)

  # If the predicted output and actual output types don't match, force cast them
  # to match.
  if K.dtype(y_pred) != K.dtype(y_true):
    y_pred = math_ops.cast(y_pred, K.dtype(y_true))

  return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())


@keras_export('keras.metrics.top_k_categorical_accuracy')
def top_k_categorical_accuracy(y_true, y_pred, k=5):
  return K.mean(
      nn.in_top_k(y_pred, math_ops.argmax(y_true, axis=-1), k), axis=-1)


@keras_export('keras.metrics.sparse_top_k_categorical_accuracy')
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
  y_pred_rank = ops.convert_to_tensor(y_pred).get_shape().ndims
  y_true_rank = ops.convert_to_tensor(y_true).get_shape().ndims
  # If the shape of y_true is (num_samples, 1), squeeze to (num_samples,)
  if (y_true_rank is not None) and (y_pred_rank is not None) and (len(
      K.int_shape(y_true)) == len(K.int_shape(y_pred))):
    y_true = array_ops.squeeze(y_true, [-1])

  return K.mean(nn.in_top_k(y_pred, math_ops.cast(y_true, 'int32'), k), axis=-1)

# Aliases

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


def clone_metric(metric):
  """Returns a clone of the metric if stateful, otherwise returns it as is."""
  if isinstance(metric, Metric):
    return metric.__class__.from_config(metric.get_config())
  return metric


def clone_metrics(metrics):
  """Clones the given metric list/dict."""
  if metrics is None:
    return None
  if isinstance(metrics, dict):
    return {key: clone_metric(value) for key, value in metrics.items()}
  return [clone_metric(metric) for metric in metrics]


@keras_export('keras.metrics.serialize')
def serialize(metric):
  return serialize_keras_object(metric)


@keras_export('keras.metrics.deserialize')
def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='metric function')


@keras_export('keras.metrics.get')
def get(identifier):
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    return deserialize(str(identifier))
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret '
                     'metric function identifier: %s' % identifier)
