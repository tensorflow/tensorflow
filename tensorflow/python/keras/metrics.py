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

from abc import ABCMeta
from abc import abstractmethod

import functools
import sys
import types
import weakref
import six

from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.losses import binary_crossentropy
from tensorflow.python.keras.losses import categorical_crossentropy
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
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.ops import weights_broadcast_ops
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
from tensorflow.tools.docs import doc_controls


def check_is_tensor_or_operation(x, name):
  """Raises type error if the given input is not a tensor or operation."""
  if not (isinstance(x, ops.Tensor) or isinstance(x, ops.Operation)):
    raise TypeError('{0} must be a Tensor or Operation, given: {1}'.format(
        name, x))


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


def update_state_wrapper(update_state_fn):
  """Decorator to wrap metric `update_state()` with `add_update()`.

  Args:
    update_state_fn: function that accumulates metric statistics.

  Returns:
    Decorated function that wraps `update_state_fn()` with `add_update()`.
  """

  def decorated(metric_obj, *args, **kwargs):
    """Decorated function with `add_update()`."""

    update_op = update_state_fn(*args, **kwargs)
    if update_op is not None:  # update_op will be None in eager execution.
      metric_obj.add_update(update_op, inputs=True)
      check_is_tensor_or_operation(
          update_op, 'Metric {0}\'s update'.format(metric_obj.name))
    return update_op

  return tf_decorator.make_decorator(update_state_fn, decorated)


def result_wrapper(result_fn):
  """Decorator to wrap metric `result()` function in `merge_call()`.

  Result computation is an idempotent operation that simply calculates the
  metric value using the state variables.

  If metric state variables are distributed across towers/devices and
  `result()` is requested from the context of one device - This function wraps
  `result()` in a distribution strategy `merge_call()`. With this,
  the metric state variables will be aggregated across devices.

  Args:
    result_fn: function that computes the metric result.

  Returns:
    Decorated function that wraps `result_fn()` in distribution strategy
    `merge_call()`.
  """

  def decorated(metric_obj, *args):
    """Decorated function with merge_call."""
    tower_context = distribution_strategy_context.get_tower_context()
    if tower_context is None:  # if in cross tower context already
      result_t = result_fn(*args)
    else:
      # TODO(psv): Test distribution of metrics using different distribution
      # strategies.

      # Creating a wrapper for merge_fn. merge_call invokes the given merge_fn
      # with distribution object as the first parameter. We create a wrapper
      # here so that the result function need not have that parameter.
      def merge_fn_wrapper(distribution, merge_fn, *args):
        # We will get `PerDevice` merge function. Taking the first one as all
        # are identical copies of the function that we had passed below.
        return distribution.unwrap(merge_fn)[0](*args)

      # Wrapping result in merge_call. merge_call is used when we want to leave
      # tower mode and compute a value in cross tower mode.
      result_t = tower_context.merge_call(merge_fn_wrapper, result_fn, *args)
    check_is_tensor_or_operation(result_t,
                                 'Metric {0}\'s result'.format(metric_obj.name))
    return result_t

  return tf_decorator.make_decorator(result_fn, decorated)


def weakmethod(method):
  """Creates a weak reference to the bound method."""

  cls = method.im_class
  func = method.im_func
  instance_ref = weakref.ref(method.im_self)

  @functools.wraps(method)
  def inner(*args, **kwargs):
    return func.__get__(instance_ref(), cls)(*args, **kwargs)

  del method
  return inner


def safe_div(numerator, denominator):
  """Divides two tensors element-wise, returning 0 if the denominator is <= 0.

  Args:
    numerator: A `Tensor`.
    denominator: A `Tensor`, with dtype matching `numerator`.

  Returns:
    0 if `denominator` <= 0, else `numerator` / `denominator`
  """
  t = math_ops.truediv(numerator, denominator)
  zero = array_ops.zeros_like(t, dtype=denominator.dtype)
  condition = math_ops.greater(denominator, zero)
  zero = math_ops.cast(zero, t.dtype)
  return array_ops.where(condition, t, zero)


def squeeze_or_expand_dimensions(y_pred, y_true, sample_weight):
  """Squeeze or expand last dimension if needed.

  1. Squeezes last dim of `y_pred` or `y_true` if their rank differs by 1
  (using `confusion_matrix.remove_squeezable_dimensions`).
  2. Squeezes or expands last dim of `sample_weight` if its rank differs by 1
  from the new rank of `y_pred`.
  If `sample_weight` is scalar, it is kept scalar.

  This will use static shape if available. Otherwise, it will add graph
  operations, which could result in a performance hit.

  Args:
    y_pred: Predicted values, a `Tensor` of arbitrary dimensions.
    y_true: Optional label `Tensor` whose dimensions match `y_pred`.
    sample_weight: Optional weight scalar or `Tensor` whose dimensions match
      `y_pred`.

  Returns:
    Tuple of `y_pred`, `y_true` and `sample_weight`. Each of them possibly has
    the last dimension squeezed,
    `sample_weight` could be extended by one dimension.
  """
  if y_true is not None:
    # squeeze last dim of `y_pred` or `y_true` if their rank differs by 1
    y_true, y_pred = confusion_matrix.remove_squeezable_dimensions(
        y_true, y_pred)

  if sample_weight is None:
    return y_pred, y_true, None

  sample_weight = ops.convert_to_tensor(sample_weight)
  weights_shape = sample_weight.get_shape()
  weights_rank = weights_shape.ndims
  if weights_rank == 0:  # If weights is scalar, do nothing.
    return y_pred, y_true, sample_weight

  y_pred_shape = y_pred.get_shape()
  y_pred_rank = y_pred_shape.ndims
  if (y_pred_rank is not None) and (weights_rank is not None):
    # Use static rank.
    if weights_rank - y_pred_rank == 1:
      sample_weight = array_ops.squeeze(sample_weight, [-1])
    elif y_pred_rank - weights_rank == 1:
      sample_weight = array_ops.expand_dims(sample_weight, [-1])
    return y_pred, y_true, sample_weight

  # Use dynamic rank.
  weights_rank_tensor = array_ops.rank(sample_weight)
  rank_diff = weights_rank_tensor - array_ops.rank(y_pred)
  maybe_squeeze_weights = lambda: array_ops.squeeze(sample_weight, [-1])

  def _maybe_expand_weights():
    return control_flow_ops.cond(
        math_ops.equal(rank_diff,
                       -1), lambda: array_ops.expand_dims(sample_weight, [-1]),
        lambda: sample_weight)

  def _maybe_adjust_weights():
    return control_flow_ops.cond(
        math_ops.equal(rank_diff, 1), maybe_squeeze_weights,
        _maybe_expand_weights)

  # squeeze or expand last dim of `sample_weight` if its rank differs by 1
  # from the new rank of `y_pred`.
  sample_weight = control_flow_ops.cond(
      math_ops.equal(weights_rank_tensor, 0), lambda: sample_weight,
      _maybe_adjust_weights)
  return y_pred, y_true, sample_weight


class Metric(Layer):
  """Encapsulates metric logic and state.

  Usage with eager execution:

  ```python
  m = SomeMetric(...)
  for input in ...:
    m.update_state(input)
  print('Final result: ', m.result().numpy())
  ```

  Usage with graph execution:

  ```python
  m = SomeMetric(...)
  init_op = tf.variables_initializer(m.variables)  # Initialize variables
  with tf.Session() as sess:
    sess.run(init_op)
    for input in ...:
      update_op = m.update_state(input)
      sess.run(update_op)
    print('Final result: ', sess.run(m.result()))
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
  __metaclass__ = ABCMeta

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
        obj.update_state = weakmethod(obj.update_state)
      obj.update_state = weakmethod(
          types.MethodType(update_state_wrapper(obj.update_state), obj))
      result = weakmethod(obj.result)
      obj.result = weakmethod(types.MethodType(result_wrapper(result), obj))
    else:
      obj.update_state = types.MethodType(
          update_state_wrapper(obj.update_state), obj)
      obj.result = types.MethodType(result_wrapper(obj.result), obj)

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
    update_op = self.update_state(*args, **kwargs)  # pylint: disable=not-callable
    with ops.control_dependencies([update_op]):
      return self.result()  # pylint: disable=not-callable

  def reset_states(self):
    """Resets all of the metric state variables.

    This function is called between epochs/steps,
    when a metric is evaluated during training.
    """
    for v in self.variables:
      K.set_value(v, 0)

  @abstractmethod
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

  @abstractmethod
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


class Mean(Metric):
  """Computes the (weighted) mean of the given values.

  This metric creates two variables, `total` and `count` that are used to
  compute the average of `values`. This average is ultimately returned as `mean`
  which is an idempotent operation that simply divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
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

    # Update state variables
    update_total_op = state_ops.assign_add(self.total, values)
    update_count_op = state_ops.assign_add(self.count, num_values)
    return control_flow_ops.group(update_total_op, update_count_op)

  def result(self):
    return safe_div(self.total, self.count)


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
    config = self._fn_kwargs
    base_config = super(MeanMetricWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class BinaryAccuracy(MeanMetricWrapper):
  """Calculates how often predictions matches labels.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `binary accuracy`: an idempotent operation that simply
  divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
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


class CategoricalAccuracy(MeanMetricWrapper):
  """Calculates how often predictions matches labels.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `categorical accuracy`: an idempotent operation that
  simply divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self, name='categorical_accuracy', dtype=None):
    """Creates a `CategoricalAccuracy` instance.

    Args:
      name: (Optional) string name of the metric instance.
      dtype: (Optional) data type of the metric result.
    """
    super(CategoricalAccuracy, self).__init__(
        categorical_accuracy, name, dtype=dtype)


class SparseCategoricalAccuracy(MeanMetricWrapper):
  """Calculates how often predictions matches integer labels.

  This metric creates two local variables, `total` and `count` that are used to
  compute the frequency with which `y_pred` matches `y_true`. This frequency is
  ultimately returned as `sparse categorical accuracy`: an idempotent operation
  that simply divides `total` by `count`.

  If `sample_weight` is `None`, weights default to 1.
  Use `sample_weight` of 0 to mask values.
  """

  def __init__(self, name='sparse_categorical_accuracy', dtype=None):
    super(SparseCategoricalAccuracy, self).__init__(
        sparse_categorical_accuracy, name, dtype=dtype)


@tf_export('keras.metrics.binary_accuracy')
def binary_accuracy(y_true, y_pred, threshold=0.5):
  threshold = math_ops.cast(threshold, y_pred.dtype)
  y_pred = math_ops.cast(y_pred > threshold, y_pred.dtype)
  return K.mean(math_ops.equal(y_true, y_pred), axis=-1)


@tf_export('keras.metrics.categorical_accuracy')
def categorical_accuracy(y_true, y_pred):
  return math_ops.cast(
      math_ops.equal(
          math_ops.argmax(y_true, axis=-1), math_ops.argmax(y_pred, axis=-1)),
      K.floatx())


@tf_export('keras.metrics.sparse_categorical_accuracy')
def sparse_categorical_accuracy(y_true, y_pred):
  y_true = math_ops.reduce_max(y_true, axis=-1)
  y_pred = math_ops.argmax(y_pred, axis=-1)

  # If the expected labels are float, we need to cast the int returned by
  # argmax to compare.
  if K.dtype(y_true) == K.floatx():
    y_pred = math_ops.cast(y_pred, K.floatx())

  return math_ops.cast(math_ops.equal(y_true, y_pred), K.floatx())


@tf_export('keras.metrics.top_k_categorical_accuracy')
def top_k_categorical_accuracy(y_true, y_pred, k=5):
  return K.mean(
      nn.in_top_k(y_pred, math_ops.argmax(y_true, axis=-1), k), axis=-1)


@tf_export('keras.metrics.sparse_top_k_categorical_accuracy')
def sparse_top_k_categorical_accuracy(y_true, y_pred, k=5):
  return K.mean(
      nn.in_top_k(y_pred,
                  math_ops.cast(math_ops.reduce_max(y_true, axis=-1), 'int32'),
                  k),
      axis=-1)

# Aliases

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
cosine = cosine_proximity


@tf_export('keras.metrics.serialize')
def serialize(metric):
  return serialize_keras_object(metric)


@tf_export('keras.metrics.deserialize')
def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='metric function')


@tf_export('keras.metrics.get')
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
