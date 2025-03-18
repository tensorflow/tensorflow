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
# pylint: disable=g-classes-have-attributes
"""Built-in loss functions."""

import abc
import functools

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.tools.docs import doc_controls


class Loss:
  """Loss base class.

  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

  Example subclass implementation:

  ```python
  class MeanSquaredError(Loss):

    def call(self, y_true, y_pred):
      y_pred = tf.convert_to_tensor_v2(y_pred)
      y_true = tf.cast(y_true, y_pred.dtype)
      return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
  ```

  When used with `tf.distribute.Strategy`, outside of built-in training loops
  such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
  types, and reduce losses explicitly in your training loop. Using 'AUTO' or
  'SUM_OVER_BATCH_SIZE' will raise an error.

  Please see this custom training [tutorial](
    https://www.tensorflow.org/tutorials/distribute/custom_training) for more
  details on this.

  You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:

  ```python
  with strategy.scope():
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ....
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
            (1. / global_batch_size))
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
    """Initializes `Loss` class.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance.
    """
    losses_utils.ReductionV2.validate(reduction)
    self.reduction = reduction
    self.name = name
    # SUM_OVER_BATCH is only allowed in losses managed by `fit` or
    # CannedEstimators.
    self._allow_sum_over_batch_size = False
    self._set_name_scope()

  def _set_name_scope(self):
    """Creates a valid `name_scope` name."""
    if self.name is None:
      self._name_scope = self.__class__.__name__
    elif self.name == '<lambda>':
      self._name_scope = 'lambda'
    else:
      # E.g. '_my_loss' => 'my_loss'
      self._name_scope = self.name.strip('_')

  def __call__(self, y_true, y_pred, sample_weight=None):
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
      sample_weight: Optional `sample_weight` acts as a coefficient for the
        loss. If a scalar is provided, then the loss is simply scaled by the
        given value. If `sample_weight` is a tensor of size `[batch_size]`, then
        the total loss for each sample of the batch is rescaled by the
        corresponding element in the `sample_weight` vector. If the shape of
        `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to
        this shape), then each loss element of `y_pred` is scaled
        by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
          functions reduce by 1 dimension, usually axis=-1.)

    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
        shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
        because all loss functions reduce by 1 dimension, usually axis=-1.)

    Raises:
      ValueError: If the shape of `sample_weight` is invalid.
    """
    # If we are wrapping a lambda function strip '<>' from the name as it is not
    # accepted in scope name.
    graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
        y_true, y_pred, sample_weight)
    with backend.name_scope(self._name_scope), graph_ctx:
      if context.executing_eagerly():
        call_fn = self.call
      else:
        call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
      losses = call_fn(y_true, y_pred)
      return losses_utils.compute_weighted_loss(
          losses, sample_weight, reduction=self._get_reduction())

  @classmethod
  def from_config(cls, config):
    """Instantiates a `Loss` from its config (output of `get_config()`).

    Args:
        config: Output of `get_config()`.

    Returns:
        A `Loss` instance.
    """
    return cls(**config)

  def get_config(self):
    """Returns the config dictionary for a `Loss` instance."""
    return {'reduction': self.reduction, 'name': self.name}

  @abc.abstractmethod
  @doc_controls.for_subclass_implementers
  def call(self, y_true, y_pred):
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`

    Returns:
      Loss values with the shape `[batch_size, d0, .. dN-1]`.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  def _get_reduction(self):
    """Handles `AUTO` reduction cases and returns the reduction value."""
    if (not self._allow_sum_over_batch_size and
        distribute_lib.has_strategy() and
        (self.reduction == losses_utils.ReductionV2.AUTO or
         self.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)):
      raise ValueError(
          'Please use `tf.keras.losses.Reduction.SUM` or '
          '`tf.keras.losses.Reduction.NONE` for loss reduction when losses are '
          'used with `tf.distribute.Strategy` outside of the built-in training '
          'loops. You can implement '
          '`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch '
          'size like:\n```\nwith strategy.scope():\n'
          '    loss_obj = tf.keras.losses.CategoricalCrossentropy('
          'reduction=tf.keras.losses.Reduction.NONE)\n....\n'
          '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * '
          '(1. / global_batch_size)\n```\nPlease see '
          'https://www.tensorflow.org/tutorials/distribute/custom_training'
          ' for more details.')

    if self.reduction == losses_utils.ReductionV2.AUTO:
      return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
    return self.reduction


class LossFunctionWrapper(Loss):
  """Wraps a loss function in the `Loss` class."""

  def __init__(self,
               fn,
               reduction=losses_utils.ReductionV2.AUTO,
               name=None,
               **kwargs):
    """Initializes `LossFunctionWrapper` class.

    Args:
      fn: The loss function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super().__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    """Invokes the `LossFunctionWrapper` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Loss values per sample.
    """
    if tensor_util.is_tf_type(y_pred) and tensor_util.is_tf_type(y_true):
      y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

    ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
    return ag_fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in self._fn_kwargs.items():
      config[k] = backend.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MeanSquaredError(LossFunctionWrapper):
  """Computes the mean of squares of errors between labels and predictions.

  `loss = square(y_true - y_pred)`

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> mse = tf.keras.losses.MeanSquaredError()
  >>> mse(y_true, y_pred).numpy()
  0.5

  >>> # Calling with 'sample_weight'.
  >>> mse(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  0.25

  >>> # Using 'sum' reduction type.
  >>> mse = tf.keras.losses.MeanSquaredError(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> mse(y_true, y_pred).numpy()
  1.0

  >>> # Using 'none' reduction type.
  >>> mse = tf.keras.losses.MeanSquaredError(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> mse(y_true, y_pred).numpy()
  array([0.5, 0.5], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.MeanSquaredError())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_squared_error'):
    """Initializes `MeanSquaredError` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'mean_squared_error'.
    """
    super().__init__(mean_squared_error, name=name, reduction=reduction)


class MeanAbsoluteError(LossFunctionWrapper):
  """Computes the mean of absolute difference between labels and predictions.

  `loss = abs(y_true - y_pred)`

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> mae = tf.keras.losses.MeanAbsoluteError()
  >>> mae(y_true, y_pred).numpy()
  0.5

  >>> # Calling with 'sample_weight'.
  >>> mae(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  0.25

  >>> # Using 'sum' reduction type.
  >>> mae = tf.keras.losses.MeanAbsoluteError(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> mae(y_true, y_pred).numpy()
  1.0

  >>> # Using 'none' reduction type.
  >>> mae = tf.keras.losses.MeanAbsoluteError(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> mae(y_true, y_pred).numpy()
  array([0.5, 0.5], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.MeanAbsoluteError())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_absolute_error'):
    """Initializes `MeanAbsoluteError` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to 'mean_absolute_error'.
    """
    super().__init__(mean_absolute_error, name=name, reduction=reduction)


class MeanAbsolutePercentageError(LossFunctionWrapper):
  """Computes the mean absolute percentage error between `y_true` and `y_pred`.

  `loss = 100 * abs(y_true - y_pred) / y_true`

  Standalone usage:

  >>> y_true = [[2., 1.], [2., 3.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> mape = tf.keras.losses.MeanAbsolutePercentageError()
  >>> mape(y_true, y_pred).numpy()
  50.

  >>> # Calling with 'sample_weight'.
  >>> mape(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  20.

  >>> # Using 'sum' reduction type.
  >>> mape = tf.keras.losses.MeanAbsolutePercentageError(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> mape(y_true, y_pred).numpy()
  100.

  >>> # Using 'none' reduction type.
  >>> mape = tf.keras.losses.MeanAbsolutePercentageError(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> mape(y_true, y_pred).numpy()
  array([25., 75.], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.MeanAbsolutePercentageError())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_absolute_percentage_error'):
    """Initializes `MeanAbsolutePercentageError` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to
        'mean_absolute_percentage_error'.
    """
    super().__init__(
        mean_absolute_percentage_error, name=name, reduction=reduction)


class MeanSquaredLogarithmicError(LossFunctionWrapper):
  """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

  `loss = square(log(y_true + 1.) - log(y_pred + 1.))`

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [1., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError()
  >>> msle(y_true, y_pred).numpy()
  0.240

  >>> # Calling with 'sample_weight'.
  >>> msle(y_true, y_pred, sample_weight=[0.7, 0.3]).numpy()
  0.120

  >>> # Using 'sum' reduction type.
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> msle(y_true, y_pred).numpy()
  0.480

  >>> # Using 'none' reduction type.
  >>> msle = tf.keras.losses.MeanSquaredLogarithmicError(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> msle(y_true, y_pred).numpy()
  array([0.240, 0.240], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.MeanSquaredLogarithmicError())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='mean_squared_logarithmic_error'):
    """Initializes `MeanSquaredLogarithmicError` instance.

    Args:
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to
        'mean_squared_logarithmic_error'.
    """
    super().__init__(
        mean_squared_logarithmic_error, name=name, reduction=reduction)


class BinaryCrossentropy(LossFunctionWrapper):
  """Computes the cross-entropy loss between true labels and predicted labels.

  Use this cross-entropy loss for binary (0 or 1) classification applications.
  The loss function requires the following inputs:

  - `y_true` (true label): This is either 0 or 1.
  - `y_pred` (predicted value): This is the model's prediction, i.e, a single
    floating-point value which either represents a
    [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
    when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
    `from_logits=False`).

  **Recommended Usage:** (set `from_logits=True`)

  With `tf.keras` API:

  ```python
  model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    ....
  )
  ```

  As a standalone function:

  >>> # Example 1: (batch_size = 1, number of samples = 4)
  >>> y_true = [0, 1, 0, 0]
  >>> y_pred = [-18.6, 0.51, 2.94, -12.8]
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  >>> bce(y_true, y_pred).numpy()
  0.865

  >>> # Example 2: (batch_size = 2, number of samples = 4)
  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
  >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
  >>> bce(y_true, y_pred).numpy()
  0.865
  >>> # Using 'sample_weight' attribute
  >>> bce(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.243
  >>> # Using 'sum' reduction` type.
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> bce(y_true, y_pred).numpy()
  1.730
  >>> # Using 'none' reduction type.
  >>> bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> bce(y_true, y_pred).numpy()
  array([0.235, 1.496], dtype=float32)

  **Default Usage:** (set `from_logits=False`)

  >>> # Make the following updates to the above "Recommended Usage" section
  >>> # 1. Set `from_logits=False`
  >>> tf.keras.losses.BinaryCrossentropy() # OR ...('from_logits=False')
  >>> # 2. Update `y_pred` to use probabilities instead of logits
  >>> y_pred = [0.6, 0.3, 0.2, 0.8] # OR [[0.6, 0.3], [0.2, 0.8]]
  """

  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='binary_crossentropy'):
    """Initializes `BinaryCrossentropy` instance.

    Args:
      from_logits: Whether to interpret `y_pred` as a tensor of
        [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
          assume that `y_pred` contains probabilities (i.e., values in [0, 1]).
      label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0,
        we compute the loss between the predicted labels and a smoothed version
        of the true labels, where the smoothing squeezes the labels towards 0.5.
        Larger values of `label_smoothing` correspond to heavier smoothing.
      axis: The axis along which to compute crossentropy (the features axis).
        Defaults to -1.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Name for the op. Defaults to 'binary_crossentropy'.
    """
    super().__init__(
        binary_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        axis=axis)
    self.from_logits = from_logits


class CategoricalCrossentropy(LossFunctionWrapper):
  """Computes the crossentropy loss between the labels and predictions.

  Use this crossentropy loss function when there are two or more label classes.
  We expect labels to be provided in a `one_hot` representation. If you want to
  provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
  There should be `# classes` floating point values per feature.

  In the snippet below, there is `# classes` floating pointing values per
  example. The shape of both `y_pred` and `y_true` are
  `[batch_size, num_classes]`.

  Standalone usage:

  >>> y_true = [[0, 1, 0], [0, 0, 1]]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> cce = tf.keras.losses.CategoricalCrossentropy()
  >>> cce(y_true, y_pred).numpy()
  1.177

  >>> # Calling with 'sample_weight'.
  >>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
  0.814

  >>> # Using 'sum' reduction type.
  >>> cce = tf.keras.losses.CategoricalCrossentropy(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> cce(y_true, y_pred).numpy()
  2.354

  >>> # Using 'none' reduction type.
  >>> cce = tf.keras.losses.CategoricalCrossentropy(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> cce(y_true, y_pred).numpy()
  array([0.0513, 2.303], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalCrossentropy())
  ```
  """

  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='categorical_crossentropy'):
    """Initializes `CategoricalCrossentropy` instance.

    Args:
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
      label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
        meaning the confidence on label values are relaxed. For example, if
        `0.1`, use `0.1 / num_classes` for non-target labels and
        `0.9 + 0.1 / num_classes` for target labels.
      axis: The axis along which to compute crossentropy (the features axis).
        Defaults to -1.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance.
        Defaults to 'categorical_crossentropy'.
    """
    super().__init__(
        categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        axis=axis)


class SparseCategoricalCrossentropy(LossFunctionWrapper):
  """Computes the crossentropy loss between the labels and predictions.

  Use this crossentropy loss function when there are two or more label classes.
  We expect labels to be provided as integers. If you want to provide labels
  using `one-hot` representation, please use `CategoricalCrossentropy` loss.
  There should be `# classes` floating point values per feature for `y_pred`
  and a single floating point value per feature for `y_true`.

  In the snippet below, there is a single floating point value per example for
  `y_true` and `# classes` floating pointing values per example for `y_pred`.
  The shape of `y_true` is `[batch_size]` and the shape of `y_pred` is
  `[batch_size, num_classes]`.

  Standalone usage:

  >>> y_true = [1, 2]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> scce = tf.keras.losses.SparseCategoricalCrossentropy()
  >>> scce(y_true, y_pred).numpy()
  1.177

  >>> # Calling with 'sample_weight'.
  >>> scce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
  0.814

  >>> # Using 'sum' reduction type.
  >>> scce = tf.keras.losses.SparseCategoricalCrossentropy(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> scce(y_true, y_pred).numpy()
  2.354

  >>> # Using 'none' reduction type.
  >>> scce = tf.keras.losses.SparseCategoricalCrossentropy(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> scce(y_true, y_pred).numpy()
  array([0.0513, 2.303], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd',
                loss=tf.keras.losses.SparseCategoricalCrossentropy())
  ```
  """

  def __init__(self,
               from_logits=False,
               reduction=losses_utils.ReductionV2.AUTO,
               name='sparse_categorical_crossentropy'):
    """Initializes `SparseCategoricalCrossentropy` instance.

    Args:
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
      reduction: Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the instance. Defaults to
        'sparse_categorical_crossentropy'.
    """
    if reduction not in {'auto', 'none', 'sum', 'sum_over_batch_size'}:
    raise ValueError(
        f"Invalid Reduction Key: {reduction}. Expected keys are "
        "('auto', 'none', 'sum', 'sum_over_batch_size')"
    )
    super().__init__(
        sparse_categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits)
