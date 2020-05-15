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
"""Built-in loss functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.losses import util as tf_losses_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


@keras_export('keras.losses.Loss')
class Loss(object):
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
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op.
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
      sample_weight: Optional `sample_weight` acts as a
        coefficient for the loss. If a scalar is provided, then the loss is
        simply scaled by the given value. If `sample_weight` is a tensor of size
        `[batch_size]`, then the total loss for each sample of the batch is
        rescaled by the corresponding element in the `sample_weight` vector. If
        the shape of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
        broadcasted to this shape), then each loss element of `y_pred` is scaled
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
    with K.name_scope(self._name_scope), graph_ctx:
      ag_call = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
      losses = ag_call(y_true, y_pred)
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
    NotImplementedError('Must be implemented in subclasses.')

  def _get_reduction(self):
    """Handles `AUTO` reduction cases and returns the reduction value."""
    if (not self._allow_sum_over_batch_size and
        distribution_strategy_context.has_strategy() and
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
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: (Optional) name for the loss.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
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
    if tensor_util.is_tensor(y_pred) and tensor_util.is_tensor(y_true):
      y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(
          y_pred, y_true)
    ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
    return ag_fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in six.iteritems(self._fn_kwargs):
      config[k] = K.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    base_config = super(LossFunctionWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.losses.MeanSquaredError')
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
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'mean_squared_error'.
    """
    super(MeanSquaredError, self).__init__(
        mean_squared_error, name=name, reduction=reduction)


@keras_export('keras.losses.MeanAbsoluteError')
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
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'mean_absolute_error'.
    """
    super(MeanAbsoluteError, self).__init__(
        mean_absolute_error, name=name, reduction=reduction)


@keras_export('keras.losses.MeanAbsolutePercentageError')
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
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to
        'mean_absolute_percentage_error'.
    """
    super(MeanAbsolutePercentageError, self).__init__(
        mean_absolute_percentage_error, name=name, reduction=reduction)


@keras_export('keras.losses.MeanSquaredLogarithmicError')
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
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to
        'mean_squared_logarithmic_error'.
    """
    super(MeanSquaredLogarithmicError, self).__init__(
        mean_squared_logarithmic_error, name=name, reduction=reduction)


@keras_export('keras.losses.BinaryCrossentropy')
class BinaryCrossentropy(LossFunctionWrapper):
  """Computes the cross-entropy loss between true labels and predicted labels.

  Use this cross-entropy loss when there are only two label classes (assumed to
  be 0 and 1). For each example, there should be a single floating-point value
  per prediction.

  In the snippet below, each of the four examples has only a single
  floating-pointing value, and both `y_pred` and `y_true` have the shape
  `[batch_size]`.

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> bce = tf.keras.losses.BinaryCrossentropy()
  >>> bce(y_true, y_pred).numpy()
  0.815

  >>> # Calling with 'sample_weight'.
  >>> bce(y_true, y_pred, sample_weight=[1, 0]).numpy()
  0.458

   >>> # Using 'sum' reduction type.
  >>> bce = tf.keras.losses.BinaryCrossentropy(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> bce(y_true, y_pred).numpy()
  1.630

  >>> # Using 'none' reduction type.
  >>> bce = tf.keras.losses.BinaryCrossentropy(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> bce(y_true, y_pred).numpy()
  array([0.916 , 0.714], dtype=float32)

  Usage with the `tf.keras` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.BinaryCrossentropy())
  ```
  """

  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.ReductionV2.AUTO,
               name='binary_crossentropy'):
    """Initializes `BinaryCrossentropy` instance.

    Args:
      from_logits: Whether to interpret `y_pred` as a tensor of
        [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
          assume that `y_pred` contains probabilities (i.e., values in [0, 1]).
          **Note - Using from_logits=True may be more numerically stable.
      label_smoothing: Float in [0, 1]. When 0, no smoothing occurs. When > 0,
        we compute the loss between the predicted labels and a smoothed version
        of the true labels, where the smoothing squeezes the labels towards 0.5.
        Larger values of `label_smoothing` correspond to heavier smoothing.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: (Optional) Name for the op. Defaults to 'binary_crossentropy'.
    """
    super(BinaryCrossentropy, self).__init__(
        binary_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing)
    self.from_logits = from_logits


@keras_export('keras.losses.CategoricalCrossentropy')
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
               reduction=losses_utils.ReductionV2.AUTO,
               name='categorical_crossentropy'):
    """Initializes `CategoricalCrossentropy` instance.

    Args:
      from_logits: Whether `y_pred` is expected to be a logits tensor. By
        default, we assume that `y_pred` encodes a probability distribution.
        **Note - Using from_logits=True is more numerically stable.**
      label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
        meaning the confidence on label values are relaxed. e.g.
        `label_smoothing=0.2` means that we will use a value of `0.1` for label
        `0` and `0.9` for label `1`"
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'categorical_crossentropy'.
    """
    super(CategoricalCrossentropy, self).__init__(
        categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing)


@keras_export('keras.losses.SparseCategoricalCrossentropy')
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
        **Note - Using from_logits=True may be more numerically stable.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to
        'sparse_categorical_crossentropy'.
    """
    super(SparseCategoricalCrossentropy, self).__init__(
        sparse_categorical_crossentropy,
        name=name,
        reduction=reduction,
        from_logits=from_logits)


@keras_export('keras.losses.Hinge')
class Hinge(LossFunctionWrapper):
  """Computes the hinge loss between `y_true` and `y_pred`.

  `loss = maximum(1 - y_true * y_pred, 0)`

  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
  provided we will convert them to -1 or 1.

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> h = tf.keras.losses.Hinge()
  >>> h(y_true, y_pred).numpy()
  1.3

  >>> # Calling with 'sample_weight'.
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
  0.55

  >>> # Using 'sum' reduction type.
  >>> h = tf.keras.losses.Hinge(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> h(y_true, y_pred).numpy()
  2.6

  >>> # Using 'none' reduction type.
  >>> h = tf.keras.losses.Hinge(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> h(y_true, y_pred).numpy()
  array([1.1, 1.5], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.Hinge())
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='hinge'):
    """Initializes `Hinge` instance.

    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'hinge'.
    """
    super(Hinge, self).__init__(hinge, name=name, reduction=reduction)


@keras_export('keras.losses.SquaredHinge')
class SquaredHinge(LossFunctionWrapper):
  """Computes the squared hinge loss between `y_true` and `y_pred`.

  `loss = square(maximum(1 - y_true * y_pred, 0))`

  `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
  provided we will convert them to -1 or 1.

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> h = tf.keras.losses.SquaredHinge()
  >>> h(y_true, y_pred).numpy()
  1.86

  >>> # Calling with 'sample_weight'.
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
  0.73

  >>> # Using 'sum' reduction type.
  >>> h = tf.keras.losses.SquaredHinge(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> h(y_true, y_pred).numpy()
  3.72

  >>> # Using 'none' reduction type.
  >>> h = tf.keras.losses.SquaredHinge(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> h(y_true, y_pred).numpy()
  array([1.46, 2.26], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.SquaredHinge())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='squared_hinge'):
    """Initializes `SquaredHinge` instance.

    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'squared_hinge'.
    """
    super(SquaredHinge, self).__init__(
        squared_hinge, name=name, reduction=reduction)


@keras_export('keras.losses.CategoricalHinge')
class CategoricalHinge(LossFunctionWrapper):
  """Computes the categorical hinge loss between `y_true` and `y_pred`.

  `loss = maximum(neg - pos + 1, 0)`
  where `neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)`

  Standalone usage:

  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> h = tf.keras.losses.CategoricalHinge()
  >>> h(y_true, y_pred).numpy()
  1.4

  >>> # Calling with 'sample_weight'.
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
  0.6

  >>> # Using 'sum' reduction type.
  >>> h = tf.keras.losses.CategoricalHinge(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> h(y_true, y_pred).numpy()
  2.8

  >>> # Using 'none' reduction type.
  >>> h = tf.keras.losses.CategoricalHinge(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> h(y_true, y_pred).numpy()
  array([1.2, 1.6], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.CategoricalHinge())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='categorical_hinge'):
    """Initializes `CategoricalHinge` instance.

    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'categorical_hinge'.
    """
    super(CategoricalHinge, self).__init__(
        categorical_hinge, name=name, reduction=reduction)


@keras_export('keras.losses.Poisson')
class Poisson(LossFunctionWrapper):
  """Computes the Poisson loss between `y_true` and `y_pred`.

  `loss = y_pred - y_true * log(y_pred)`

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [0., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> p = tf.keras.losses.Poisson()
  >>> p(y_true, y_pred).numpy()
  0.5

  >>> # Calling with 'sample_weight'.
  >>> p(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.4

  >>> # Using 'sum' reduction type.
  >>> p = tf.keras.losses.Poisson(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> p(y_true, y_pred).numpy()
  0.999

  >>> # Using 'none' reduction type.
  >>> p = tf.keras.losses.Poisson(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> p(y_true, y_pred).numpy()
  array([0.999, 0.], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.Poisson())
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='poisson'):
    """Initializes `Poisson` instance.

    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'poisson'.
    """
    super(Poisson, self).__init__(poisson, name=name, reduction=reduction)


@keras_export('keras.losses.LogCosh')
class LogCosh(LossFunctionWrapper):
  """Computes the logarithm of the hyperbolic cosine of the prediction error.

  `logcosh = log((exp(x) + exp(-x))/2)`,
  where x is the error `y_pred - y_true`.

  Standalone usage:

  >>> y_true = [[0., 1.], [0., 0.]]
  >>> y_pred = [[1., 1.], [0., 0.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> l = tf.keras.losses.LogCosh()
  >>> l(y_true, y_pred).numpy()
  0.108

  >>> # Calling with 'sample_weight'.
  >>> l(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.087

  >>> # Using 'sum' reduction type.
  >>> l = tf.keras.losses.LogCosh(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> l(y_true, y_pred).numpy()
  0.217

  >>> # Using 'none' reduction type.
  >>> l = tf.keras.losses.LogCosh(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> l(y_true, y_pred).numpy()
  array([0.217, 0.], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.LogCosh())
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name='log_cosh'):
    """Initializes `LogCosh` instance.

    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'log_cosh'.
    """
    super(LogCosh, self).__init__(log_cosh, name=name, reduction=reduction)


@keras_export('keras.losses.KLDivergence')
class KLDivergence(LossFunctionWrapper):
  """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

  `loss = y_true * log(y_true / y_pred)`

  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

  Standalone usage:

  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> kl = tf.keras.losses.KLDivergence()
  >>> kl(y_true, y_pred).numpy()
  0.458

  >>> # Calling with 'sample_weight'.
  >>> kl(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.366

  >>> # Using 'sum' reduction type.
  >>> kl = tf.keras.losses.KLDivergence(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> kl(y_true, y_pred).numpy()
  0.916

  >>> # Using 'none' reduction type.
  >>> kl = tf.keras.losses.KLDivergence(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> kl(y_true, y_pred).numpy()
  array([0.916, -3.08e-06], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.KLDivergence())
  ```
  """

  def __init__(self,
               reduction=losses_utils.ReductionV2.AUTO,
               name='kl_divergence'):
    """Initializes `KLDivergence` instance.

    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'kl_divergence'.
    """
    super(KLDivergence, self).__init__(
        kl_divergence, name=name, reduction=reduction)


@keras_export('keras.losses.Huber')
class Huber(LossFunctionWrapper):
  """Computes the Huber loss between `y_true` and `y_pred`.

  For each value x in `error = y_true - y_pred`:

  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Standalone usage:

  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> h = tf.keras.losses.Huber()
  >>> h(y_true, y_pred).numpy()
  0.155

  >>> # Calling with 'sample_weight'.
  >>> h(y_true, y_pred, sample_weight=[1, 0]).numpy()
  0.09

  >>> # Using 'sum' reduction type.
  >>> h = tf.keras.losses.Huber(
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> h(y_true, y_pred).numpy()
  0.31

  >>> # Using 'none' reduction type.
  >>> h = tf.keras.losses.Huber(
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> h(y_true, y_pred).numpy()
  array([0.18, 0.13], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.Huber())
  ```
  """

  def __init__(self,
               delta=1.0,
               reduction=losses_utils.ReductionV2.AUTO,
               name='huber_loss'):
    """Initializes `Huber` instance.

    Args:
      delta: A float, the point where the Huber loss function changes from a
        quadratic to linear.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training)
        for more details.
      name: Optional name for the op. Defaults to 'huber_loss'.
    """
    super(Huber, self).__init__(
        huber, name=name, reduction=reduction, delta=delta)


@keras_export('keras.metrics.mean_squared_error',
              'keras.metrics.mse',
              'keras.metrics.MSE',
              'keras.losses.mean_squared_error',
              'keras.losses.mse',
              'keras.losses.MSE')
@dispatch.add_dispatch_support
def mean_squared_error(y_true, y_pred):
  """Computes the mean squared error between labels and predictions.

  After computing the squared distance between the inputs, the mean value over
  the last dimension is returned.

  `loss = mean(square(y_true - y_pred), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(), np.mean(np.square(y_true - y_pred), axis=-1))

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)


@keras_export('keras.metrics.mean_absolute_error',
              'keras.metrics.mae',
              'keras.metrics.MAE',
              'keras.losses.mean_absolute_error',
              'keras.losses.mae',
              'keras.losses.MAE')
@dispatch.add_dispatch_support
def mean_absolute_error(y_true, y_pred):
  """Computes the mean absolute error between labels and predictions.

  `loss = mean(abs(y_true - y_pred), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.mean_absolute_error(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(), np.mean(np.abs(y_true - y_pred), axis=-1))

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean absolute error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.mean(math_ops.abs(y_pred - y_true), axis=-1)


@keras_export('keras.metrics.mean_absolute_percentage_error',
              'keras.metrics.mape',
              'keras.metrics.MAPE',
              'keras.losses.mean_absolute_percentage_error',
              'keras.losses.mape',
              'keras.losses.MAPE')
@dispatch.add_dispatch_support
def mean_absolute_percentage_error(y_true, y_pred):
  """Computes the mean absolute percentage error between `y_true` and `y_pred`.

  `loss = 100 * mean(abs(y_true - y_pred) / y_true, axis=-1)`

  Standalone usage:

  >>> y_true = np.random.random(size=(2, 3))
  >>> y_true = np.maximum(y_true, 1e-7)  # Prevent division by zero
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(),
  ...     100. * np.mean(np.abs((y_true - y_pred) / y_true), axis=-1))

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean absolute percentage error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  diff = math_ops.abs(
      (y_true - y_pred) / K.maximum(math_ops.abs(y_true), K.epsilon()))
  return 100. * K.mean(diff, axis=-1)


@keras_export('keras.metrics.mean_squared_logarithmic_error',
              'keras.metrics.msle',
              'keras.metrics.MSLE',
              'keras.losses.mean_squared_logarithmic_error',
              'keras.losses.msle',
              'keras.losses.MSLE')
@dispatch.add_dispatch_support
def mean_squared_logarithmic_error(y_true, y_pred):
  """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

  `loss = mean(square(log(y_true + 1) - log(y_pred + 1)), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.mean_squared_logarithmic_error(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> y_true = np.maximum(y_true, 1e-7)
  >>> y_pred = np.maximum(y_pred, 1e-7)
  >>> assert np.array_equal(
  ...     loss.numpy(),
  ...     np.mean(
  ...         np.square(np.log(y_true + 1.) - np.log(y_pred + 1.)), axis=-1))

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Mean squared logarithmic error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  first_log = math_ops.log(K.maximum(y_pred, K.epsilon()) + 1.)
  second_log = math_ops.log(K.maximum(y_true, K.epsilon()) + 1.)
  return K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)


def _maybe_convert_labels(y_true):
  """Converts binary labels into -1/1."""
  are_zeros = math_ops.equal(y_true, 0)
  are_ones = math_ops.equal(y_true, 1)
  is_binary = math_ops.reduce_all(math_ops.logical_or(are_zeros, are_ones))

  def _convert_binary_labels():
    # Convert the binary labels to -1 or 1.
    return 2. * y_true - 1.

  updated_y_true = smart_cond.smart_cond(is_binary,
                                         _convert_binary_labels, lambda: y_true)
  return updated_y_true


@keras_export('keras.metrics.squared_hinge', 'keras.losses.squared_hinge')
@dispatch.add_dispatch_support
def squared_hinge(y_true, y_pred):
  """Computes the squared hinge loss between `y_true` and `y_pred`.

  `loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.choice([-1, 1], size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.squared_hinge(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(),
  ...     np.mean(np.square(np.maximum(1. - y_true * y_pred, 0.)), axis=-1))

  Args:
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
      If binary (0 or 1) labels are provided we will convert them to -1 or 1.
      shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
     Squared hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = _maybe_convert_labels(y_true)
  return K.mean(
      math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)


@keras_export('keras.metrics.hinge', 'keras.losses.hinge')
@dispatch.add_dispatch_support
def hinge(y_true, y_pred):
  """Computes the hinge loss between `y_true` and `y_pred`.

  `loss = mean(maximum(1 - y_true * y_pred, 0), axis=-1)`

  Standalone usage:

  >>> y_true = np.random.choice([-1, 1], size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.hinge(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> assert np.array_equal(
  ...     loss.numpy(),
  ...     np.mean(np.maximum(1. - y_true * y_pred, 0.), axis=-1))

  Args:
    y_true: The ground truth values. `y_true` values are expected to be -1 or 1.
      If binary (0 or 1) labels are provided they will be converted to -1 or 1.
      shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Hinge loss values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = _maybe_convert_labels(y_true)
  return K.mean(math_ops.maximum(1. - y_true * y_pred, 0.), axis=-1)


@keras_export('keras.losses.categorical_hinge')
@dispatch.add_dispatch_support
def categorical_hinge(y_true, y_pred):
  """Computes the categorical hinge loss between `y_true` and `y_pred`.

  `loss = maximum(neg - pos + 1, 0)`
  where `neg=maximum((1-y_true)*y_pred) and pos=sum(y_true*y_pred)`

  Standalone usage:

  >>> y_true = np.random.randint(0, 3, size=(2,))
  >>> y_true = tf.keras.utils.to_categorical(y_true, num_classes=3)
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.categorical_hinge(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> pos = np.sum(y_true * y_pred, axis=-1)
  >>> neg = np.amax((1. - y_true) * y_pred, axis=-1)
  >>> assert np.array_equal(loss.numpy(), np.maximum(0., neg - pos + 1.))

  Args:
    y_true: The ground truth values. `y_true` values are expected to be 0 or 1.
    y_pred: The predicted values.

  Returns:
    Categorical hinge loss values.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  pos = math_ops.reduce_sum(y_true * y_pred, axis=-1)
  neg = math_ops.reduce_max((1. - y_true) * y_pred, axis=-1)
  return math_ops.maximum(0., neg - pos + 1.)


@keras_export('keras.losses.huber', v1=[])
@dispatch.add_dispatch_support
def huber(y_true, y_pred, delta=1.0):
  """Computes Huber loss value.

  For each value x in `error = y_true - y_pred`:

  ```
  loss = 0.5 * x^2                  if |x| <= d
  loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    delta: A float, the point where the Huber loss function changes from a
      quadratic to linear.

  Returns:
    Tensor with one scalar loss entry per sample.
  """
  y_pred = math_ops.cast(y_pred, dtype=K.floatx())
  y_true = math_ops.cast(y_true, dtype=K.floatx())
  delta = math_ops.cast(delta, dtype=K.floatx())
  error = math_ops.subtract(y_pred, y_true)
  abs_error = math_ops.abs(error)
  quadratic = math_ops.minimum(abs_error, delta)
  linear = math_ops.subtract(abs_error, quadratic)
  return K.mean(
      math_ops.add(
          math_ops.multiply(
              ops.convert_to_tensor_v2(0.5, dtype=quadratic.dtype),
              math_ops.multiply(quadratic, quadratic)),
          math_ops.multiply(delta, linear)),
      axis=-1)


@keras_export('keras.losses.log_cosh', 'keras.losses.logcosh')
@dispatch.add_dispatch_support
def log_cosh(y_true, y_pred):
  """Logarithm of the hyperbolic cosine of the prediction error.

  `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
  to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
  like the mean squared error, but will not be so strongly affected by the
  occasional wildly incorrect prediction.

  Standalone usage:

  >>> y_true = np.random.random(size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.logcosh(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> x = y_pred - y_true
  >>> assert np.allclose(
  ...     loss.numpy(),
  ...     np.mean(x + np.log(np.exp(-2. * x) + 1.) - math_ops.log(2.), axis=-1),
  ...     atol=1e-5)

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
    Logcosh error values. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)

  def _logcosh(x):
    return x + nn.softplus(-2. * x) - math_ops.cast(math_ops.log(2.), x.dtype)

  return K.mean(_logcosh(y_pred - y_true), axis=-1)


@keras_export('keras.metrics.categorical_crossentropy',
              'keras.losses.categorical_crossentropy')
@dispatch.add_dispatch_support
def categorical_crossentropy(y_true,
                             y_pred,
                             from_logits=False,
                             label_smoothing=0):
  """Computes the categorical crossentropy loss.

  Standalone usage:

  >>> y_true = [[0, 1, 0], [0, 0, 1]]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> loss.numpy()
  array([0.0513, 2.303], dtype=float32)

  Args:
    y_true: Tensor of one-hot true targets.
    y_pred: Tensor of predicted targets.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.

  Returns:
    Categorical crossentropy loss value.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  label_smoothing = ops.convert_to_tensor_v2(label_smoothing, dtype=K.floatx())

  def _smooth_labels():
    num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

  y_true = smart_cond.smart_cond(label_smoothing,
                                 _smooth_labels, lambda: y_true)
  return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


@keras_export('keras.metrics.sparse_categorical_crossentropy',
              'keras.losses.sparse_categorical_crossentropy')
@dispatch.add_dispatch_support
def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
  """Computes the sparse categorical crossentropy loss.

  Standalone usage:

  >>> y_true = [1, 2]
  >>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
  >>> loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> loss.numpy()
  array([0.0513, 2.303], dtype=float32)

  Args:
    y_true: Ground truth values.
    y_pred: The predicted values.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    axis: (Optional) Defaults to -1. The dimension along which the entropy is
      computed.

  Returns:
    Sparse categorical crossentropy loss value.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.sparse_categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits, axis=axis)


@keras_export('keras.metrics.binary_crossentropy',
              'keras.losses.binary_crossentropy')
@dispatch.add_dispatch_support
def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
  """Computes the binary crossentropy loss.

  Standalone usage:

  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> loss.numpy()
  array([0.916 , 0.714], dtype=float32)

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.

  Returns:
    Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  label_smoothing = ops.convert_to_tensor_v2(label_smoothing, dtype=K.floatx())

  def _smooth_labels():
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

  y_true = smart_cond.smart_cond(label_smoothing,
                                 _smooth_labels, lambda: y_true)
  return K.mean(
      K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)


@keras_export('keras.metrics.kl_divergence',
              'keras.metrics.kullback_leibler_divergence',
              'keras.metrics.kld',
              'keras.metrics.KLD',
              'keras.losses.kl_divergence',
              'keras.losses.kullback_leibler_divergence',
              'keras.losses.kld',
              'keras.losses.KLD')
@dispatch.add_dispatch_support
def kl_divergence(y_true, y_pred):
  """Computes Kullback-Leibler divergence loss between `y_true` and `y_pred`.

  `loss = y_true * log(y_true / y_pred)`

  See: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3)).astype(np.float64)
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.kullback_leibler_divergence(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> y_true = tf.keras.backend.clip(y_true, 1e-7, 1)
  >>> y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1)
  >>> assert np.array_equal(
  ...     loss.numpy(), np.sum(y_true * np.log(y_true / y_pred), axis=-1))

  Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.

  Returns:
    A `Tensor` with loss.

  Raises:
    TypeError: If `y_true` cannot be cast to the `y_pred.dtype`.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  y_true = K.clip(y_true, K.epsilon(), 1)
  y_pred = K.clip(y_pred, K.epsilon(), 1)
  return math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)


@keras_export('keras.metrics.poisson', 'keras.losses.poisson')
@dispatch.add_dispatch_support
def poisson(y_true, y_pred):
  """Computes the Poisson loss between y_true and y_pred.

  The Poisson loss is the mean of the elements of the `Tensor`
  `y_pred - y_true * log(y_pred)`.

  Standalone usage:

  >>> y_true = np.random.randint(0, 2, size=(2, 3))
  >>> y_pred = np.random.random(size=(2, 3))
  >>> loss = tf.keras.losses.poisson(y_true, y_pred)
  >>> assert loss.shape == (2,)
  >>> y_pred = y_pred + 1e-7
  >>> assert np.allclose(
  ...     loss.numpy(), np.mean(y_pred - y_true * np.log(y_pred), axis=-1),
  ...     atol=1e-5)

  Args:
    y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`.
    y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`.

  Returns:
     Poisson loss value. shape = `[batch_size, d0, .. dN-1]`.

  Raises:
    InvalidArgumentError: If `y_true` and `y_pred` have incompatible shapes.
  """
  y_pred = ops.convert_to_tensor_v2(y_pred)
  y_true = math_ops.cast(y_true, y_pred.dtype)
  return K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)


@keras_export(
    'keras.losses.cosine_similarity',
    v1=[
        'keras.metrics.cosine_proximity',
        'keras.metrics.cosine',
        'keras.losses.cosine_proximity',
        'keras.losses.cosine',
        'keras.losses.cosine_similarity',
    ])
@dispatch.add_dispatch_support
def cosine_similarity(y_true, y_pred, axis=-1):
  """Computes the cosine similarity between labels and predictions.

  Note that it is a number between -1 and 1. When it is a negative number
  between -1 and 0, 0 indicates orthogonality and values closer to -1
  indicate greater similarity. The values closer to 1 indicate greater
  dissimilarity. This makes it usable as a loss function in a setting
  where you try to maximize the proximity between predictions and
  targets. If either `y_true` or `y_pred` is a zero vector, cosine
  similarity will be 0 regardless of the proximity between predictions
  and targets.

  `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`

  Standalone usage:

  >>> y_true = [[0., 1.], [1., 1.], [1., 1.]]
  >>> y_pred = [[1., 0.], [1., 1.], [-1., -1.]]
  >>> loss = tf.keras.losses.cosine_similarity(y_true, y_pred, axis=1)
  >>> loss.numpy()
  array([-0., -0.999, 0.999], dtype=float32)

  Args:
    y_true: Tensor of true targets.
    y_pred: Tensor of predicted targets.
    axis: Axis along which to determine similarity.

  Returns:
    Cosine similarity tensor.
  """
  y_true = nn.l2_normalize(y_true, axis=axis)
  y_pred = nn.l2_normalize(y_pred, axis=axis)
  return -math_ops.reduce_sum(y_true * y_pred, axis=axis)


@keras_export('keras.losses.CosineSimilarity')
class CosineSimilarity(LossFunctionWrapper):
  """Computes the cosine similarity between labels and predictions.

  Note that it is a negative quantity between -1 and 0, where 0 indicates
  orthogonality and values closer to -1 indicate greater similarity. This makes
  it usable as a loss function in a setting where you try to maximize the
  proximity between predictions and targets. If either `y_true` or `y_pred`
  is a zero vector, cosine similarity will be 0 regardless of the proximity
  between predictions and targets.

  `loss = -sum(l2_norm(y_true) * l2_norm(y_pred))`

  Standalone usage:

  >>> y_true = [[0., 1.], [1., 1.]]
  >>> y_pred = [[1., 0.], [1., 1.]]
  >>> # Using 'auto'/'sum_over_batch_size' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
  >>> # l2_norm(y_true) = [[0., 1.], [1./1.414], 1./1.414]]]
  >>> # l2_norm(y_pred) = [[1., 0.], [1./1.414], 1./1.414]]]
  >>> # l2_norm(y_true) . l2_norm(y_pred) = [[0., 0.], [0.5, 0.5]]
  >>> # loss = mean(sum(l2_norm(y_true) . l2_norm(y_pred), axis=1))
  >>> #       = -((0. + 0.) +  (0.5 + 0.5)) / 2
  >>> cosine_loss(y_true, y_pred).numpy()
  -0.5

  >>> # Calling with 'sample_weight'.
  >>> cosine_loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  -0.0999

  >>> # Using 'sum' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> cosine_loss(y_true, y_pred).numpy()
  -0.999

  >>> # Using 'none' reduction type.
  >>> cosine_loss = tf.keras.losses.CosineSimilarity(axis=1,
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> cosine_loss(y_true, y_pred).numpy()
  array([-0., -0.999], dtype=float32)

  Usage with the `compile()` API:

  ```python
  model.compile(optimizer='sgd', loss=tf.keras.losses.CosineSimilarity(axis=1))
  ```

  Args:
    axis: (Optional) Defaults to -1. The dimension along which the cosine
      similarity is computed.
    reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. `AUTO` indicates that the reduction option will
      be determined by the usage context. For almost all cases this defaults to
      `SUM_OVER_BATCH_SIZE`. When used with `tf.distribute.Strategy`, outside of
      built-in training loops such as `tf.keras` `compile` and `fit`, using
      `AUTO` or `SUM_OVER_BATCH_SIZE` will raise an error. Please see this
      custom training [tutorial]
      (https://www.tensorflow.org/tutorials/distribute/custom_training) for more
        details.
    name: Optional name for the op.
  """

  def __init__(self,
               axis=-1,
               reduction=losses_utils.ReductionV2.AUTO,
               name='cosine_similarity'):
    super(CosineSimilarity, self).__init__(
        cosine_similarity, reduction=reduction, name=name, axis=axis)


# Aliases.

bce = BCE = binary_crossentropy
mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence = kl_divergence
logcosh = log_cosh
huber_loss = huber


def is_categorical_crossentropy(loss):
  result = ((isinstance(loss, CategoricalCrossentropy) or
             (isinstance(loss, LossFunctionWrapper) and
              loss.fn == categorical_crossentropy) or
             (hasattr(loss, '__name__') and
              loss.__name__ == 'categorical_crossentropy') or
             (loss == 'categorical_crossentropy')))
  return result


@keras_export('keras.losses.serialize')
def serialize(loss):
  """Serializes loss function or `Loss` instance.

  Arguments:
    loss: A Keras `Loss` instance or a loss function.

  Returns:
    Loss configuration dictionary.
  """
  return serialize_keras_object(loss)


@keras_export('keras.losses.deserialize')
def deserialize(name, custom_objects=None):
  """Deserializes a serialized loss class/function instance.

  Arguments:
      name: Loss configuration.
      custom_objects: Optional dictionary mapping names (strings) to custom
        objects (classes and functions) to be considered during deserialization.

  Returns:
      A Keras `Loss` instance or a loss function.
  """
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='loss function')


@keras_export('keras.losses.get')
def get(identifier):
  """Retrieves a Keras loss as a `function`/`Loss` class instance.

  The `identifier` may be the string name of a loss function or `Loss` class.

  >>> loss = tf.keras.losses.get("categorical_crossentropy")
  >>> type(loss)
  <class 'function'>
  >>> loss = tf.keras.losses.get("CategoricalCrossentropy")
  >>> type(loss)
  <class '...tensorflow.python.keras.losses.CategoricalCrossentropy'>

  You can also specify `config` of the loss to this function by passing dict
  containing `class_name` and `config` as an identifier. Also note that the
  `class_name` must map to a `Loss` class

  >>> identifier = {"class_name": "CategoricalCrossentropy",
  ...               "config": {"from_logits": True}}
  >>> loss = tf.keras.losses.get(identifier)
  >>> type(loss)
  <class '...tensorflow.python.keras.losses.CategoricalCrossentropy'>

  Arguments:
    identifier: A loss identifier. One of None or string name of a loss
      function/class or loss configuration dictionary or a loss function or a
      loss class instance

  Returns:
    A Keras loss as a `function`/ `Loss` class instance.

  Raises:
    ValueError: If `identifier` cannot be interpreted.
  """
  if identifier is None:
    return None
  if isinstance(identifier, six.string_types):
    identifier = str(identifier)
    return deserialize(identifier)
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError(
        'Could not interpret loss function identifier: {}'.format(identifier))


LABEL_DTYPES_FOR_LOSSES = {
    losses_impl.sparse_softmax_cross_entropy: 'int32',
    sparse_categorical_crossentropy: 'int32'
}
