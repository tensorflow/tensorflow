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
"""Built-in loss functions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc

import six

from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.losses_utils import compute_weighted_loss
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls


@keras_export('keras.losses.Loss')
class Loss(object):
  """Loss base class.

  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.

  Example subclass implementation:
  ```
  class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
      y_pred = ops.convert_to_tensor(y_pred)
      y_true = math_ops.cast(y_true, y_pred.dtype)
      return K.mean(math_ops.square(y_pred - y_true), axis=-1)
  ```

  Args:
    reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
      `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
  """

  def __init__(self,
               reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    self.reduction = reduction
    self.name = name

  def __call__(self, y_true, y_pred, sample_weight=None):
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
      sample_weight: Optional `Tensor` whose rank is either 0, or the same rank
        as `y_true`, or is broadcastable to `y_true`. `sample_weight` acts as a
        coefficient for the loss. If a scalar is provided, then the loss is
        simply scaled by the given value. If `sample_weight` is a tensor of size
        `[batch_size]`, then the total loss for each sample of the batch is
        rescaled by the corresponding element in the `sample_weight` vector. If
        the shape of `sample_weight` matches the shape of `y_pred`, then the
        loss of each measurable element of `y_pred` is scaled by the
        corresponding value of `sample_weight`.

    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has the same
        shape as `y_true`; otherwise, it is scalar.

    Raises:
      ValueError: If the shape of `sample_weight` is invalid.
    """
    # If we are wrapping a lambda function strip '<>' from the name as it is not
    # accepted in scope name.
    scope_name = 'lambda' if self.name == '<lambda>' else self.name
    with ops.name_scope(scope_name, format(self.__class__.__name__),
                        (y_pred, y_true, sample_weight)):
      losses = self.call(y_true, y_pred)
      return compute_weighted_loss(
          losses, sample_weight, reduction=self.reduction)

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
    return {'reduction': self.reduction, 'name': self.name}

  @abc.abstractmethod
  @doc_controls.for_subclass_implementers
  def call(self, y_true, y_pred):
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values, with the same shape as 'y_pred'.
      y_pred: The predicted values.
    """
    NotImplementedError('Must be implemented in subclasses.')


class LossFunctionWrapper(Loss):
  """Wraps a loss function in the `Loss` class.

  Args:
    fn: The loss function to wrap, with signature `fn(y_true, y_pred,
      **kwargs)`.
    reduction: (Optional) Type of `tf.losses.Reduction` to apply to loss.
      Default value is `SUM_OVER_BATCH_SIZE`.
    name: (Optional) name for the loss.
    **kwargs: The keyword arguments that are passed on to `fn`.
  """

  def __init__(self,
               fn,
               reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None,
               **kwargs):
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
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return self.fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {'fn': self.fn}
    config.update(self._fn_kwargs)
    base_config = super(LossFunctionWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.losses.MeanSquaredError')
class MeanSquaredError(Loss):
  """Computes the mean of squares of errors between labels and predictions.

  For example, if `y_true` is [0., 0., 1., 1.] and `y_pred` is [1., 1., 1., 0.]
  then the mean squared error value is 3/4 (0.75).

  Usage:

  ```python
  mse = tf.keras.losses.MeanSquaredError()
  loss = mse([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 0.75
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.MeanSquaredError())
  ```
  """

  def call(self, y_true, y_pred):
    """Invokes the `MeanSquaredError` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Mean squared error losses.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return mean_squared_error(y_true, y_pred)


@keras_export('keras.losses.MeanAbsoluteError')
class MeanAbsoluteError(Loss):
  """Computes the mean of absolute difference between labels and predictions.

  For example, if `y_true` is [0., 0., 1., 1.] and `y_pred` is [1., 1., 1., 0.]
  then the mean absolute error value is 3/4 (0.75).

  Usage:

  ```python
  mae = tf.keras.losses.MeanAbsoluteError()
  loss = mae([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 0.75
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.MeanAbsoluteError())
  ```
  """

  def call(self, y_true, y_pred):
    """Invokes the `MeanAbsoluteError` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Mean absolute error losses.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return mean_absolute_error(y_true, y_pred)


@keras_export('keras.losses.MeanAbsolutePercentageError')
class MeanAbsolutePercentageError(Loss):
  """Computes the mean absolute percentage error between `y_true` and `y_pred`.

  For example, if `y_true` is [0., 0., 1., 1.] and `y_pred` is [1., 1., 1., 0.]
  then the mean absolute percentage error value is 5e+08.

  Usage:

  ```python
  mape = tf.keras.losses.MeanAbsolutePercentageError()
  loss = mape([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 5e+08
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.MeanAbsolutePercentageError())
  ```
  """

  def call(self, y_true, y_pred):
    """Invokes the `MeanAbsolutePercentageError` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Mean absolute percentage error losses.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return mean_absolute_percentage_error(y_true, y_pred)


@keras_export('keras.losses.MeanSquaredLogarithmicError')
class MeanSquaredLogarithmicError(Loss):
  """Computes the mean squared logarithmic error between `y_true` and `y_pred`.

  For example, if `y_true` is [0., 0., 1., 1.] and `y_pred` is [1., 1., 1., 0.]
  then the mean squared logarithmic error value is 0.36034.

  Usage:

  ```python
  msle = tf.keras.losses.MeanSquaredLogarithmicError()
  loss = msle([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 0.36034
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.MeanSquaredLogarithmicError())
  ```
  """

  def call(self, y_true, y_pred):
    """Invokes the `MeanSquaredLogarithmicError` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Mean squared logarithmic error losses.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return mean_squared_logarithmic_error(y_true, y_pred)


@keras_export('keras.losses.BinaryCrossentropy')
class BinaryCrossentropy(Loss):
  """Computes the crossentropy loss between the labels and predictions.

  Use this crossentropy loss function when there are only two label classes
  (assumed to be 0 and 1). There should be a single floating point value per
  feature.

  In the snippet below, there is a single floating pointing value per example,
  and the shape of both `y_pred` and `y_true` are `[batch_size]`.

  Usage:

  ```python
  bce = tf.keras.losses.BinaryCrossentropy()
  loss = bce([0., 0., 1., 1.], [1., 1., 1., 0.])
  print('Loss: ', loss.numpy())  # Loss: 12.007
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.BinaryCrossentropy())
  ```

  Args:
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
    reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
      `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
  """

  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    super(BinaryCrossentropy, self).__init__(reduction=reduction, name=name)
    self.from_logits = from_logits
    self.label_smoothing = ops.convert_to_tensor(
        label_smoothing, dtype=K.floatx())

  def call(self, y_true, y_pred):
    """Invokes the `BinaryCrossentropy` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Binary cross entropy losses.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return binary_crossentropy(
        y_true,
        y_pred,
        from_logits=self.from_logits,
        label_smoothing=self.label_smoothing)


@keras_export('keras.losses.CategoricalCrossentropy')
class CategoricalCrossentropy(Loss):
  """Computes the crossentropy loss between the labels and predictions.

  Use this crossentropy loss function when there are two or more label classes.
  We expect labels to be provided in a `one_hot` representation. If you want to
  provide labels as integers, please use `SparseCategoricalCrossentropy` loss.
  There should be `# classes` floating point values per feature.

  In the snippet below, there is `# classes` floating pointing values per
  example. The shape of both `y_pred` and `y_true` are
  `[batch_size, num_classes]`.

  Usage:

  ```python
  cce = tf.keras.losses.CategoricalCrossentropy()
  loss = cce(
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
  print('Loss: ', loss.numpy())  # Loss: 0.3239
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.CategoricalCrossentropy())
  ```

  Args:
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. When > 0, label values are smoothed,
      meaning the confidence on label values are relaxed. e.g.
      `label_smoothing=0.2` means that we will use a value of `0.1` for label
      `0` and `0.9` for label `1`"
    reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
      `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
  """

  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    super(CategoricalCrossentropy, self).__init__(
        reduction=reduction, name=name)
    self.from_logits = from_logits
    self.label_smoothing = ops.convert_to_tensor(
        label_smoothing, dtype=K.floatx())

  def call(self, y_true, y_pred):
    """Invokes the `CategoricalCrossentropy` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Categorical cross entropy losses.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return categorical_crossentropy(
        y_true,
        y_pred,
        from_logits=self.from_logits,
        label_smoothing=self.label_smoothing)


@keras_export('keras.losses.SparseCategoricalCrossentropy')
class SparseCategoricalCrossentropy(Loss):
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

  Usage:

  ```python
  cce = tf.keras.losses.SparseCategoricalCrossentropy()
  loss = cce(
    [0, 1, 2],
    [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
  print('Loss: ', loss.numpy())  # Loss: 0.3239
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.keras.losses.SparseCategoricalCrossentropy())
  ````

  Args:
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
      `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
  """

  def __init__(self,
               from_logits=False,
               reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    super(SparseCategoricalCrossentropy, self).__init__(
        reduction=reduction, name=name)
    self.from_logits = from_logits

  def call(self, y_true, y_pred):
    """Invokes the `SparseCategoricalCrossentropy` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Sparse categorical cross entropy losses.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true)
    return sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=self.from_logits)


@keras_export('keras.losses.Hinge')
class Hinge(Loss):
  """Computes the hinge loss between `y_true` and `y_pred`.

  Usage:

  ```python
  h = tf.losses.Hinge()
  loss = h([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 0.66
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.losses.Hinge())
  ```
  """

  def call(self, y_true, y_pred):
    """Calculates the hinge loss.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Hinge loss.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return hinge(y_true, y_pred)


@keras_export('keras.losses.SquaredHinge')
class SquaredHinge(Loss):
  """Computes the squared hinge loss between `y_true` and `y_pred`.

  Usage:

  ```python
  sh = tf.losses.SquaredHinge()
  loss = sh([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 0.66
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.losses.SquaredHinge())
  ```
  """

  def call(self, y_true, y_pred):
    """Calculates the squared hinge loss.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Squared hinge loss.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return squared_hinge(y_true, y_pred)


@keras_export('keras.losses.CategoricalHinge')
class CategoricalHinge(Loss):
  """Computes the categorical hinge loss between `y_true` and `y_pred`.

  Usage:

  ```python
  ch = tf.losses.CategoricalHinge()
  loss = ch([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 1.0
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.losses.CategoricalHinge())
  ```
  """

  def call(self, y_true, y_pred):
    """Calculates the categorical hinge loss.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Categorical hinge loss.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return categorical_hinge(y_true, y_pred)


@keras_export('keras.losses.LogLoss')
class LogLoss(Loss):
  """Computes the log loss between `y_true` and `y_pred`.

  `logloss = - y_true * log(y_pred) - (1 - y_true) * log(1 - y_pred)`

  Usage:

  ```python
  l = tf.losses.LogLoss()
  loss = l([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 10.745
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.losses.LogLoss())
  ```
  """

  def call(self, y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return logloss(y_true, y_pred)


@keras_export('keras.losses.Poisson')
class Poisson(Loss):
  """Computes the Poisson loss between `y_true` and `y_pred`.

  `loss = y_pred - y_true * log(y_pred)`

  Usage:

  ```python
  p = tf.losses.Poisson()
  loss = p([1, 9, 2], [4, 8, 12])
  print('Loss: ', loss.numpy())  # Loss: -4.63
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.losses.Poisson())
  ```
  """

  def call(self, y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return poisson(y_true, y_pred)


@keras_export('keras.losses.LogCosh')
class LogCosh(Loss):
  """Computes the logarithm of the hyperbolic cosine of the prediction error.

  `logcosh = log((exp(x) + exp(-x))/2)`, where x is the error (y_pred - y_true)

  Usage:

  ```python
  l = tf.losses.LogCosh()
  loss = l([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 0.289
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.losses.LogCosh())
  ```
  """

  def call(self, y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return logcosh(y_true, y_pred)


@keras_export('keras.losses.KLDivergence')
class KLDivergence(Loss):
  """Computes Kullback Leibler divergence loss between `y_true` and `y_pred`.

  `loss = y_true * log(y_true / y_pred)`

  Usage:

  ```python
  k = tf.losses.KLDivergence()
  loss = k([.4, .9, .2], [.5, .8, .12])
  print('Loss: ', loss.numpy())  # Loss: -0.043
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.losses.KLDivergence())
  ```
  """

  def call(self, y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return kullback_leibler_divergence(y_true, y_pred)


@keras_export('keras.losses.Huber')
class Huber(Loss):
  """Computes the Huber loss between `y_true` and `y_pred`.

  For each value x in `error=y_true-y_pred`, the following is calculated:

  ```
  0.5 * x^2                  if |x| <= d
  0.5 * d^2 + d * (|x| - d)  if |x| > d
  ```
  where d is `delta`. See: https://en.wikipedia.org/wiki/Huber_loss

  Usage:

  ```python
  l = tf.losses.Huber()
  loss = l([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: 0.333
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.losses.Huber())
  ```

  Args:
    delta: A float, the point where the Huber loss function changes from a
      quadratic to linear.
    reduction: Type of `tf.losses.Reduction` to apply to loss. Default value is
      `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
  """

  def __init__(self,
               delta=1.0,
               reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    super(Huber, self).__init__(reduction=reduction, name=name)
    self.delta = delta

  def call(self, y_true, y_pred):
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return huber_loss(y_true, y_pred, delta=self.delta)


@keras_export('keras.metrics.mean_squared_error',
              'keras.metrics.mse',
              'keras.metrics.MSE',
              'keras.losses.mean_squared_error',
              'keras.losses.mse',
              'keras.losses.MSE')
def mean_squared_error(y_true, y_pred):
  return K.mean(math_ops.squared_difference(y_pred, y_true), axis=-1)


@keras_export('keras.metrics.mean_absolute_error',
              'keras.metrics.mae',
              'keras.metrics.MAE',
              'keras.losses.mean_absolute_error',
              'keras.losses.mae',
              'keras.losses.MAE')
def mean_absolute_error(y_true, y_pred):
  return K.mean(math_ops.abs(y_pred - y_true), axis=-1)


@keras_export('keras.metrics.mean_absolute_percentage_error',
              'keras.metrics.mape',
              'keras.metrics.MAPE',
              'keras.losses.mean_absolute_percentage_error',
              'keras.losses.mape',
              'keras.losses.MAPE')
def mean_absolute_percentage_error(y_true, y_pred):
  diff = math_ops.abs(
      (y_true - y_pred) / K.clip(math_ops.abs(y_true), K.epsilon(), None))
  return 100. * K.mean(diff, axis=-1)


@keras_export('keras.metrics.mean_squared_logarithmic_error',
              'keras.metrics.msle',
              'keras.metrics.MSLE',
              'keras.losses.mean_squared_logarithmic_error',
              'keras.losses.msle',
              'keras.losses.MSLE')
def mean_squared_logarithmic_error(y_true, y_pred):
  first_log = math_ops.log(K.clip(y_pred, K.epsilon(), None) + 1.)
  second_log = math_ops.log(K.clip(y_true, K.epsilon(), None) + 1.)
  return K.mean(math_ops.squared_difference(first_log, second_log), axis=-1)


@keras_export('keras.metrics.squared_hinge', 'keras.losses.squared_hinge')
def squared_hinge(y_true, y_pred):
  return K.mean(
      math_ops.square(math_ops.maximum(1. - y_true * y_pred, 0.)), axis=-1)


@keras_export('keras.metrics.hinge', 'keras.losses.hinge')
def hinge(y_true, y_pred):
  return K.mean(math_ops.maximum(1. - y_true * y_pred, 0.), axis=-1)


@keras_export('keras.losses.categorical_hinge')
def categorical_hinge(y_true, y_pred):
  pos = math_ops.reduce_sum(y_true * y_pred, axis=-1)
  neg = math_ops.reduce_max((1. - y_true) * y_pred, axis=-1)
  return math_ops.maximum(0., neg - pos + 1.)


def logloss(y_true, y_pred):
  losses = math_ops.multiply(y_true, math_ops.log(y_pred + K.epsilon()))
  losses += math_ops.multiply((1 - y_true),
                              math_ops.log(1 - y_pred + K.epsilon()))
  return K.mean(-losses, axis=-1)


def huber_loss(y_true, y_pred, delta=1.0):
  """Computes Huber loss value.

  For each value x in `error=y_true-y_pred`, the following is calculated:

  ```
  0.5 * x^2                  if |x| <= d
  0.5 * d^2 + d * (|x| - d)  if |x| > d
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
  error = math_ops.subtract(y_pred, y_true)
  abs_error = math_ops.abs(error)
  quadratic = math_ops.minimum(abs_error, delta)
  linear = math_ops.subtract(abs_error, quadratic)
  return math_ops.add(
      math_ops.multiply(
          ops.convert_to_tensor(0.5, dtype=quadratic.dtype),
          math_ops.multiply(quadratic, quadratic)),
      math_ops.multiply(delta, linear))


@keras_export('keras.losses.logcosh')
def logcosh(y_true, y_pred):
  """Logarithm of the hyperbolic cosine of the prediction error.

  `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
  to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
  like the mean squared error, but will not be so strongly affected by the
  occasional wildly incorrect prediction.

  Arguments:
      y_true: tensor of true targets.
      y_pred: tensor of predicted targets.

  Returns:
      Tensor with one scalar loss entry per sample.
  """

  def _logcosh(x):
    return x + nn.softplus(-2. * x) - math_ops.log(2.)

  return K.mean(_logcosh(y_pred - y_true), axis=-1)


@keras_export('keras.metrics.categorical_crossentropy',
              'keras.losses.categorical_crossentropy')
def categorical_crossentropy(y_true,
                             y_pred,
                             from_logits=False,
                             label_smoothing=0):
  """Computes the categorical crossentropy loss.

  Args:
    y_true: tensor of true targets.
    y_pred: tensor of predicted targets.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.

  Returns:
    Categorical crossentropy loss value.
  """

  def _smooth_labels():
    num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
    return y_true * (1.0 - label_smoothing) + (label_smoothing / num_classes)

  y_true = smart_cond.smart_cond(label_smoothing,
                                 _smooth_labels, lambda: y_true)
  return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


@keras_export('keras.metrics.sparse_categorical_crossentropy',
              'keras.losses.sparse_categorical_crossentropy')
def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False):
  return K.sparse_categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits)


@keras_export('keras.metrics.binary_crossentropy',
              'keras.losses.binary_crossentropy')
def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):

  def _smooth_labels():
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

  y_true = smart_cond.smart_cond(label_smoothing,
                                 _smooth_labels, lambda: y_true)
  return K.mean(
      K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)


@keras_export('keras.metrics.kullback_leibler_divergence',
              'keras.metrics.kld',
              'keras.metrics.KLD',
              'keras.losses.kullback_leibler_divergence',
              'keras.losses.kld',
              'keras.losses.KLD')
def kullback_leibler_divergence(y_true, y_pred):
  y_true = K.clip(y_true, K.epsilon(), 1)
  y_pred = K.clip(y_pred, K.epsilon(), 1)
  return math_ops.reduce_sum(y_true * math_ops.log(y_true / y_pred), axis=-1)


@keras_export('keras.metrics.poisson', 'keras.losses.poisson')
def poisson(y_true, y_pred):
  return K.mean(y_pred - y_true * math_ops.log(y_pred + K.epsilon()), axis=-1)


@keras_export('keras.metrics.cosine_proximity',
              'keras.metrics.cosine',
              'keras.losses.cosine_proximity',
              'keras.losses.cosine')
def cosine_proximity(y_true, y_pred, axis=-1):
  y_true = nn.l2_normalize(y_true, axis=axis)
  y_pred = nn.l2_normalize(y_pred, axis=axis)
  return -math_ops.reduce_sum(y_true * y_pred, axis=axis)


@keras_export('keras.losses.CosineProximity')
class CosineProximity(Loss):
  """Computes the cosine proximity between `y_true` and `y_pred`.

  Usage:

  ```python
  cosine_loss = tf.losses.CosineProximity()
  loss = cosine_loss([0., 1., 1.], [1., 0., 1.])
  print('Loss: ', loss.numpy())  # Loss: -0.5
  ```

  Usage with tf.keras API:

  ```python
  model = keras.models.Model(inputs, outputs)
  model.compile('sgd', loss=tf.losses.CosineProximity())
  ```

  Args:
    axis: (Optional) Defaults to -1. The dimension along which the cosine
      proximity is computed.
    reduction: (Optional) Type of `tf.losses.Reduction` to apply to loss.
      Default value is `SUM_OVER_BATCH_SIZE`.
    name: Optional name for the op.
  """

  def __init__(self,
               axis=-1,
               reduction=losses_impl.ReductionV2.SUM_OVER_BATCH_SIZE,
               name=None):
    super(CosineProximity, self).__init__(reduction=reduction, name=name)
    self.axis = axis

  def call(self, y_true, y_pred):
    """Calculates the cosine proximity loss.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Cosine distance loss.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return cosine_proximity(y_true, y_pred, axis=self.axis)


# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity


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
  return serialize_keras_object(loss)


@keras_export('keras.losses.deserialize')
def deserialize(name, custom_objects=None):
  return deserialize_keras_object(
      name,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='loss function')


@keras_export('keras.losses.get')
def get(identifier):
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
    raise ValueError('Could not interpret '
                     'loss function identifier:', identifier)


LABEL_DTYPES_FOR_LOSSES = {
    losses_impl.sparse_softmax_cross_entropy: 'int32',
    sparse_categorical_crossentropy: 'int32'
}
