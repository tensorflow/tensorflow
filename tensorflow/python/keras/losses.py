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
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.keras.utils.losses_utils import compute_weighted_loss
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util.tf_export import keras_export


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
    with ops.name_scope(self.name, format(self.__class__.__name__),
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
  def call(self, y_true, y_pred):
    """Invokes the `Loss` instance.

    Args:
      y_true: Ground truth values, with the same shape as 'y_pred'.
      y_pred: The predicted values.
    """
    NotImplementedError('Must be implemented in subclasses.')


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
  """Computes the binary cross entropy loss between the labels and predictions.

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
  ````

  Args:
    from_logits: Whether `output` is expected to be a logits tensor. By default,
      we consider that `output` encodes a probability distribution.
    label_smoothing: If greater than `0` then smooth the labels.
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
    self.label_smoothing = label_smoothing

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

    if self.label_smoothing > 0:
      y_true = y_true * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

    return binary_crossentropy(y_true, y_pred, from_logits=self.from_logits)


@keras_export('keras.losses.CategoricalCrossentropy')
class CategoricalCrossentropy(Loss):
  """Computes categorical cross entropy loss between the `y_true` and `y_pred`.

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
  ````

  Args:
    from_logits: Whether `output` is expected to be a logits tensor. By default,
      we consider that `output` encodes a probability distribution.
    label_smoothing: If greater than `0` then smooth the labels. This option is
      currently not supported when `y_pred` is a sparse input (not one-hot).
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
    self.label_smoothing = label_smoothing

  def call(self, y_true, y_pred):
    """Invokes the `CategoricalCrossentropy` instance.

    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.

    Returns:
      Categorical cross entropy losses.
    """
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true)
    is_sparse = y_pred.shape != y_true.shape

    if is_sparse:
      return sparse_categorical_crossentropy(
          y_true, y_pred, from_logits=self.from_logits)
    else:
      y_true = math_ops.cast(y_true, y_pred.dtype)
      if self.label_smoothing > 0:
        num_classes = math_ops.cast(array_ops.shape(y_true)[1], y_pred.dtype)
        smooth_positives = 1.0 - self.label_smoothing
        smooth_negatives = self.label_smoothing / num_classes
        y_true = y_true * smooth_positives + smooth_negatives

      return categorical_crossentropy(
          y_true, y_pred, from_logits=self.from_logits)


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


@keras_export('keras.metrics.mean_squared_error',
              'keras.metrics.mse',
              'keras.metrics.MSE',
              'keras.losses.mean_squared_error',
              'keras.losses.mse',
              'keras.losses.MSE')
def mean_squared_error(y_true, y_pred):
  return K.mean(math_ops.square(y_pred - y_true), axis=-1)


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
  return K.mean(math_ops.square(first_log - second_log), axis=-1)


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
def categorical_crossentropy(y_true, y_pred, from_logits=False):
  return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


@keras_export('keras.metrics.sparse_categorical_crossentropy',
              'keras.losses.sparse_categorical_crossentropy')
def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False):
  return K.sparse_categorical_crossentropy(
      y_true, y_pred, from_logits=from_logits)


@keras_export('keras.metrics.binary_crossentropy',
              'keras.losses.binary_crossentropy')
def binary_crossentropy(y_true, y_pred, from_logits=False):
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
def cosine_proximity(y_true, y_pred):
  y_true = nn.l2_normalize(y_true, axis=-1)
  y_pred = nn.l2_normalize(y_pred, axis=-1)
  return -math_ops.reduce_sum(y_true * y_pred, axis=-1)


class CosineProximity(Loss):
  """Computes the cosine distance between `y_true` and `y_pred`.

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
  """

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
    return cosine_proximity(y_true, y_pred)


# Aliases.

mse = MSE = mean_squared_error
mae = MAE = mean_absolute_error
mape = MAPE = mean_absolute_percentage_error
msle = MSLE = mean_squared_logarithmic_error
kld = KLD = kullback_leibler_divergence
cosine = cosine_proximity


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
