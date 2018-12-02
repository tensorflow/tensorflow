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
"""Batch Norm bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.util import deprecation


__all__ = [
    "BatchNormalization",
]


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def _undo_batch_normalization(x,
                              mean,
                              variance,
                              offset,
                              scale,
                              variance_epsilon,
                              name=None):
  r"""Inverse of tf.nn.batch_normalization.

  Args:
    x: Input `Tensor` of arbitrary dimensionality.
    mean: A mean `Tensor`.
    variance: A variance `Tensor`.
    offset: An offset `Tensor`, often denoted `beta` in equations, or
      None. If present, will be added to the normalized tensor.
    scale: A scale `Tensor`, often denoted `gamma` in equations, or
      `None`. If present, the scale is applied to the normalized tensor.
    variance_epsilon: A small `float` added to the minibatch `variance` to
      prevent dividing by zero.
    name: A name for this operation (optional).

  Returns:
    batch_unnormalized: The de-normalized, de-scaled, de-offset `Tensor`.
  """
  with ops.name_scope(
      name, "undo_batchnorm", [x, mean, variance, scale, offset]):
    # inv = math_ops.rsqrt(variance + variance_epsilon)
    # if scale is not None:
    #   inv *= scale
    # return x * inv + (
    #     offset - mean * inv if offset is not None else -mean * inv)
    rescale = math_ops.sqrt(variance + variance_epsilon)
    if scale is not None:
      rescale /= scale
    batch_unnormalized = x * rescale + (
        mean - offset * rescale if offset is not None else mean)
    return batch_unnormalized


class BatchNormalization(bijector.Bijector):
  """Compute `Y = g(X) s.t. X = g^-1(Y) = (Y - mean(Y)) / std(Y)`.

  Applies Batch Normalization [(Ioffe and Szegedy, 2015)][1] to samples from a
  data distribution. This can be used to stabilize training of normalizing
  flows ([Papamakarios et al., 2016][3]; [Dinh et al., 2017][2])

  When training Deep Neural Networks (DNNs), it is common practice to
  normalize or whiten features by shifting them to have zero mean and
  scaling them to have unit variance.

  The `inverse()` method of the `BatchNormalization` bijector, which is used in
  the log-likelihood computation of data samples, implements the normalization
  procedure (shift-and-scale) using the mean and standard deviation of the
  current minibatch.

  Conversely, the `forward()` method of the bijector de-normalizes samples (e.g.
  `X*std(Y) + mean(Y)` with the running-average mean and standard deviation
  computed at training-time. De-normalization is useful for sampling.

  ```python

  dist = tfd.TransformedDistribution(
      distribution=tfd.Normal()),
      bijector=tfb.BatchNorm())

  y = tfd.MultivariateNormalDiag(loc=1., scale=2.).sample(100)  # ~ N(1, 2)
  x = dist.bijector.inverse(y)  # ~ N(0, 1)
  y = dist.sample()  # ~ N(1, 2)
  ```

  During training time, `BatchNorm.inverse` and `BatchNorm.forward` are not
  guaranteed to be inverses of each other because `inverse(y)` uses statistics
  of the current minibatch, while `forward(x)` uses running-average statistics
  accumulated from training. In other words,
  `BatchNorm.inverse(BatchNorm.forward(...))` and
  `BatchNorm.forward(BatchNorm.inverse(...))` will be identical when
  `training=False` but may be different when `training=True`.

  #### References

  [1]: Sergey Ioffe and Christian Szegedy. Batch Normalization: Accelerating
       Deep Network Training by Reducing Internal Covariate Shift. In
       _International Conference on Machine Learning_, 2015.
       https://arxiv.org/abs/1502.03167

  [2]: Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation
       using Real NVP. In _International Conference on Learning
       Representations_, 2017. https://arxiv.org/abs/1605.08803

  [3]: George Papamakarios, Theo Pavlakou, and Iain Murray. Masked
       Autoregressive Flow for Density Estimation. In _Neural Information
       Processing Systems_, 2017. https://arxiv.org/abs/1705.07057
  """

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self,
               batchnorm_layer=None,
               training=True,
               validate_args=False,
               name="batch_normalization"):
    """Instantiates the `BatchNorm` bijector.

    Args:
      batchnorm_layer: `tf.layers.BatchNormalization` layer object. If `None`,
        defaults to
        `tf.layers.BatchNormalization(gamma_constraint=nn_ops.relu(x) + 1e-6)`.
        This ensures positivity of the scale variable.

      training: If True, updates running-average statistics during call to
        `inverse()`.
      validate_args: Python `bool` indicating whether arguments should be
        checked for correctness.
      name: Python `str` name given to ops managed by this object.
    Raises:
      ValueError: If bn_layer is not an instance of
        `tf.layers.BatchNormalization`, or if it is specified with `renorm=True`
        or a virtual batch size.
    """
    # Scale must be positive.
    g_constraint = lambda x: nn.relu(x) + 1e-6
    self.batchnorm = batchnorm_layer or normalization.BatchNormalization(
        gamma_constraint=g_constraint)
    self._validate_bn_layer(self.batchnorm)
    self._training = training
    if isinstance(self.batchnorm.axis, int):
      forward_min_event_ndims = 1
    else:
      forward_min_event_ndims = len(self.batchnorm.axis)
    super(BatchNormalization, self).__init__(
        forward_min_event_ndims=forward_min_event_ndims,
        validate_args=validate_args, name=name)

  def _validate_bn_layer(self, layer):
    """Check for valid BatchNormalization layer.

    Args:
      layer: Instance of `tf.layers.BatchNormalization`.
    Raises:
      ValueError: If batchnorm_layer argument is not an instance of
      `tf.layers.BatchNormalization`, or if `batchnorm_layer.renorm=True` or
      if `batchnorm_layer.virtual_batch_size` is specified.
    """
    if not isinstance(layer, normalization.BatchNormalization):
      raise ValueError(
          "batchnorm_layer must be an instance of BatchNormalization layer.")
    if layer.renorm:
      raise ValueError("BatchNorm Bijector does not support renormalization.")
    if layer.virtual_batch_size:
      raise ValueError(
          "BatchNorm Bijector does not support virtual batch sizes.")

  def _get_broadcast_fn(self, x):
    # Compute shape to broadcast scale/shift parameters to.
    if not x.shape.is_fully_defined():
      raise ValueError("Input must have shape known at graph construction.")
    input_shape = np.int32(x.shape.as_list())

    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.batchnorm.axis]
    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.batchnorm.axis[0]] = (
        input_shape[self.batchnorm.axis[0]])
    def _broadcast(v):
      if (v is not None and
          len(v.get_shape()) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return array_ops.reshape(v, broadcast_shape)
      return v
    return _broadcast

  def _normalize(self, y):
    return self.batchnorm.apply(y, training=self._training)

  def _de_normalize(self, x):
    # Uses the saved statistics.
    if not self.batchnorm.built:
      input_shape = x.get_shape()
      self.batchnorm.build(input_shape)
    broadcast_fn = self._get_broadcast_fn(x)
    mean = broadcast_fn(self.batchnorm.moving_mean)
    variance = broadcast_fn(self.batchnorm.moving_variance)
    beta = broadcast_fn(self.batchnorm.beta) if self.batchnorm.center else None
    gamma = broadcast_fn(self.batchnorm.gamma) if self.batchnorm.scale else None
    return _undo_batch_normalization(
        x, mean, variance, beta, gamma, self.batchnorm.epsilon)

  def _forward(self, x):
    return self._de_normalize(x)

  def _inverse(self, y):
    return self._normalize(y)

  def _forward_log_det_jacobian(self, x):
    # Uses saved statistics to compute volume distortion.
    return -self._inverse_log_det_jacobian(x, use_saved_statistics=True)

  def _inverse_log_det_jacobian(self, y, use_saved_statistics=False):
    if not y.shape.is_fully_defined():
      raise ValueError("Input must have shape known at graph construction.")
    input_shape = np.int32(y.shape.as_list())

    if not self.batchnorm.built:
      # Create variables.
      self.batchnorm.build(input_shape)

    event_dims = self.batchnorm.axis
    reduction_axes = [i for i in range(len(input_shape)) if i not in event_dims]

    if use_saved_statistics or not self._training:
      log_variance = math_ops.log(
          self.batchnorm.moving_variance + self.batchnorm.epsilon)
    else:
      # At training-time, ildj is computed from the mean and log-variance across
      # the current minibatch.
      _, v = nn.moments(y, axes=reduction_axes, keep_dims=True)
      log_variance = math_ops.log(v + self.batchnorm.epsilon)

    # `gamma` and `log Var(y)` reductions over event_dims.
    # Log(total change in area from gamma term).
    log_total_gamma = math_ops.reduce_sum(math_ops.log(self.batchnorm.gamma))

    # Log(total change in area from log-variance term).
    log_total_variance = math_ops.reduce_sum(log_variance)
    # The ildj is scalar, as it does not depend on the values of x and are
    # constant across minibatch elements.
    return log_total_gamma - 0.5 * log_total_variance
