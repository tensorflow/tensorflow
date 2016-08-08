# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""The Laplace distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.contrib.framework.python.framework import tensor_util as contrib_tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class Laplace(distribution.Distribution):
  """The Laplace distribution with location and scale > 0 parameters.

  #### Mathematical details

  The PDF of this distribution is:

  ```f(x | mu, b, b > 0) = 0.5 / b exp(-|x - mu| / b)```

  Note that the Laplace distribution can be thought of two exponential
  distributions spliced together "back-to-back."
  """

  def __init__(self,
               loc,
               scale,
               validate_args=True,
               allow_nan_stats=False,
               name="Laplace"):
    """Construct Laplace distribution with parameters `loc` and `scale`.

    The parameters `loc` and `scale` must be shaped in a way that supports
    broadcasting (e.g., `loc / scale` is a valid operation).

    Args:
      loc: Floating point tensor which characterizes the location (center)
        of the distribution.
      scale: Positive floating point tensor which characterizes the spread of
        the distribution.
      validate_args: Whether to validate input with asserts.  If `validate_args`
        is `False`, and the inputs are invalid, correct behavior is not
        guaranteed.
      allow_nan_stats:  Boolean, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: The name to give Ops created by the initializer.

    Raises:
      TypeError: if `loc` and `scale` are of different dtype.
    """
    self._allow_nan_stats = allow_nan_stats
    self._validate_args = validate_args
    with ops.op_scope([loc, scale], name):
      loc = ops.convert_to_tensor(loc)
      scale = ops.convert_to_tensor(scale)
      with ops.control_dependencies([check_ops.assert_positive(scale)] if
                                    validate_args else []):
        self._name = name
        self._loc = array_ops.identity(loc, name="loc")
        self._scale = array_ops.identity(scale, name="scale")
        self._batch_shape = self._ones().get_shape()
        self._event_shape = tensor_shape.TensorShape([])

    contrib_tensor_util.assert_same_float_dtype((loc, scale))

  @property
  def allow_nan_stats(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._allow_nan_stats

  @property
  def validate_args(self):
    """Boolean describing behavior on invalid input."""
    return self._validate_args

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._loc.dtype

  def batch_shape(self, name="batch_shape"):
    """Batch dimensions of this instance as a 1-D int32 `Tensor`.

    The product of the dimensions of the `batch_shape` is the number of
    independent distributions of this kind the instance represents.

    Args:
      name: name to give to the op.

    Returns:
      `Tensor` `batch_shape`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return array_ops.shape(self._ones())

  def get_batch_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `batch_shape`. May be only partially defined.

    Returns:
      batch shape
    """
    return self._batch_shape

  def event_shape(self, name="event_shape"):
    """Shape of a sample from a single distribution as a 1-D int32 `Tensor`.

    Args:
      name: name to give to the op.

    Returns:
      `Tensor` `event_shape`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return constant_op.constant([], dtype=dtypes.int32)

  def get_event_shape(self):
    """`TensorShape` available at graph construction time.

    Same meaning as `event_shape`. May be only partially defined.

    Returns:
      event shape
    """
    return self._event_shape

  @property
  def loc(self):
    """Distribution parameter for the location."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for scale."""
    return self._scale

  def mean(self, name="mean"):
    """Mean of this distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._scale, self._loc], name):
        return self._loc + array_ops.zeros_like(self._scale)

  def median(self, name="median"):
    """Median of this distribution."""
    return self.mean(name="median")

  def mode(self, name="mode"):
    """Mode of this distribution."""
    return self.mean(name="mode")

  def std(self, name="std"):
    """Standard deviation of this distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([self._scale, self._loc], name):
        sqrt_2 = constant_op.constant(math.sqrt(2.), dtype=self.dtype)
        return sqrt_2 * self._scale + array_ops.zeros_like(self._loc)

  def variance(self, name="variance"):
    """Variance of this distribution."""
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        return math_ops.square(self.std())

  def prob(self, x, name="pdf"):
    """The prob of observations in `x` under the Laplace distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `loc` and `scale`.
      name: The name to give this op.

    Returns:
      pdf: tensor of dtype `dtype`, the pdf values of `x`.
    """
    return 0.5 / self._scale * math_ops.exp(
        -math_ops.abs(x - self._loc) / self._scale)

  def log_prob(self, x, name="log_prob"):
    """Log prob of observations in `x` under these Laplace distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `loc` and `scale`.
      name: The name to give this op.

    Returns:
      log_prob: tensor of dtype `dtype`, the log-probability of `x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._loc, self._scale, x], name):
        x = ops.convert_to_tensor(x)
        if x.dtype != self.dtype:
          raise TypeError("Input x dtype does not match dtype: %s vs. %s"
                          % (x.dtype, self.dtype))
        log_2 = constant_op.constant(math.log(2.), dtype=self.dtype)
        return (-log_2 - math_ops.log(self._scale) -
                math_ops.abs(x - self._loc) / self._scale)

  def cdf(self, x, name="cdf"):
    """CDF of observations in `x` under the Laplace distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `loc` and `scale`.
      name: The name to give this op.

    Returns:
      cdf: tensor of dtype `dtype`, the CDFs of `x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._loc, self._scale, x], name):
        x = ops.convert_to_tensor(x)
        if x.dtype != self.dtype:
          raise TypeError("Input x dtype does not match dtype: %s vs. %s"
                          % (x.dtype, self.dtype))
        y = x - self._loc
        return 0.5 + 0.5 * math_ops.sign(y) * (
            1. - math_ops.exp(-math_ops.abs(y) / self._scale))

  def log_cdf(self, x, name="log_cdf"):
    """Log CDF of observations `x` under the Laplace distribution(s).

    Args:
      x: tensor of dtype `dtype`, must be broadcastable with `loc` and `scale`.
      name: The name to give this op.

    Returns:
      log_cdf: tensor of dtype `dtype`, the log-CDFs of `x`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._loc, self._scale, x], name):
        return math_ops.log(self.cdf(x))

  def entropy(self, name="entropy"):
    """The entropy of Laplace distribution(s).

    Args:
      name: The name to give this op.

    Returns:
      entropy: tensor of dtype `dtype`, the entropy.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._loc, self._scale], name):
        log_2_e = constant_op.constant(math.log(2.) + 1., dtype=self.dtype)
        # Use broadcasting rules to calculate the full broadcast scale.
        scale = self._scale + array_ops.zeros_like(self._loc)
        return log_2_e + math_ops.log(scale)

  def sample_n(self, n, seed=None, name="sample_n"):
    """Sample `n` observations from the Laplace Distributions.

    Args:
      n: `Scalar`, type int32, the number of observations to sample.
      seed: Python integer, the random seed.
      name: The name to give this op.

    Returns:
      samples: `[n, ...]`, a `Tensor` of `n` samples for each
        of the distributions determined by broadcasting the parameters.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._loc, self._scale, n], name):
        n = ops.convert_to_tensor(n)
        n_val = tensor_util.constant_value(n)
        shape = array_ops.concat(0, ([n], self.batch_shape()))
        # Sample uniformly-at-random from the open-interval (-1, 1).
        uniform_samples = random_ops.random_uniform(
            shape=shape,
            minval=np.nextafter(self.dtype.as_numpy_dtype(-1.),
                                self.dtype.as_numpy_dtype(0.)),
            maxval=self.dtype.as_numpy_dtype(1.),
            dtype=self.dtype,
            seed=seed)

        # Provide some hints to shape inference
        inferred_shape = tensor_shape.vector(n_val).concatenate(
            self.get_batch_shape())
        uniform_samples.set_shape(inferred_shape)

        return (self._loc - self._scale * math_ops.sign(uniform_samples) *
                math_ops.log(1. - math_ops.abs(uniform_samples)))

  @property
  def is_reparameterized(self):
    return True

  def _ones(self):
    return array_ops.ones_like(self._loc + self._scale)

  def _zeros(self):
    return array_ops.zeros_like(self._loc + self._scale)

  @property
  def is_continuous(self):
    return True
