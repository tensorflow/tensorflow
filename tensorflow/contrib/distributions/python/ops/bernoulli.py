# Copyright 2016 Google Inc. All Rights Reserved.
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
"""The Bernoulli distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class Bernoulli(distribution.DiscreteDistribution):
  """Bernoulli distribution.

  The Bernoulli distribution is parameterized by p, the probability of a
  positive event.

  Note, the following methods of the base class aren't implemented:
    * cdf
    * log_cdf
  """

  def __init__(
      self, p, dtype=dtypes.int32, strict=True, strict_statistics=True,
      name="Bernoulli"):
    """Construct Bernoulli distributions.

    Args:
      p: An N-D `Tensor` representing the probability of a positive
          event. Each entry in the `Tensor` parameterizes an independent
          Bernoulli distribution.
      dtype: dtype for samples. Note that other values will take the dtype of p.
      strict: Whether to assert that `0 <= p <= 1`. If not strict, `log_pmf` may
        return nans.
      strict_statistics:  Boolean, default True.  If True, raise an exception if
        a statistic (e.g. mean/mode/etc...) is undefined for any batch member.
        If False, batch members with valid parameters leading to undefined
        statistics will return NaN for this statistic.
      name: A name for this distribution.
    """
    self._strict_statistics = strict_statistics
    self._name = name
    self._dtype = dtype
    self._strict = strict
    check_op = check_ops.assert_less_equal
    with ops.op_scope([p], name):
      with ops.control_dependencies(
          [check_op(p, 1.), check_op(0., p)] if strict else []):
        p = array_ops.identity(p, name="p")
      self._p = p
      self._q = array_ops.identity(1. - p, name="q")
      self._batch_shape = array_ops.shape(self._p)
      self._event_shape = array_ops.constant([], dtype=dtypes.int32)

  @property
  def strict_statistics(self):
    """Boolean describing behavior when a stat is undefined for batch member."""
    return self._strict_statistics

  @property
  def strict(self):
    """Boolean describing behavior on invalid input."""
    return self._strict

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return self._dtype

  @property
  def is_reparameterized(self):
    return False

  def batch_shape(self, name="batch_shape"):
    with ops.name_scope(self.name):
      with ops.op_scope([self._batch_shape], name):
        return array_ops.identity(self._batch_shape)

  def get_batch_shape(self):
    return self.p.get_shape()

  def event_shape(self, name="event_shape"):
    with ops.name_scope(self.name):
      with ops.op_scope([self._batch_shape], name):
        return array_ops.constant([], dtype=self._batch_shape.dtype)

  def get_event_shape(self):
    return tensor_shape.scalar()

  @property
  def p(self):
    return self._p

  @property
  def q(self):
    """1-p."""
    return self._q

  def pmf(self, event, name="pmf"):
    """Probability mass function.

    Args:
      event: `int32` or `int64` binary Tensor; must be broadcastable with `p`.
      name: A name for this operation.

    Returns:
      The probabilities of the events.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.p, self.q, event], name):
        event = ops.convert_to_tensor(event, name="event")
        event = math_ops.cast(event, self.p.dtype)
        return event * self.p + (1. - event) * self.q

  def log_pmf(self, event, name="log_pmf"):
    """Log of the probability mass function.

    Args:
      event: `int32` or `int64` binary Tensor.
      name: A name for this operation (optional).

    Returns:
      The log-probabilities of the events.
    """
    return super(Bernoulli, self).log_pmf(event, name)

  def sample(self, n, seed=None, name="sample"):
    """Generate `n` samples.

    Args:
      n: scalar.  Number of samples to draw from each distribution.
      seed: Python integer seed for RNG.
      name: name to give to the op.

    Returns:
      samples: a `Tensor` of shape `(n,) + self.batch_shape` with values of type
          `self.dtype`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.p, n], name):
        n = ops.convert_to_tensor(n, name="n")
        p_2d = array_ops.reshape(self.p, array_ops.pack([-1, 1]))
        q_2d = 1. - p_2d
        probs = array_ops.concat(1, [q_2d, p_2d])
        samples = random_ops.multinomial(math_ops.log(probs), n, seed=seed)
        ret = array_ops.reshape(
            array_ops.transpose(samples),
            array_ops.concat(0,
                             [array_ops.expand_dims(n, 0), self.batch_shape()]))
        ret.set_shape(tensor_shape.vector(tensor_util.constant_value(n))
                      .concatenate(self.get_batch_shape()))
        return math_ops.cast(ret, self.dtype)

  def entropy(self, name="entropy"):
    """Entropy of the distribution.

    Args:
      name: Name for the op.

    Returns:
      entropy: `Tensor` of the same type and shape as `p`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.q, self.p], name):
        e = array_ops.constant(
            np.finfo(self.p.dtype.as_numpy_dtype).tiny,
            dtype=self.p.dtype)
        return (-self.q * math_ops.log(self.q + e) - self.p *
                math_ops.log(self.p + e))

  def mean(self, name="mean"):
    """Mean of the distribution.

    Args:
      name: Name for the op.

    Returns:
      mean: `Tensor` of the same type and shape as `p`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.p], name):
        return array_ops.identity(self.p)

  def mode(self, name="mode"):
    """Mode of the distribution.

    1 if p > 1-p. 0 otherwise.

    Args:
      name: Name for the op.

    Returns:
      mode: binary `Tensor` of type self.dtype.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.p, self.q], name):
        return math_ops.cast(self.p > self.q, self.dtype)

  def variance(self, name="variance"):
    """Variance of the distribution.

    Args:
      name: Name for the op.

    Returns:
      variance: `Tensor` of the same type and shape as `p`.
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.p, self.q], name):
        return self.q * self.p

  def std(self, name="std"):
    """Standard deviation of the distribution.

    Args:
      name: Name for the op.

    Returns:
      std: `Tensor` of the same type and shape as `p`.
    """
    with ops.name_scope(self.name):
      with ops.name_scope(name):
        return math_ops.sqrt(self.variance())
