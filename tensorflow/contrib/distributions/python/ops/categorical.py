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
"""The Categorical distribution class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.distributions.python.ops import distribution
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops


class Categorical(distribution.Distribution):
  """Categorical distribution.

  The categorical distribution is parameterized by the log-probabilities
  of a set of classes.
  """

  def __init__(
      self,
      logits,
      dtype=dtypes.int32,
      validate_args=True,
      allow_nan_stats=False,
      name="Categorical"):
    """Initialize Categorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities
          of a set of Categorical distributions. The first `N - 1` dimensions
          index into a batch of independent distributions and the last dimension
          indexes into the classes.
      dtype: The type of the event samples (default: int32).
      validate_args: Unused in this distribution.
      allow_nan_stats:  Boolean, default `False`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: A name for this distribution (optional).
    """
    self._allow_nan_stats = allow_nan_stats
    self._name = name
    self._dtype = dtype
    self._validate_args = validate_args
    with ops.op_scope([logits], name):
      self._logits = ops.convert_to_tensor(logits, name="logits")
      logits_shape = array_ops.shape(self._logits)
      self._batch_rank = array_ops.size(logits_shape) - 1
      self._batch_shape = array_ops.slice(
          logits_shape, [0], array_ops.pack([self._batch_rank]))
      self._num_classes = array_ops.gather(logits_shape, self._batch_rank)

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
    return self._dtype

  @property
  def is_reparameterized(self):
    return False

  def batch_shape(self, name="batch_shape"):
    with ops.name_scope(self.name):
      return array_ops.identity(self._batch_shape, name=name)

  def get_batch_shape(self):
    return self.logits.get_shape()[:-1]

  def event_shape(self, name="event_shape"):
    with ops.name_scope(self.name):
      return array_ops.constant([], dtype=self._batch_shape.dtype, name=name)

  def get_event_shape(self):
    return tensor_shape.scalar()

  @property
  def num_classes(self):
    return self._num_classes

  @property
  def logits(self):
    return self._logits

  def log_prob(self, k, name="log_prob"):
    """Log-probability of class `k`.

    Args:
      k: `int32` or `int64` Tensor. Must be broadcastable with a `batch_shape`
        `Tensor`.
      name: A name for this operation (optional).

    Returns:
      The log-probabilities of the classes indexed by `k`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([k, self.logits], name):
        k = ops.convert_to_tensor(k, name="k")

        logits = self.logits * array_ops.ones_like(
            array_ops.expand_dims(k, -1),
            dtype=self.logits.dtype)
        k *= array_ops.ones(
            array_ops.slice(
                array_ops.shape(logits), [0], [array_ops.rank(logits) - 1]),
            dtype=k.dtype)
        k.set_shape(tensor_shape.TensorShape(logits.get_shape()[:-1]))

        return -nn_ops.sparse_softmax_cross_entropy_with_logits(logits, k)

  def prob(self, k, name="prob"):
    """Probability of class `k`.

    Args:
      k: `int32` or `int64` Tensor. Must be broadcastable with logits.
      name: A name for this operation (optional).

    Returns:
      The probabilities of the classes indexed by `k`
    """
    return super(Categorical, self).prob(k, name)

  def sample_n(self, n, seed=None, name="sample_n"):
    """Sample `n` observations from the Categorical distribution.

    Args:
      n: 0-D.  Number of independent samples to draw for each distribution.
      seed: Random seed (optional).
      name: A name for this operation (optional).

    Returns:
      An `int64` `Tensor` with shape `[n, batch_shape, event_shape]`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.logits, n], name):
        n = ops.convert_to_tensor(n, name="n")
        logits_2d = array_ops.reshape(
            self.logits, array_ops.pack([-1, self.num_classes]))
        samples = random_ops.multinomial(logits_2d, n, seed=seed)
        samples = math_ops.cast(samples, self._dtype)
        ret = array_ops.reshape(
            array_ops.transpose(samples),
            array_ops.concat(0, ([n], self.batch_shape())))
        ret.set_shape(tensor_shape.vector(tensor_util.constant_value(n))
                      .concatenate(self.get_batch_shape()))
        return ret

  def entropy(self, name="sample"):
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        logits_2d = array_ops.reshape(
            self.logits, array_ops.pack([-1, self.num_classes]))
        histogram_2d = nn_ops.softmax(logits_2d)
        ret = array_ops.reshape(
            nn_ops.softmax_cross_entropy_with_logits(logits_2d, histogram_2d),
            self.batch_shape())
        ret.set_shape(self.get_batch_shape())
        return ret

  def mode(self, name="mode"):
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        ret = math_ops.argmax(self.logits, dimension=self._batch_rank)
        ret = math_ops.cast(ret, self._dtype)
        ret.set_shape(self.get_batch_shape())
        return ret

  @property
  def is_continuous(self):
    return False
