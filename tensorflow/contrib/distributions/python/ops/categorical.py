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
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
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
      validate_args=False,
      allow_nan_stats=True,
      name="Categorical"):
    """Initialize Categorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor`, `N >= 1`, representing the log probabilities
          of a set of Categorical distributions. The first `N - 1` dimensions
          index into a batch of independent distributions and the last dimension
          indexes into the classes.
      dtype: The type of the event samples (default: int32).
      validate_args: Unused in this distribution.
      allow_nan_stats: `Boolean`, default `True`.  If `False`, raise an
        exception if a statistic (e.g. mean/mode/etc...) is undefined for any
        batch member.  If `True`, batch members with valid parameters leading to
        undefined statistics will return NaN for this statistic.
      name: A name for this distribution (optional).
    """
    with ops.name_scope(name, values=[logits]) as ns:
      self._logits = ops.convert_to_tensor(logits, name="logits")

      logits_shape_static = self._logits.get_shape().with_rank_at_least(1)
      if logits_shape_static.ndims is not None:
        self._batch_rank = ops.convert_to_tensor(
            logits_shape_static.ndims - 1,
            dtype=dtypes.int32,
            name="batch_rank")
      else:
        with ops.name_scope(name="batch_rank"):
          self._batch_rank = array_ops.rank(self._logits) - 1

      logits_shape = array_ops.shape(self._logits, name="logits_shape")
      if logits_shape_static[-1].value is not None:
        self._num_classes = ops.convert_to_tensor(
            logits_shape_static[-1].value,
            dtype=dtypes.int32,
            name="num_classes")
      else:
        self._num_classes = array_ops.gather(logits_shape,
                                             self._batch_rank,
                                             name="num_classes")

      if logits_shape_static[:-1].is_fully_defined():
        self._batch_shape_val = constant_op.constant(
            logits_shape_static[:-1].as_list(),
            dtype=dtypes.int32,
            name="batch_shape")
      else:
        with ops.name_scope(name="batch_shape"):
          self._batch_shape_val = logits_shape[:-1]
      super(Categorical, self).__init__(
          dtype=dtype,
          parameters={"logits": self._logits, "num_classes": self._num_classes},
          is_continuous=False,
          is_reparameterized=False,
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=ns)

  @property
  def num_classes(self):
    """Scalar `int32` tensor: the number of classes."""
    return self._num_classes

  @property
  def logits(self):
    return self._logits

  def _batch_shape(self):
    # Use identity to inherit callers "name".
    return array_ops.identity(self._batch_shape_val)

  def _get_batch_shape(self):
    return self.logits.get_shape()[:-1]

  def _event_shape(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _get_event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
    if self.logits.get_shape().ndims == 2:
      logits_2d = self.logits
    else:
      logits_2d = array_ops.reshape(self.logits, [-1, self.num_classes])
    samples = random_ops.multinomial(logits_2d, n, seed=seed)
    samples = math_ops.cast(samples, self.dtype)
    ret = array_ops.reshape(
        array_ops.transpose(samples),
        array_ops.concat(0, ([n], self.batch_shape())))
    return ret

  def _log_prob(self, k):
    k = ops.convert_to_tensor(k, name="k")
    if self.logits.get_shape()[:-1] == k.get_shape():
      logits = self.logits
    else:
      logits = self.logits * array_ops.ones_like(
          array_ops.expand_dims(k, -1), dtype=self.logits.dtype)
      logits_shape = array_ops.shape(logits)[:-1]
      k *= array_ops.ones(logits_shape, dtype=k.dtype)
      k.set_shape(tensor_shape.TensorShape(logits.get_shape()[:-1]))
    return -nn_ops.sparse_softmax_cross_entropy_with_logits(logits, k)

  def _prob(self, k):
    return math_ops.exp(self._log_prob(k))

  def _entropy(self):
    if self.logits.get_shape().ndims == 2:
      logits_2d = self.logits
    else:
      logits_2d = array_ops.reshape(self.logits, [-1, self.num_classes])
    histogram_2d = nn_ops.softmax(logits_2d)
    ret = array_ops.reshape(
        nn_ops.softmax_cross_entropy_with_logits(logits_2d, histogram_2d),
        self.batch_shape())
    ret.set_shape(self.get_batch_shape())
    return ret

  def _mode(self):
    ret = math_ops.argmax(self.logits, dimension=self._batch_rank)
    ret = math_ops.cast(ret, self.dtype)
    ret.set_shape(self.get_batch_shape())
    return ret
