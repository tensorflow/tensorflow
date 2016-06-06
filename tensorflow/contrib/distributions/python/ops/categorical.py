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
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


# TODO(ysulsky): Move batch_index into array_ops.
def batch_index(vectors, indices, name=None):
  """Indexes into a batch of vectors.

  Args:
    vectors: An N-D Tensor.
    indices: A K-D integer Tensor, K <= N. The first K - 1 dimensions of indices
        must be broadcastable to the first N - 1 dimensions of vectors.
    name: A name for this operation (optional).

  Returns:
    An N-D Tensor comprised of one element selected from each of the vectors.

  Example usage:
    vectors = [[[1, 2, 3], [4, 5, 6]],
               [[7, 8, 9], [1, 2, 3]]]

    batch_index(vectors, 0)
    => [[1, 4],
        [7, 1]]

    batch_index(vectors, [0])
    => [[[1], [4]],
        [[7], [1]]]

    batch_index(vectors, [0, 0, 2, 2])
    => [[[1, 1, 3, 3], [4, 4, 6, 6]],
        [[7, 7, 9, 9], [1, 1, 3, 3]]]

    batch_index(vectors, [[0, 0, 2, 2], [0, 1, 2, 0]])
    => [[[1, 1, 3, 3], [4, 5, 6, 4]],
        [[7, 7, 9, 9], [1, 2, 3, 1]]]
  """
  with ops.op_scope([vectors, indices], name, "BatchIndex"):
    vectors = ops.convert_to_tensor(vectors, name="vectors")
    vectors_shape = array_ops.shape(vectors)
    vectors_rank = array_ops.size(vectors_shape)

    indices = ops.convert_to_tensor(indices, name="indices")
    indices_shape = array_ops.shape(indices)
    indices_rank = array_ops.size(indices_shape)

    # Support scalar indices.
    indices_are_scalar = None
    indices_are_scalar_tensor = math_ops.equal(0, indices_rank)
    if indices.get_shape().ndims is not None:
      indices_are_scalar = indices.get_shape().ndims == 0

    if indices_are_scalar is None:
      indices, num_selected = control_flow_ops.cond(
          indices_are_scalar_tensor,
          lambda: [array_ops.expand_dims(indices, 0),  # pylint: disable=g-long-lambda
                   array_ops.constant(1, dtype=indices_shape.dtype)],
          lambda: [indices, array_ops.gather(indices_shape, indices_rank - 1)])
    elif indices_are_scalar:
      num_selected = 1
      indices = array_ops.expand_dims(indices, 0)
    else:
      num_selected = array_ops.gather(indices_shape, indices_rank - 1)

    # The batch shape is the first N-1 dimensions of `vectors`.
    batch_shape = array_ops.slice(
        vectors_shape, [0], array_ops.pack([vectors_rank - 1]))
    batch_size = math_ops.reduce_prod(batch_shape)

    # Broadcast indices to have shape `batch_shape + [num_selected]`
    bcast_shape = array_ops.concat(0, [batch_shape, [1]])
    bcast_indices = indices + array_ops.zeros(bcast_shape, dtype=indices.dtype)

    # At this point, the first N-1 dimensions of `vectors` and
    # `bcast_indices` agree, and we're almost ready to call
    # `gather_nd`. But first we need to assign each index to a batch,
    # and we do that below by counting up to `batch_size`, repeating
    # each element `num_selected` times.
    batch_count = array_ops.tile(
        array_ops.expand_dims(math_ops.range(batch_size), 1),
        array_ops.pack([1, num_selected]))
    batch_count.set_shape([vectors.get_shape()[:-1].num_elements(),
                           indices.get_shape()[-1]])

    # Flatten the batch dimensions and gather.
    nd_indices = array_ops.concat(
        1, [array_ops.reshape(batch_count, [-1, 1]),
            array_ops.reshape(bcast_indices, [-1, 1])])
    nd_batches = array_ops.reshape(vectors, array_ops.pack([batch_size, -1]))
    ret = array_ops.gather_nd(nd_batches, nd_indices)

    # Reshape the output.
    if indices_are_scalar is None:
      ret = control_flow_ops.cond(
          indices_are_scalar_tensor,
          lambda: array_ops.reshape(ret, batch_shape),
          lambda: array_ops.reshape(  # pylint: disable=g-long-lambda
              ret,
              array_ops.concat(
                  0, [batch_shape, array_ops.expand_dims(num_selected, 0)])))
    elif indices_are_scalar:
      ret = array_ops.reshape(ret, batch_shape)
      ret.set_shape(vectors.get_shape()[:-1])
    else:
      ret = array_ops.reshape(
          ret,
          array_ops.concat(
              0, [batch_shape, array_ops.expand_dims(num_selected, 0)]))
      ret.set_shape(vectors.get_shape()[:-1]
                    .concatenate(indices.get_shape()[-1:]))
    return ret


class Categorical(distribution.DiscreteDistribution):
  """Categorical distribution.

  The categorical distribution is parameterized by the log-probabilities
  of a set of classes.

  Note, the following methods of the base class aren't implemented:
    * mean
    * cdf
    * log_cdf
  """

  def __init__(self, logits, name="Categorical"):
    """Initialize Categorical distributions using class log-probabilities.

    Args:
      logits: An N-D `Tensor` representing the log probabilities of a set of
          Categorical distributions. The first N - 1 dimensions index into a
          batch of independent distributions and the last dimension indexes
          into the classes.
      name: A name for this distribution (optional).
    """
    self._name = name
    with ops.op_scope([logits], name):
      self._logits = ops.convert_to_tensor(logits, name="logits")
      self._histogram = math_ops.exp(self._logits, name="histogram")
      logits_shape = array_ops.shape(self._logits)
      self._batch_rank = array_ops.size(logits_shape) - 1
      self._batch_shape = array_ops.slice(
          logits_shape, [0], array_ops.pack([self._batch_rank]))
      self._num_classes = array_ops.gather(logits_shape, self._batch_rank)

  @property
  def name(self):
    return self._name

  @property
  def dtype(self):
    return dtypes.int64

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

  def pmf(self, k, name="pmf"):
    """Probability of class `k`.

    Args:
      k: `int32` or `int64` Tensor.
      name: A name for this operation (optional).

    Returns:
      The probabilities of the classes indexed by `k`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self._histogram, k], name):
        k = ops.convert_to_tensor(k, name="k")
        return batch_index(self._histogram, k)

  def log_pmf(self, k, name="log_pmf"):
    """Log-probability of class `k`.

    Args:
      k: `int32` or `int64` Tensor.
      name: A name for this operation (optional).

    Returns:
      The log-probabilities of the classes indexed by `k`
    """
    with ops.name_scope(self.name):
      with ops.op_scope([self.logits, k], name):
        k = ops.convert_to_tensor(k, name="k")
        return batch_index(self.logits, k)

  def sample(self, n, seed=None, name="sample"):
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
        ret = array_ops.reshape(
            array_ops.transpose(samples),
            array_ops.concat(
                0, [array_ops.expand_dims(n, 0), self.batch_shape()]))
        ret.set_shape(tensor_shape.vector(tensor_util.constant_value(n))
                      .concatenate(self.get_batch_shape()))
        return ret

  def entropy(self, name="sample"):
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        ret = -math_ops.reduce_sum(
            self._histogram * self._logits,
            array_ops.pack([self._batch_rank]))
        ret.set_shape(self.get_batch_shape())
        return ret

  def mode(self, name="mode"):
    with ops.name_scope(self.name):
      with ops.op_scope([], name):
        ret = math_ops.argmax(self.logits, dimension=self._batch_rank)
        ret.set_shape(self.get_batch_shape())
        return ret
