# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Helper library for sharding during TPU compilation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import tensor_shape

_DEFAULT_NUMBER_OF_SHARDS = 1
_DEFAULT_SHARD_DIMENSION = 0


# TODO(b/36777903) change other parts of tpu.py to use this class.
class ShardingPolicy(object):
  """An object use to hold the sharding policy for a Tensor.
  """

  def __init__(self):
    self._number_of_shards = None
    self._shard_dimension = None
    self._frozen = False

  def __str__(self):
    if self.number_of_shards is None or self.shard_dimension is None:
      return "ShardingPolicy(unset)"
    else:
      return ("ShardingPolicy(%d shards dimension %d)" %
              (self.number_of_shards, self.shard_dimension))

  def _fill_default_values(self):
    if self._number_of_shards is None:
      self._number_of_shards = _DEFAULT_NUMBER_OF_SHARDS
    if self._shard_dimension is None:
      self._shard_dimension = tensor_shape.as_dimension(
          _DEFAULT_SHARD_DIMENSION)

  def freeze(self):
    """Prevents further modification to the sharding policy.

    Any values that have not been set when freeze is called are set to
    defaults. If the ShardingPolicy is already frozen, this is a NoOp.
    """
    if not self._frozen:
      self._fill_default_values()
      self._frozen = True

  @property
  def number_of_shards(self):
    """Returns the number of shards in the policy or None if unspecified."""
    return self._number_of_shards

  def set_number_of_shards(self, number_of_shards):
    """Sets the number of shards for the current policy.

    If the policy has been frozen then number_of_shards must match the
    existing setting.

    Args:
      number_of_shards: The number of shards to use in the policy.

    Raises:
      ValueError: If the policy has been frozen and number_of_shards
        differs from the frozen value; or number_of_shards <= 0.
    """
    if self._frozen:
      if self._number_of_shards != number_of_shards:
        raise ValueError(
            "Can't set sharding policy to use %d shards since it has been "
            "frozen to use %d." % (number_of_shards, self._number_of_shards))
    else:
      if number_of_shards > 0:
        self._number_of_shards = number_of_shards
      else:
        raise ValueError(
            "Can't set sharding policy to use %s shards; value must be >0",
            str(number_of_shards))

  @property
  def shard_dimension(self):
    """Returns the shard dimension of the policy or None if unspecified."""
    return self._shard_dimension

  def set_shard_dimension(self, shard_dimension):
    """Sets the shard dimension for the current policy.

    If the policy has been frozen then shard_dimension must match the
    existing setting.

    Args:
      shard_dimension: The shard dimension to use in the policy.

    Raises:
      ValueError: If the policy has been frozen and shard_dimension
        differs from the frozen value, or shard_dimension can't be
        interpreted as a Dimension.
    """
    if self._frozen:
      if self._shard_dimension != shard_dimension:
        raise ValueError(
            "Can't set shard dimension to %d since it has been frozen to "
            "use %d." % (shard_dimension, self._shard_dimension))
    else:
      self._shard_dimension = tensor_shape.as_dimension(shard_dimension)

  def merge(self, other):
    """Merges the policy of another policy into the current policy.

    Args:
      other: The policy to merge into this one.

    Raises:
      ValueError: If this policy has been frozen and the merge conflicts with
      the frozen policy.
    """
    if other.number_of_shards is not None:
      self.set_number_of_shards(other.number_of_shards)
    if other.shard_dimension is not None:
      self.set_shard_dimension(other.shard_dimension)

  def get_sharded_shape(self, shape, shard_index=None):
    """Returns the shape of a shard of a full Tensor.

    When given the shape of a 'full-size' Tensor, returns the shape of
    the sub-Tensor after it has been sharded. Freezes the policy if it
    has not yet been frozen.

    Args:
      shape: The shape of the full-size Tensor to be sharded.
      shard_index: The index of the shard whose shape should be returned.
        shard_index can be None for sharding policies that use the same
        shape for every shard.
      freeze_config:

    Returns:
      The shape of the sharded version of the Tensor.

    Raises:
      ValueError: If shard_index is None when shards are of different
        shapes; or shard_index is not None and
        !(0<=shard_index<number_of_shards); or shape does not have at
        least self.shard_dimension+1 dimensions; or the value of
        shape's shard dimension is not a multiple of
        self.number_of_shards
    """
    if self._shard_dimension is None or self._number_of_shards is None:
      # Don't raise an error if the config is unset.
      return None
    if shard_index is not None:
      if shard_index < 0 or shard_index >= self.number_of_shards:
        raise ValueError("shard_index %d, but must be in [0,%d)." %
                         (shard_index, self._number_of_shards))
    shape = tensor_shape.as_shape(shape)
    if self._number_of_shards == 1:
      # Don't do anything when there's only one shard.
      return shape
    ndims = shape.ndims
    if ndims is None:
      raise ValueError("shape must be a specified shape not Unknown")
    if ndims <= self._shard_dimension:
      raise ValueError("shape %s does not contain shard_dimension %d" %
                       (shape.as_list(), self._shard_dimension))
    dims = shape.as_list()
    if dims[self._shard_dimension] is None:
      raise ValueError("shape %s must have a fixed size for dimension %d "
                       "that is known at graph construction time." %
                       (shape.as_list(), self._shard_dimension))
    if (dims[self._shard_dimension] % self._number_of_shards) != 0:
      raise ValueError("shape %s cannot be sharded %d ways along dimension %d" %
                       (shape.as_list(), self._number_of_shards,
                        self._shard_dimension))
    dims[self._shard_dimension] /= self._number_of_shards
    return tensor_shape.as_shape(dims)

  def _unshard_shape(self, shape):
    """Return the unsharded shape that would generate a given sharded shape.

    Args:
      shape: the sharded shape to unshard

    Returns:
      The unsharded shape.

    Raises:
      ValueError: if shape is unknown or does not contain
        self.shard_dimension
      TypeError: if shape is not convertible to a TensorShape
    """
    shape = tensor_shape.as_shape(shape)
    if self._number_of_shards == 1:
      # Don't do anything when there's only one shard.
      return shape
    ndims = shape.ndims
    if ndims is None:
      raise ValueError("shape must be a specified shape not Unknown")
    if ndims <= self._shard_dimension:
      raise ValueError("shape %s does not contain shard_dimension %d" %
                       (shape.as_list(), self._shard_dimension))
    dims = shape.as_list()
    dims[self._shard_dimension] *= self._number_of_shards
    return tensor_shape.as_shape(dims)

  def get_unsharded_shape(self, shapes):
    """Returns the shape of an unsharded Tensor given a list of shards.

    When given a list of shapes of shards, returns the shape of the
    unsharded Tensor that would generate the shards. Sets defaults for the
    policy if number_of_shards or shard_dimension is None.

    Args:
      shapes: The shapes of the Tensor shards to be combined.

    Returns:
      The shape of the unsharded version of the Tensor.

    Raises:
      ValueError: if shapes is not a list of length
        self.number_of_shards; or any element of shapes is not a valid
        shape consistent with the sharding policy; or the list of
        shapes is not a valid sharding of a full shape.
      TypeError: if an element of shapes is not convertible to a
        TensorShape
    """
    self._fill_default_values()
    if len(shapes) != self.number_of_shards:
      raise ValueError(
          "shapes is %s but must be a list of length number_of_shards=%d" % (
              str(shapes), self.number_of_shards))
    unsharded_shapes = [self._unshard_shape(s) for s in shapes]
    for i in xrange(self.number_of_shards - 1):
      if not unsharded_shapes[i].is_compatible_with(
          unsharded_shapes[self.number_of_shards - 1]):
        raise ValueError(
            "sharded shapes %s are not consistent shards of a full shape "
            "sharded %d ways along dimension %d" % (
                str(shapes), self.number_of_shards, self.shard_dimension))
    return unsharded_shapes[0]
