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

"""Helper functions for creating partitioned variables.

This is a convenient abstraction to partition a large variable across
multiple smaller variables that can be assigned to different devices.

The full variable can be reconstructed by concatenating the smaller variables.
Using partitioned variables instead of a single variable is mostly a
performance choice.  It however also has an impact on:

1. Random initialization, as the random number generator is called once per
   slice
2. Updates, as they happen in parallel across slices

A key design goal is to allow a different graph to repartition a variable
with the same name but different slicings, including possibly no partitions.

TODO(touts): If an initializer provides a seed, the seed must be changed
deterministicaly for each slice, maybe by adding one to it, otherwise each
slice will use the same values.  Maybe this can be done by passing the
slice offsets to the initializer functions.

Typical usage:

```python
# Create a list of partitioned variables with:
vs = create_partitioned_variables(
    <shape>, <slicing>, <initializer>, name=<optional-name>)

# Pass the list as inputs to embedding_lookup for sharded, parallel lookup:
y = embedding_lookup(vs, ids, partition_strategy="div")

# Or fetch the variables in parallel to speed up large matmuls:
z = matmul(x, concat(slice_dim, vs))
```
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import six  # pylint: disable=unused-import

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging

__all__ = ["create_partitioned_variables", "variable_axis_size_partitioner"]


def variable_axis_size_partitioner(
    max_shard_bytes, axis=0, bytes_per_string_element=16, max_shards=None):
  """Get a partitioner for VariableScope to keep shards below `max_shard_bytes`.

  This partitioner will shard a Variable along one axis, attempting to keep
  the maximum shard size below `max_shard_bytes`.  In practice, this is not
  always possible when sharding along only one axis.  When this happens,
  this axis is sharded as much as possible (i.e., every dimension becomes
  a separate shard).

  If the partitioner hits the `max_shards` limit, then each shard may end up
  larger than `max_shard_bytes`. By default `max_shards` equals `None` and no
  limit on the number of shards is enforced.

  One reasonable value for `max_shard_bytes` is `(64 << 20) - 1`, or almost
  `64MB`, to keep below the protobuf byte limit.

  Args:
    max_shard_bytes: The maximum size any given shard is allowed to be.
    axis: The axis to partition along.  Default: outermost axis.
    bytes_per_string_element: If the `Variable` is of type string, this provides
      an estimate of how large each scalar in the `Variable` is.
    max_shards: The maximum number of shards in int created taking precedence
      over `max_shard_bytes`.

  Returns:
    A partition function usable as the `partitioner` argument to
    `variable_scope`, `get_variable`, and `get_partitioned_variable_list`.

  Raises:
    ValueError: If any of the byte counts are non-positive.
  """
  if max_shard_bytes < 1 or bytes_per_string_element < 1:
    raise ValueError(
        "Both max_shard_bytes and bytes_per_string_element must be positive.")
  if max_shards and max_shards < 1:
    raise ValueError(
        "max_shards must be positive.")

  def _partitioner(shape, dtype):
    """Partitioner that partitions shards to have max_shard_bytes total size.

    Args:
      shape: A `TensorShape`.
      dtype: A `DType`.

    Returns:
      A tuple representing how much to slice each axis in shape.

    Raises:
      ValueError: If shape is not a fully defined `TensorShape` or dtype is not
        a `DType`.
    """
    if not isinstance(shape, tensor_shape.TensorShape):
      raise ValueError("shape is not a TensorShape: %s" % shape)
    if not shape.is_fully_defined():
      raise ValueError("shape is not fully defined: %s" % shape)
    if not isinstance(dtype, dtypes.DType):
      raise ValueError("dtype is not a DType: %s" % dtype)

    if dtype.base_dtype == dtypes.string:
      element_size = bytes_per_string_element
    else:
      element_size = dtype.size

    partitions = [1] * shape.ndims
    bytes_per_slice = 1.0 * (
        shape.num_elements() / shape[axis].value) * element_size
    # How many slices can we fit on one shard of size at most max_shard_bytes?
    # At least one slice is required.
    slices_per_shard = max(1, math.floor(max_shard_bytes / bytes_per_slice))
    # How many shards do we need for axis given that each shard fits
    # slices_per_shard slices from a total of shape[axis].value slices?
    axis_shards = int(math.ceil(1.0 * shape[axis].value / slices_per_shard))
    if max_shards:
      axis_shards = min(max_shards, axis_shards)

    partitions[axis] = axis_shards

    return partitions

  return _partitioner


def create_partitioned_variables(
    shape, slicing, initializer, dtype=dtypes.float32,
    trainable=True, collections=None, name=None, reuse=None):
  """Create a list of partitioned variables according to the given `slicing`.

  Currently only one dimension of the full variable can be sliced, and the
  full variable can be reconstructed by the concatenation of the returned
  list along that dimension.

  Args:
    shape: List of integers.  The shape of the full variable.
    slicing: List of integers.  How to partition the variable.
      Must be of the same length as `shape`.  Each value
      indicate how many slices to create in the corresponding
      dimension.  Presently only one of the values can be more than 1;
      that is, the variable can only be sliced along one dimension.

      For convenience, The requested number of partitions does not have to
      divide the corresponding dimension evenly.  If it does not, the
      shapes of the partitions are incremented by 1 starting from partition
      0 until all slack is absorbed.  The adjustment rules may change in the
      future, but as you can save/restore these variables with different
      slicing specifications this should not be a problem.
    initializer: A `Tensor` of shape `shape` or a variable initializer
      function.  If a function, it will be called once for each slice,
      passing the shape and data type of the slice as parameters.  The
      function must return a tensor with the same shape as the slice.
    dtype: Type of the variables. Ignored if `initializer` is a `Tensor`.
    trainable: If True also add all the variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES`.
    collections: List of graph collections keys to add the variables to.
      Defaults to `[GraphKeys.VARIABLES]`.
    name: Optional name for the full variable.  Defaults to
      `"PartitionedVariable"` and gets uniquified automatically.
    reuse: Boolean or `None`; if `True` and name is set, it would reuse
      previously created variables. if `False` it will create new variables.
      if `None`, it would inherit the parent scope reuse.

  Returns:
    A list of Variables corresponding to the slicing.

  Raises:
    ValueError: If any of the arguments is malformed.
  """
  logging.warn(
      "create_partitioned_variables is deprecated.  Use "
      "tf.get_variable with a partitioner set, or "
      "tf.get_partitioned_variable_list, instead.")

  if len(shape) != len(slicing):
    raise ValueError("The 'shape' and 'slicing' of a partitioned Variable "
                     "must have the length: shape: %s, slicing: %s" %
                     (shape, slicing))
  if len(shape) < 1:
    raise ValueError("A partitioned Variable must have rank at least 1: "
                     "shape: %s" % shape)

  # Legacy: we are provided the slicing directly, so just pass it to
  # the partitioner.
  partitioner = lambda **unused_kwargs: slicing

  with variable_scope.variable_op_scope(
      [], name, "PartitionedVariable", reuse=reuse):
    # pylint: disable=protected-access
    partitioned_var = variable_scope._get_partitioned_variable(
        name=None,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        trainable=trainable,
        partitioner=partitioner,
        collections=collections)
    return partitioned_var._get_variable_list()
    # pylint: enable=protected-access
