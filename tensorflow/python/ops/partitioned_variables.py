# Copyright 2015 Google Inc. All Rights Reserved.
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

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables

__all__ = ["create_partitioned_variables"]

def _compute_slice_dim_and_shape(full_shape, slicing):
  """Computes which dimension is being sliced and the typical slice shape."""

  slice_shape = [0] * len(full_shape)
  slice_dim = None
  for dim, num_slices in enumerate(slicing):
    dim_size = full_shape[dim]
    if num_slices <= 0 or dim_size < num_slices:
      raise ValueError("Cannot create %d slices for size %d. shape: %s, "
                       "slicing: %s" %
                       (num_slices, full_shape[dim], full_shape, slicing))
    if num_slices == 1:
      # Not slicing in this dimension.
      slice_shape[dim] = dim_size
    elif slice_dim is not None:
      # We only support slicing along one of the dimensions.
      raise ValueError("Can only slice a variable along one dimension: "
                       "shape: %s, slicing: %s" % (full_shape, slicing))
    else:
      # Note: We will add any extras onto the last slice, later.
      slice_dim = dim
      slice_shape[dim] = dim_size // num_slices

  # Degenerate case: If "slicing" was all ones, pretend we are slicing along
  # the first dimension.
  if slice_dim is None:
    slice_dim = 0
  return slice_dim, slice_shape


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
  if len(shape) != len(slicing):
    raise ValueError("The 'shape' and 'slicing' of a partitioned Variable "
                     "must have the length: shape: %s, slicing: %s" %
                     (shape, slicing))
  if len(shape) < 1:
    raise ValueError("A partitioned Variable must have rank at least 1: "
                     "shape: %s" % shape)
  full_shape = tensor_shape.as_shape(shape)
  full_shape.assert_is_fully_defined()
  full_shape = full_shape.as_list()

  slice_dim, slice_shape = _compute_slice_dim_and_shape(full_shape, slicing)

  vs = []
  num_slices = slicing[slice_dim]
  num_slices_with_excess = full_shape[slice_dim] % num_slices

  with variable_scope.variable_op_scope([], name,
                                        "PartitionedVariable",
                                        reuse=reuse) as scope:
    full_name = scope.name
    slice_offset = [0] * len(full_shape)
    for i in xrange(num_slices):
      var_shape = slice_shape[:]
      var_offset = slice_offset[:]
      if i < num_slices_with_excess:
        var_shape[slice_dim] += 1
      slice_offset[slice_dim] += var_shape[slice_dim]

      if callable(initializer):
        init = initializer
        init_shape = var_shape
      elif isinstance(initializer, ops.Tensor):
        init = array_ops.slice(initializer, var_offset, var_shape)
        # Use the dtype of the given tensor.
        dtype = init.dtype.base_dtype
        init_shape = None
      else:
        init = ops.convert_to_tensor(initializer, dtype=dtype)
        init = array_ops.slice(init, var_offset, var_shape)
        init_shape = None

      var = variable_scope.get_variable(name="part_%d" % i,
                                        shape=init_shape,
                                        dtype=dtype,
                                        initializer=init,
                                        trainable=trainable,
                                        collections=collections)

      # pylint: disable=protected-access
      var._set_save_slice_info(variables.Variable.SaveSliceInfo(
          full_name, full_shape, var_offset, var_shape))
      # pylint: enable=protected-access
      vs.append(var)
    assert slice_offset[slice_dim] == full_shape[slice_dim]
  return vs
