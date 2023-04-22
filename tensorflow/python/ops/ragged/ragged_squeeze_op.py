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
"""Operator Squeeze for RaggedTensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor


def squeeze(input, axis=None, name=None):  # pylint: disable=redefined-builtin
  """Ragged compatible squeeze.

  If `input` is a `tf.Tensor`, then this calls `tf.squeeze`.

  If `input` is a `tf.RaggedTensor`, then this operation takes `O(N)` time,
  where `N` is the number of elements in the squeezed dimensions.

  Args:
    input: A potentially ragged tensor. The input to squeeze.
    axis: An optional list of ints. Defaults to `None`. If the `input` is
      ragged, it only squeezes the dimensions listed. It fails if `input` is
      ragged and axis is []. If `input` is not ragged it calls tf.squeeze. Note
      that it is an error to squeeze a dimension that is not 1. It must be in
      the range of [-rank(input), rank(input)).
   name: A name for the operation (optional).

  Returns:
    A potentially ragged tensor. Contains the same data as input,
    but has one or more dimensions of size 1 removed.
  """
  with ops.name_scope(name, 'RaggedSqueeze', [input]):
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)
    if isinstance(input, ops.Tensor):
      return array_ops.squeeze(input, axis, name)

    if axis is None:
      raise ValueError('Ragged.squeeze must have an axis argument.')
    if isinstance(axis, int):
      axis = [axis]
    elif ((not isinstance(axis, (list, tuple))) or
          (not all(isinstance(d, int) for d in axis))):
      raise TypeError('Axis must be a list or tuple of integers.')

    dense_dims = []
    ragged_dims = []
    # Normalize all the dims in axis to be positive
    axis = [
        array_ops.get_positive_axis(d, input.shape.ndims, 'axis[%d]' % i,
                                    'rank(input)') for i, d in enumerate(axis)
    ]
    for dim in axis:
      if dim > input.ragged_rank:
        dense_dims.append(dim - input.ragged_rank)
      else:
        ragged_dims.append(dim)

    # Make sure the specified ragged dimensions are squeezable.
    assertion_list = []
    scalar_tensor_one = constant_op.constant(1, dtype=input.row_splits.dtype)
    for i, r in enumerate(input.nested_row_lengths()):
      if i + 1 in ragged_dims:
        assertion_list.append(
            control_flow_ops.Assert(
                math_ops.reduce_all(math_ops.equal(r, scalar_tensor_one)),
                ['the given axis (axis = %d) is not squeezable!' % (i + 1)]))
    if 0 in ragged_dims:
      scalar_tensor_two = constant_op.constant(2, dtype=dtypes.int32)
      assertion_list.append(
          control_flow_ops.Assert(
              math_ops.equal(
                  array_ops.size(input.row_splits), scalar_tensor_two),
              ['the given axis (axis = 0) is not squeezable!']))

    # Till now, we are sure that the ragged dimensions are squeezable.
    squeezed_rt = None
    squeezed_rt = control_flow_ops.with_dependencies(assertion_list,
                                                     input.flat_values)

    if dense_dims:
      # Gives error if the dense dimension is not squeezable.
      squeezed_rt = array_ops.squeeze(squeezed_rt, dense_dims)

    remaining_row_splits = []
    remaining_row_splits = list()
    for i, row_split in enumerate(input.nested_row_splits):
      # each row_splits tensor is for dimension #(i+1) .
      if (i + 1) not in ragged_dims:
        remaining_row_splits.append(row_split)
    # Take care of the first row if it is to be squeezed.
    if remaining_row_splits and 0 in ragged_dims:
      remaining_row_splits.pop(0)

    squeezed_rt = RaggedTensor.from_nested_row_splits(squeezed_rt,
                                                      remaining_row_splits)

    # Corner case: when removing all the ragged dimensions and the output is
    # a scalar tensor e.g. ragged.squeeze(ragged.constant([[[1]]])).
    if set(range(0, input.ragged_rank + 1)).issubset(set(ragged_dims)):
      squeezed_rt = array_ops.squeeze(squeezed_rt, [0], name)

    return squeezed_rt
