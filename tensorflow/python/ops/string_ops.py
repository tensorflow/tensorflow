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

"""## Hashing

String hashing ops take a string input tensor and map each element to an
integer.

@@string_to_hash_bucket_fast
@@string_to_hash_bucket_strong
@@string_to_hash_bucket

## Joining

String joining ops concatenate elements of input string tensors to produce a new
string tensor.

@@reduce_join
@@string_join

## Splitting

@@string_split

## Conversion

@@as_string
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import six

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util

# pylint: disable=unused-import
from tensorflow.python.ops import gen_string_ops
# pylint: enable=unused-import
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_string_ops import *
# pylint: enable=wildcard-import


def string_split(source, delimiter=" "):  # pylint: disable=invalid-name
  """Split elements of `source` based on `delimiter` into a `SparseTensor`.

  Let N be the size of source (typically N will be the batch size). Split each
  element of `source` based on `delimiter` and return a `SparseTensor`
  containing the splitted tokens. Empty tokens are ignored.

  If `delimiter` is an empty string, each element of the `source` is split
  into individual 1 character strings.

  For example:
  N = 2, source[0] is 'hello world' and source[1] is 'a b c', then the output
  will be

  st.indices = [0, 0;
                0, 1;
                1, 0;
                1, 1;
                1, 2]
  st.shape = [2, 3]
  st.values = ['hello', 'world', 'a', 'b', 'c']

  Args:
    source: `1-D` string `Tensor`, the strings to split.
    delimiter: `0-D` string `Tensor`, the delimiter character, the string should
      be length 0 or 1.

  Returns:
    A `SparseTensor` of rank `2`, the strings split according to the delimiter.
    The first column of the indices corresponds to the row in `source` and the
    second column corresponds to the index of the split component in this row.

  Raises:
    ValueError: If delimiter is not a character.
  """
  if isinstance(delimiter, six.string_types) and len(delimiter) > 1:
    raise ValueError("delimiter must be a character, got %s" % delimiter)
  delimiter = ops.convert_to_tensor(delimiter, dtype=dtypes.string)
  source = ops.convert_to_tensor(source, dtype=dtypes.string)

  # pylint: disable=protected-access
  indices, values, shape = gen_string_ops._string_split(
      source, delimiter=delimiter)
  # pylint: enable=protected-access
  indices.set_shape([None, 2])
  values.set_shape([None])
  shape.set_shape([2])
  return ops.SparseTensor(indices, values, shape)


ops.NoGradient("StringToHashBucket")
ops.NoGradient("StringToHashBucketFast")
ops.NoGradient("StringToHashBucketStrong")
ops.NoGradient("ReduceJoin")
ops.NoGradient("StringJoin")
ops.NoGradient("StringSplit")
ops.NoGradient("AsString")

ops.RegisterShape("StringToHashBucket")(common_shapes.unchanged_shape)
ops.RegisterShape("StringToHashBucketFast")(common_shapes.unchanged_shape)
ops.RegisterShape("StringToHashBucketStrong")(common_shapes.unchanged_shape)
ops.RegisterShape("AsString")(common_shapes.unchanged_shape)


@ops.RegisterShape("ReduceJoin")
def _ReduceJoinShape(op):
  """Shape function for the ReduceJoin op."""
  reduction_indices = tensor_util.constant_value(op.inputs[1])
  if reduction_indices is None:
    return [tensor_shape.unknown_shape()]

  input_shape = op.inputs[0].get_shape()
  keep_dims = op.get_attr("keep_dims")

  if input_shape.ndims is None:
    return [tensor_shape.unknown_shape()]

  if input_shape.ndims == 0:
    raise ValueError("Input string tensor cannot be a scalar.")

  true_indices = set()
  for reduction_index in np.ravel(reduction_indices):
    if (reduction_index < -input_shape.ndims or
        reduction_index >= input_shape.ndims):
      raise ValueError("Invalid reduction dimension %d for input with %d "
                       "dimensions" % (reduction_index, input_shape.ndims))

    true_index = reduction_index % input_shape.ndims
    if true_index in true_indices:
      raise ValueError("Duplicate reduction index %d." % reduction_index)

    if input_shape.dims[true_index] == 0:
      raise ValueError("Cannot reduce dimension %d with size 0." %
                       reduction_index)

    true_indices.add(true_index)

  returned_dims = []
  reduce_all = reduction_indices.size == 0
  for i, dim in enumerate(input_shape.dims):
    if reduce_all or i in true_indices:
      if keep_dims:
        returned_dims.append(1)
    else:
      returned_dims.append(dim)

  return [tensor_shape.TensorShape(returned_dims)]


@ops.RegisterShape("StringJoin")
def _StringJoinShape(op):
  """Shape function for the StringJoin op."""
  input_shapes = [x.get_shape() for x in op.inputs]

  # First check if all inputs are scalars.  In the next section
  # we may have *some* scalars and we will be broadcasting them
  if all([s.ndims == 0 for s in input_shapes]):
    return [tensor_shape.scalar()]

  base_shape = tensor_shape.unknown_shape()
  for shape in input_shapes:
    if shape.ndims != 0:
      base_shape = base_shape.merge_with(shape)
  return [base_shape]


@ops.RegisterShape("StringSplit")
def _StringSplitShape(op):
  """Shape function for string_ops.string_split."""
  unused_sfs_shape = op.inputs[0].get_shape().with_rank(1)
  unused_sfs_shape = op.inputs[1].get_shape().merge_with(tensor_shape.scalar())

  indices_shape = tensor_shape.TensorShape([None, 2])
  values_shape = tensor_shape.TensorShape([None])
  shape_shape = tensor_shape.TensorShape([2])
  return [indices_shape, values_shape, shape_shape]
