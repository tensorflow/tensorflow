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
@@encode_base64
@@decode_base64
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

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


ops.NotDifferentiable("StringToHashBucket")
ops.NotDifferentiable("StringToHashBucketFast")
ops.NotDifferentiable("StringToHashBucketStrong")
ops.NotDifferentiable("ReduceJoin")
ops.NotDifferentiable("StringJoin")
ops.NotDifferentiable("StringSplit")
ops.NotDifferentiable("AsString")
ops.NotDifferentiable("EncodeBase64")
ops.NotDifferentiable("DecodeBase64")

ops.RegisterShape("StringToHashBucket")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("StringToHashBucketFast")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("StringToHashBucketStrong")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("AsString")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("EncodeBase64")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("DecodeBase64")(common_shapes.call_cpp_shape_fn)


@ops.RegisterShape("ReduceJoin")
def _ReduceJoinShape(op):
  return common_shapes.call_cpp_shape_fn(op, input_tensors_needed=[1])


ops.RegisterShape("StringJoin")(common_shapes.call_cpp_shape_fn)
ops.RegisterShape("StringSplit")(common_shapes.call_cpp_shape_fn)
