# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Tokenize text ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops

# The maximum number of bits that can be encoded by create_feature_bitmask
# in each datatype.
_max_bits = {
    dtypes.uint8: 8,
    dtypes.int8: 7,
    dtypes.uint16: 16,
    dtypes.int16: 15,
    dtypes.int32: 31,
    dtypes.int64: 63,
}


def create_feature_bitmask(tensor, dtype=dtypes.int32, name=None):
  """Packs the innermost dimension of a boolean tensor into integer values.

  `result[i1...iN]` is the integer formed by interpreting the booleans
  `tensor[i1...iN, 0:num_bits]` as individual bits, with big-endian order.
  E.g., if `tensor[i1...iN, 0:num_bits] = [True, False, False, True, False]`,
  then `result[i1...iN] = 0b10010 = 18`.  The return tensor is of type `dtype`,
  if specified; if `dtype` is not set, `int32` will be used.

  If `num_bits` is too large to fit in `dtype`, then an exception is raised
  when this op is called (if `num_bits` is statically known) or when it is
  evaluated (if `num_bits` is not statically known).

  Args:
    tensor: `<bool>[D1...DN, num_bits]` The boolean tensor whose innermost
      dimension should be packed to form integer values.
    dtype: The datatype to output for this op (optional).
    name: The name for this op (optional).

  Returns:
    `<dtype> [D1...DN]`
      An integer tensor formed by interpreting the innermost dimension of
      `tensor` as individual bits.

  Raises:
    ValueError: If the data to be packed is too large for the chosen data
      type.
    ValueError: If the data to be packed is not boolean.
    InvalidArgumentError: If the input tensor is a list, or the dtype is not a
      supported integer type.

  Examples:

  >>> assert create_feature_bitmask([True, False, False, True]) == 0b1001
  >>> create_feature_bitmask([[True, False], [False, True], [True, True]])
  <tf.Tensor: shape=(3,), dtype=int32, numpy=array([2, 1, 3], dtype=int32)>
  """
  with ops.name_scope(name, 'CreateFeatureBitmask', [tensor]):
    if (isinstance(tensor, (list, tuple)) and tensor and
        isinstance(tensor[0], tensor_lib.Tensor)):
      raise errors.InvalidArgumentError(
          None, None,
          'CreateFeatureBitmask does not support lists of tensors. Consider '
          'using tf.stack(list,-1) to create a single tensor before invoking '
          'this op.')

    tensor = ops.convert_to_tensor(tensor, dtypes.bool, 'tensor')

    if dtype not in _max_bits.keys():
      raise errors.InvalidArgumentError(
          None, None, 'dtype must be one of: [%s], was %s' %
          (sorted(_max_bits.items(), key=lambda kv: kv[1]), dtype.name))

    integer_data = math_ops.cast(tensor, dtype=dtype)
    shape = tensor.shape
    if shape.ndims is not None and shape.dims[-1].value is not None:
      num_bits = shape.dims[-1].value
      if num_bits > 63:
        raise ValueError(
            'data.shape[-1] must be less than 64, is %d.' % num_bits)
      elif num_bits > _max_bits[dtype]:
        raise ValueError(
            'data.shape[-1] is too large for %s (was %d, cannot exceed %d); '
            'consider switching condense_boolean_tensor to a larger '
            'dtype.' % (dtype.name, num_bits, _max_bits[dtype]))
      bit_masks = constant_op.constant(
          [2**pos for pos in range(num_bits - 1, -1, -1)], dtype)
    else:
      bit_masks = constant_op.constant(
          [2**pos for pos in range(_max_bits[dtype] - 1, -1, -1)], dtype)
      num_bits = array_ops.shape(tensor)[-1]
      with ops.control_dependencies([
          check_ops.assert_less_equal(
              num_bits,
              _max_bits[dtype],
              message='data.shape[-1] is too large for %s (cannot exceed %s)' %
              (dtype.name, _max_bits[dtype]))
      ]):
        # The second slice ("[:num_bits]") is a no-op unless num_bits==0.
        bit_masks = bit_masks[-num_bits:][:num_bits]
    return math_ops.reduce_sum(integer_data * bit_masks, axis=-1)
