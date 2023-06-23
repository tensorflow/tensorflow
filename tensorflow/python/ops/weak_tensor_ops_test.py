# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for TF ops with WeakTensor input."""
from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_ops_list
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import googletest


_TF_UNARY_APIS = weak_tensor_ops_list.ALL_UNARY_OPS
_TF_UNARY_APIS_SPECIFIC_DTYPE = [
    math_ops.to_float,
    math_ops.to_double,
    math_ops.to_int32,
    math_ops.to_int64,
    math_ops.to_bfloat16,
    math_ops.to_complex64,
    math_ops.to_complex128,
]
_TF_UNARY_APIS_WITH_MULT_INPUT = [
    gen_array_ops.check_numerics,
    image_ops_impl.random_brightness,
    image_ops_impl.stateless_random_brightness,
    image_ops_impl.adjust_brightness,
    clip_ops.clip_by_value,
    np_array_ops.expand_dims,
    np_array_ops.moveaxis,
    np_array_ops.reshape,
    np_array_ops.swapaxes,
    array_ops.depth_to_space,
    array_ops.depth_to_space_v2,
    array_ops.expand_dims,
    array_ops.expand_dims_v2,
    array_ops.extract_image_patches,
    array_ops.extract_image_patches_v2,
    array_ops.space_to_depth,
    array_ops.space_to_depth_v2,
]
_TF_UNARY_APIS_WITH_INT_INPUT = [
    gen_bitwise_ops.invert,
    np_math_ops.bitwise_not,
    np_array_ops.arange,
]
_TF_UNARY_APIS_WITH_2D_INPUT = [
    np_math_ops.trace,
    np_array_ops.diagonal,
    np_array_ops.flip,
    np_array_ops.fliplr,
    np_array_ops.flipud,
    np_array_ops.rot90,
    np_array_ops.triu,
    math_ops.trace,
    array_ops.matrix_diag,
    array_ops.matrix_diag_part,
    array_ops.matrix_transpose,
    array_ops.tensor_diag_part,
    array_ops.transpose,
    array_ops.transpose_v2,
]


class WeakTensorOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  # Test unary ops with one input.
  @parameterized.parameters(
      set(_TF_UNARY_APIS) - set(_TF_UNARY_APIS_WITH_MULT_INPUT)
  )
  def test_unary_ops_return_weak_tensor(self, unary_api):
    op_input = _get_test_input(unary_api)
    res = unary_api(op_input)
    # Check that WeakTensor is returned.
    self.assertIsInstance(res, WeakTensor)
    # Check that the actual result is correct.
    expected_result = unary_api(op_input.tensor)
    self.assertAllEqual(res, expected_result)

  # Test unary ops with multiple inputs.
  def test_multi_arg_unary_ops_return_weak_tensor(self):
    a = WeakTensor(constant_op.constant([1, 2, 3], dtypes.float32))
    self.assertIsInstance(
        gen_array_ops.check_numerics(a, message=""), WeakTensor
    )
    self.assertIsInstance(image_ops_impl.random_brightness(a, 0.2), WeakTensor)
    self.assertIsInstance(
        image_ops_impl.stateless_random_brightness(
            image=a, max_delta=0.2, seed=(1, 2)
        ),
        WeakTensor,
    )
    self.assertIsInstance(
        image_ops_impl.adjust_brightness(a, delta=0.2), WeakTensor
    )
    self.assertIsInstance(
        clip_ops.clip_by_value(a, clip_value_min=1.1, clip_value_max=2.2),
        WeakTensor,
    )
    self.assertIsInstance(np_array_ops.expand_dims(a, axis=0), WeakTensor)
    self.assertIsInstance(
        np_array_ops.moveaxis(a, source=0, destination=0), WeakTensor
    )
    self.assertIsInstance(np_array_ops.reshape(a, newshape=(3,)), WeakTensor)
    self.assertIsInstance(
        np_array_ops.swapaxes(a, axis1=0, axis2=0), WeakTensor
    )
    self.assertIsInstance(array_ops.expand_dims(a, axis=0), WeakTensor)

  # Test unary ops with a specific return dtype.
  @parameterized.parameters(_TF_UNARY_APIS_SPECIFIC_DTYPE)
  def test_unary_ops_return_normal_tensor(self, unary_api_specific_dtype):
    a = WeakTensor(constant_op.constant([1, 2, 3], dtypes.float32))
    res = unary_api_specific_dtype(a)
    self.assertIsInstance(res, ops.Tensor)

  # Test unary ops with optional dtype arg.
  def test_elementwise_unary_ops_optional_dtype(self):
    a = WeakTensor(constant_op.constant([1, 2, 3], dtypes.float32))
    # No dtype specified in the argument.
    self.assertIsInstance(array_ops.zeros_like(a), WeakTensor)
    self.assertIsInstance(array_ops.ones_like(a), WeakTensor)
    self.assertIsInstance(array_ops.ones_like(a, dtype=None), WeakTensor)

    # dtype specified in the argument.
    self.assertIsInstance(
        array_ops.zeros_like(a, dtype=dtypes.int32), ops.Tensor
    )
    self.assertIsInstance(
        array_ops.ones_like(a, dtype=dtypes.int32), ops.Tensor
    )
    self.assertIsInstance(array_ops.zeros_like(a, dtypes.int32), ops.Tensor)
    self.assertIsInstance(array_ops.ones_like(a, dtypes.int32), ops.Tensor)
    self.assertIsInstance(
        np_array_ops.arange(
            WeakTensor(constant_op.constant(5)), 0, 1, dtypes.float32
        ),
        ops.Tensor,
    )

  # Test unary ops that require dtype arg.
  def test_unary_ops_explicit_dtype_return(self):
    a = WeakTensor(constant_op.constant([1, 2, 3], dtypes.float32))
    self.assertIsInstance(math_ops.cast(a, dtypes.int32), ops.Tensor)
    self.assertIsInstance(math_ops.saturate_cast(a, dtypes.int32), ops.Tensor)


def _get_test_input(op):
  if op in _TF_UNARY_APIS_WITH_INT_INPUT:
    return WeakTensor(constant_op.constant(5, dtypes.int32))
  elif op in _TF_UNARY_APIS_WITH_2D_INPUT:
    return WeakTensor(constant_op.constant([[1, 2], [3, 4]], dtypes.int32))
  else:
    return WeakTensor(constant_op.constant([1, 2, 3], dtypes.float32))


if __name__ == "__main__":
  ops.enable_eager_execution()
  # Enabling numpy behavior adds some NumPy methods to the Tensor class, which
  # TF-NumPy ops depend on.
  np_config.enable_numpy_behavior()
  googletest.main()
