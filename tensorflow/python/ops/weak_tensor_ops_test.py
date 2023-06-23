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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_ops_list
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
]
_TF_UNARY_APIS_WITH_INT_INPUT = [
    gen_bitwise_ops.invert,
]


class WeakTensorOpsTest(test_util.TensorFlowTestCase):

  def test_elementwise_unary_ops_return_weak_tensor(self):
    for op in _TF_UNARY_APIS:
      if op in _TF_UNARY_APIS_WITH_MULT_INPUT:
        # Test unary ops with multple inputs separately.
        continue
      op_input = _get_test_input(op)
      res = op(op_input)
      # Check that WeakTensor is returned.
      self.assertIsInstance(res, weak_tensor.WeakTensor)
      # Check that the actual result is correct.
      wt_result = self.evaluate(res)
      t_result = self.evaluate(op(op_input.tensor))
      self.assertAllEqual(wt_result, t_result)

  def test_elementwise_multi_arg_unary_ops_return_weak_tensor(self):
    a = weak_tensor.WeakTensor(constant_op.constant([1, 2, 3], dtypes.float32))
    results = [
        gen_array_ops.check_numerics(a, message=""),
        image_ops_impl.random_brightness(a, 0.2),
        image_ops_impl.stateless_random_brightness(
            image=a, max_delta=0.2, seed=(1, 2)
        ),
        image_ops_impl.adjust_brightness(a, delta=0.2),
        clip_ops.clip_by_value(a, clip_value_min=1.1, clip_value_max=2.2),
    ]
    for result in results:
      self.assertIsInstance(result, weak_tensor.WeakTensor)

  def test_elementwise_unary_ops_return_normal_tensor(self):
    a = weak_tensor.WeakTensor(constant_op.constant([1, 2, 3], dtypes.float32))
    for op in _TF_UNARY_APIS_SPECIFIC_DTYPE:
      res = op(a)
      self.assertNotIsInstance(res, weak_tensor.WeakTensor)

  def test_elementwise_unary_ops_optional_dtype(self):
    a = weak_tensor.WeakTensor(constant_op.constant([1, 2, 3], dtypes.float32))
    # No dtype specified in the argument.
    self.assertIsInstance(array_ops.zeros_like(a), weak_tensor.WeakTensor)
    self.assertIsInstance(array_ops.ones_like(a), weak_tensor.WeakTensor)
    self.assertIsInstance(
        array_ops.ones_like(a, dtype=None), weak_tensor.WeakTensor
    )

    # dtype specified in the argument.
    self.assertIsInstance(
        array_ops.zeros_like(a, dtype=dtypes.int32), ops.Tensor
    )
    self.assertIsInstance(
        array_ops.ones_like(a, dtype=dtypes.int32), ops.Tensor
    )
    self.assertIsInstance(array_ops.zeros_like(a, dtypes.int32), ops.Tensor)
    self.assertIsInstance(array_ops.ones_like(a, dtypes.int32), ops.Tensor)

  def test_unary_ops_explicit_dtype_return(self):
    a = weak_tensor.WeakTensor(constant_op.constant([1, 2, 3], dtypes.float32))
    self.assertIsInstance(math_ops.cast(a, dtypes.int32), ops.Tensor)
    self.assertIsInstance(math_ops.saturate_cast(a, dtypes.int32), ops.Tensor)


def _get_test_input(op):
  if op in _TF_UNARY_APIS_WITH_INT_INPUT:
    return weak_tensor.WeakTensor(constant_op.constant([1, 2, 3], dtypes.int32))
  else:
    return weak_tensor.WeakTensor(
        constant_op.constant([1, 2, 3], dtypes.float32)
    )


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
