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
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import weak_tensor_ops
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import dispatch


DtypeConversionTestEnv = weak_tensor_test_util.DtypeConversionTestEnv
_get_weak_tensor = weak_tensor_test_util.get_weak_tensor
_convert_to_input_type = weak_tensor_test_util.convert_to_input_type
_get_test_input_for_binary_op = weak_tensor_test_util.get_test_input_for_op
_DTYPE_PROMO_RES = flexible_dtypes._BINARY_DTYPE_RES_HALF

_TF_UNARY_APIS = weak_tensor_ops._TF_UNARY_APIS
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
    image_ops.random_brightness,
    image_ops.stateless_random_brightness,
    image_ops.adjust_brightness,
    clip_ops.clip_by_value,
    np_array_ops.expand_dims,
    np_array_ops.full_like,
    np_array_ops.moveaxis,
    np_array_ops.repeat,
    np_array_ops.reshape,
    np_array_ops.swapaxes,
    array_ops.reshape,
    array_ops.depth_to_space,
    array_ops.depth_to_space_v2,
    array_ops.expand_dims,
    array_ops.expand_dims_v2,
    array_ops.extract_image_patches,
    array_ops.extract_image_patches_v2,
    array_ops.space_to_depth,
    array_ops.space_to_depth_v2,
    math_ops.scalar_mul,
    math_ops.scalar_mul_v2,
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

all_dtype_promos_list = []
safe_mode_unallowed_promos_list = []
for key in _DTYPE_PROMO_RES:
  if key[0] == dtypes.bool:
    continue
  for k, v in _DTYPE_PROMO_RES[key].items():
    if v[1] == ops.PromoMode.ALL:
      safe_mode_unallowed_promos_list.append((key, k))
    all_dtype_promos_list.append((key, k, v[0]))


class MyTensor(extension_type.ExtensionType):
  value: tensor.Tensor


class WeakTensorUnaryOpsTest(
    test_util.TensorFlowTestCase, parameterized.TestCase
):

  # Test unary ops with one input.
  @parameterized.named_parameters(
      (api.__module__ + "." + api.__name__, api)
      for api in set(_TF_UNARY_APIS) - set(_TF_UNARY_APIS_WITH_MULT_INPUT)
  )
  def test_unary_ops_return_weak_tensor(self, unary_api):
    (
        weak_tensor_input,
        python_input,
        tensor_input_no_dtype_specified,
        tensor_input,
        numpy_input,
    ) = get_test_input_for_unary_op(unary_api)

    # Check that WeakTensor input outputs a WeakTensor.
    res = unary_api(weak_tensor_input)
    self.assertIsInstance(res, WeakTensor)
    expected_result = unary_api(weak_tensor_input.tensor)
    # Check that the actual result is correct.
    self.assertAllEqual(res, expected_result)

    # Check that tf.constant with no dtype specified (weak type) returns a
    # WeakTensor.
    res = unary_api(tensor_input_no_dtype_specified)
    self.assertIsInstance(res, WeakTensor)

    # Check that python nested scalar type (weak type) returns a WeakTensor.
    res = unary_api(python_input)
    self.assertIsInstance(res, WeakTensor)

    # Check that normal Tensor input outputs a Tensor.
    res = unary_api(tensor_input)
    self.assertIsInstance(res, tensor.Tensor)

    # Check that numpy type input outputs a Tensor.
    res = unary_api(numpy_input)
    self.assertIsInstance(res, tensor.Tensor)

  # Test unary ops with multiple inputs.
  @parameterized.parameters(
      ("WeakTensor", dtypes.float32, WeakTensor),
      ("Python", dtypes.float32, WeakTensor),
      ("Tensor", None, WeakTensor),
      ("NumPy", dtypes.float32, tensor.Tensor),
      ("NumPy", None, tensor.Tensor),
      ("Tensor", dtypes.float32, tensor.Tensor),
  )
  def test_multi_arg_unary_ops_return_weak_tensor(
      self, input_type, input_dtype, result_type
  ):
    test_input = _convert_to_input_type(
        [1.0, 2.0, 3.0], input_type, input_dtype
    )
    self.assertIsInstance(
        gen_array_ops.check_numerics(test_input, message=""), result_type
    )
    self.assertIsInstance(
        image_ops.random_brightness(test_input, 0.2), result_type
    )
    self.assertIsInstance(
        image_ops.stateless_random_brightness(
            image=test_input, max_delta=0.2, seed=(1, 2)
        ),
        result_type,
    )
    self.assertIsInstance(
        image_ops.adjust_brightness(test_input, delta=0.2), result_type
    )
    self.assertIsInstance(
        clip_ops.clip_by_value(
            test_input, clip_value_min=1.1, clip_value_max=2.2
        ),
        result_type,
    )
    self.assertIsInstance(
        np_array_ops.expand_dims(test_input, axis=0), result_type
    )
    self.assertIsInstance(
        np_array_ops.moveaxis(test_input, source=0, destination=0), result_type
    )
    self.assertIsInstance(
        np_array_ops.reshape(test_input, newshape=(3,)), result_type
    )
    self.assertIsInstance(
        np_array_ops.swapaxes(test_input, axis1=0, axis2=0), result_type
    )
    self.assertIsInstance(
        array_ops.reshape(test_input, shape=(3,)), result_type
    )
    self.assertIsInstance(
        array_ops.expand_dims(test_input, axis=0), result_type
    )

  # Test unary ops with a specific return dtype.
  @parameterized.parameters(_TF_UNARY_APIS_SPECIFIC_DTYPE)
  def test_unary_ops_return_normal_tensor(self, unary_api_specific_dtype):
    # All inputs should output a normal Tensor because return dtype is
    # specified.
    weak_tensor_input = _get_weak_tensor([1, 2, 3], dtypes.float32)
    res = unary_api_specific_dtype(weak_tensor_input)
    self.assertIsInstance(res, tensor.Tensor)

    python_input = [1.0, 2.0, 3.0]
    res = unary_api_specific_dtype(python_input)
    self.assertIsInstance(res, tensor.Tensor)

    tensor_input_no_dtype_specified = constant_op.constant([1.0, 2.0, 3.0])
    res = unary_api_specific_dtype(tensor_input_no_dtype_specified)
    self.assertIsInstance(res, tensor.Tensor)

    tensor_input = constant_op.constant([1.0, 2.0, 3.0], dtypes.float32)
    res = unary_api_specific_dtype(tensor_input)
    self.assertIsInstance(res, tensor.Tensor)

    tensor_input = np.array([1.0, 2.0, 3.0])
    res = unary_api_specific_dtype(tensor_input)
    self.assertIsInstance(res, tensor.Tensor)

  @test_util.run_in_graph_and_eager_modes
  def test_weak_tensor_from_scalar_in_tf_func(self):
    @def_function.function()
    def f():
      return 1

    res = f()
    self.assertIsInstance(res, WeakTensor)

  # Test unary ops with optional dtype arg.
  @parameterized.parameters(
      ("WeakTensor", dtypes.float32, WeakTensor),
      ("Python", None, WeakTensor),
      ("Tensor", None, WeakTensor),
      ("NumPy", dtypes.float32, tensor.Tensor),
      ("NumPy", None, tensor.Tensor),
      ("Tensor", dtypes.float32, tensor.Tensor),
  )
  def test_elementwise_unary_ops_optional_dtype(
      self, input_type, input_dtype, result_type
  ):
    test_input = _convert_to_input_type(
        [1.0, 2.0, 3.0], input_type, input_dtype
    )
    # No dtype specified in the argument.
    self.assertIsInstance(array_ops.zeros_like(test_input), result_type)
    self.assertIsInstance(array_ops.ones_like(test_input), result_type)
    self.assertIsInstance(
        array_ops.ones_like(test_input, dtype=None), result_type
    )

    # dtype specified in the argument.
    self.assertIsInstance(
        array_ops.zeros_like(test_input, dtype=dtypes.int32), tensor.Tensor
    )
    self.assertIsInstance(
        array_ops.ones_like(test_input, dtype=dtypes.int32), tensor.Tensor
    )
    self.assertIsInstance(
        array_ops.zeros_like(test_input, dtypes.int32), tensor.Tensor
    )
    self.assertIsInstance(
        array_ops.ones_like(test_input, dtypes.int32), tensor.Tensor
    )

  @parameterized.parameters(
      ("WeakTensor", dtypes.float32, None, WeakTensor),
      ("WeakTensor", dtypes.float32, dtypes.int32, tensor.Tensor),
      ("Python", None, None, WeakTensor),
      ("Python", None, dtypes.int32, tensor.Tensor),
      ("NumPy", None, None, tensor.Tensor),
      ("NumPy", None, dtypes.int32, tensor.Tensor),
      ("Tensor", None, None, WeakTensor),
      ("Tensor", None, dtypes.int32, tensor.Tensor),
      ("Tensor", dtypes.float32, None, tensor.Tensor),
      ("Tensor", dtypes.float32, dtypes.int32, tensor.Tensor),
  )
  # Test unary ops with multiple args that includes an optional dtype arg.
  def test_elementwise_unary_ops_optional_dtype_with_multi_args(
      self, input_type, input_dtype, dtype_arg, result_type
  ):
    test_input = _convert_to_input_type(5, input_type, input_dtype)
    self.assertIsInstance(
        np_array_ops.arange(test_input, 10, dtype=dtype_arg), result_type
    )
    self.assertIsInstance(
        np_array_ops.full_like(test_input, 1, dtype=dtype_arg), result_type
    )

  # Test unary ops that require dtype arg.
  def test_unary_ops_explicit_dtype_return(self):
    wt_input = _get_weak_tensor([1, 2, 3], dtypes.float32)
    self.assertIsInstance(math_ops.cast(wt_input, dtypes.int32), tensor.Tensor)
    self.assertIsInstance(
        math_ops.saturate_cast(wt_input, dtypes.int32), tensor.Tensor
    )

    python_input = [1.0, 2.0, 3.0]
    self.assertIsInstance(
        math_ops.cast(python_input, dtypes.int32), tensor.Tensor
    )
    self.assertIsInstance(
        math_ops.saturate_cast(python_input, dtypes.int32), tensor.Tensor
    )

    tensor_no_dtype_specified = constant_op.constant([1.0, 2.0, 3.0])
    self.assertIsInstance(
        math_ops.cast(tensor_no_dtype_specified, dtypes.int32),
        tensor.Tensor,
    )
    self.assertIsInstance(
        math_ops.saturate_cast(tensor_no_dtype_specified, dtypes.int32),
        tensor.Tensor,
    )

  def test_unsupported_input_type_in_weak_tensor_ops(self):
    rt = ragged_tensor.RaggedTensor.from_row_splits(
        values=[3, 1, 4, 1, 5, 9, 2, 6], row_splits=[0, 4, 4, 7, 8, 8]
    )
    # Any unsupported type should be ignored in WeakTensor wrapper.
    self.assertIsInstance(math_ops.abs(rt), ragged_tensor.RaggedTensor)

  def test_update_weak_tensor_patched_ops_in_dispatch_dict(self):
    dispatch_dict = dispatch._TYPE_BASED_DISPATCH_SIGNATURES
    # Test that we can use the updated op reference as a key to the dispatch
    # dictionary.
    self.assertTrue(hasattr(math_ops.abs, "_tf_decorator"))
    self.assertNotEmpty(dispatch_dict[math_ops.abs])

  def test_weak_tensor_ops_dispatch(self):
    @dispatch.dispatch_for_api(math_ops.abs)
    def my_abs(x: MyTensor):
      return MyTensor(math_ops.abs(x.value))

    self.assertIsInstance(my_abs(MyTensor(constant_op.constant(1.0))), MyTensor)

    # Test unregistering dispatch with patched op reference.
    dispatch.unregister_dispatch_for(my_abs)
    with self.assertRaises(ValueError):
      math_ops.abs(MyTensor(constant_op.constant(1.0)))

  def testWeakTensorDunderMethods(self):
    x = _get_weak_tensor([1, 2, 3])

    self.assertIsInstance(abs(x), WeakTensor)
    self.assertIsInstance(~x, WeakTensor)
    self.assertIsInstance(-x, WeakTensor)

  @parameterized.parameters(
      ("T", WeakTensor),
      ("ndim", int),
      ("size", None),
      ("data", WeakTensor),
  )
  def testNumpyAttributesOnWeakTensor(self, np_attribute, result_type):
    a = weak_tensor_test_util.get_weak_tensor(([1, 2, 3]))
    b = constant_op.constant([1, 2, 3])

    self.assertTrue(hasattr(a, np_attribute))
    wt_np_attr = getattr(a, np_attribute)
    t_np_attr = getattr(b, np_attribute)
    if result_type is None:
      # The result type may differ depending on which machine test runs on
      # (e.g. size)
      self.assertEqual(type(wt_np_attr), type(t_np_attr))
    else:
      self.assertIsInstance(wt_np_attr, result_type)
    self.assertAllEqual(wt_np_attr, t_np_attr)

  @parameterized.parameters(
      ("__pos__", WeakTensor),
      ("__round__", WeakTensor, 2),
      ("tolist", list),
      ("flatten", WeakTensor),
      ("transpose", WeakTensor),
      ("reshape", WeakTensor, (3, 1)),
      ("ravel", WeakTensor),
      ("clip", tensor.Tensor, 1.1, 2.2),
      ("astype", tensor.Tensor, dtypes.float32),
      ("max", WeakTensor),
      ("mean", WeakTensor),
      ("min", WeakTensor),
  )
  def testNumpyMethodsOnWeakTensor(self, np_method, result_type, *args):
    a = weak_tensor_test_util.get_weak_tensor(([1, 2, 3]))
    b = constant_op.constant([1, 2, 3])
    self.assertTrue(hasattr(a, np_method))

    wt_np_method_call = getattr(a, np_method)
    t_np_method_call = getattr(b, np_method)
    wt_np_result = wt_np_method_call(*args)
    t_np_result = t_np_method_call(*args)
    self.assertIsInstance(wt_np_result, result_type)
    # Clip returns a float64 Tensor with Tensor input but a float32 Tensor with
    # WeakTensor input.
    self.assertAllClose(wt_np_result, t_np_result)


@parameterized.parameters(all_dtype_promos_list)
class WeakTensorBinaryOpsTest(
    test_util.TensorFlowTestCase, parameterized.TestCase
):

  def match_expected(self, actual, expected_val, expected_dtype):
    dtype, weak = expected_dtype
    expected_type = WeakTensor if weak else tensor.Tensor
    self.assertIsInstance(actual, expected_type)
    self.assertEqual(actual.dtype, dtype)
    self.assertAllEqual(actual, expected_val)

  def test_weak_tensor_add(self, a_dtype, b_dtype, expected_dtype):
    def run_test_add(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)
      expected_val = constant_op.constant(
          a, expected_dtype[0]
      ) + constant_op.constant(b, expected_dtype[0])
      for x, y in zip(a_list, b_list):
        self.match_expected(math_ops.add(x, y), expected_val, expected_dtype)
        self.match_expected(math_ops.add(y, x), expected_val, expected_dtype)
        if at_least_one_tensor_type(x, y):
          self.match_expected(x + y, expected_val, expected_dtype)
          self.match_expected(y + x, expected_val, expected_dtype)

    # Limit testing values to positive numbers inputs to account for
    # both unsigned and signed input types.
    run_test_add(a=2, b=4)
    run_test_add(a=100, b=100)
    run_test_add(a=10, b=41)

  def test_weak_tensor_sub(self, a_dtype, b_dtype, expected_dtype):
    def run_test_sub(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)
      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = a_tensor - b_tensor
      expected_val_reverse = b_tensor - a_tensor
      for x, y in zip(a_list, b_list):
        self.match_expected(
            math_ops.subtract(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            math_ops.subtract(y, x), expected_val_reverse, expected_dtype
        )
        if at_least_one_tensor_type(x, y):
          self.match_expected(x - y, expected_val, expected_dtype)
          self.match_expected(y - x, expected_val_reverse, expected_dtype)

    run_test_sub(a=4, b=2)
    run_test_sub(a=41, b=0)
    run_test_sub(a=100, b=50)

  def test_weak_tensor_mul(self, a_dtype, b_dtype, expected_dtype):
    def run_test_mul(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)
      expected_val = constant_op.constant(
          a, expected_dtype[0]
      ) * constant_op.constant(b, expected_dtype[0])
      for x, y in zip(a_list, b_list):
        self.match_expected(
            math_ops.multiply(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            math_ops.multiply(y, x), expected_val, expected_dtype
        )
        if at_least_one_tensor_type(a, b):
          self.match_expected(x * y, expected_val, expected_dtype)
          self.match_expected(y * x, expected_val, expected_dtype)

    run_test_mul(a=4, b=2)
    run_test_mul(a=41, b=10)
    run_test_mul(a=10, b=5)

  def test_weak_tensor_pow(self, a_dtype, b_dtype, expected_dtype):
    def run_test_pow(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)

      # Skip if provided dtype is not a valid input dtype for the op.
      if not output_dtype_supported_in_op("pow", expected_dtype[0]):
        return

      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = a_tensor**b_tensor
      reverse_expected_val = b_tensor**a_tensor
      for x, y in zip(a_list, b_list):
        self.match_expected(math_ops.pow(x, y), expected_val, expected_dtype)
        self.match_expected(
            math_ops.pow(y, x), reverse_expected_val, expected_dtype
        )
        if at_least_one_tensor_type(x, y):
          self.match_expected(x**y, expected_val, expected_dtype)
          self.match_expected(y**x, reverse_expected_val, expected_dtype)

    run_test_pow(a=4, b=2)
    run_test_pow(a=10, b=5)

  def test_weak_tensor_mod(self, a_dtype, b_dtype, expected_dtype):
    def run_test_mod(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)

      # Skip if provided dtype is not a valid input dtype for the op.
      if not output_dtype_supported_in_op("mod", expected_dtype[0]):
        return

      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = a_tensor % b_tensor
      reverse_expected_val = b_tensor % a_tensor
      for x, y in zip(a_list, b_list):
        self.match_expected(math_ops.mod(x, y), expected_val, expected_dtype)
        self.match_expected(
            math_ops.mod(y, x), reverse_expected_val, expected_dtype
        )
        # math_ops.mod and gen_math_ops.floor_mod are used interchangeably.
        self.match_expected(
            gen_math_ops.floor_mod(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            gen_math_ops.floor_mod(y, x), reverse_expected_val, expected_dtype
        )
        if at_least_one_tensor_type(x, y):
          self.match_expected(x % y, expected_val, expected_dtype)
          self.match_expected(y % x, reverse_expected_val, expected_dtype)

    run_test_mod(a=4, b=2)
    run_test_mod(a=41, b=124)
    run_test_mod(a=2, b=6)

  def test_weak_tensor_floor_div(self, a_dtype, b_dtype, expected_dtype):
    def run_test_floor_div(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)

      # Skip if provided dtype is not a valid input dtype for the op.
      if not output_dtype_supported_in_op("floor_div", expected_dtype[0]):
        return

      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = a_tensor // b_tensor
      reverse_expected_val = b_tensor // a_tensor
      for x, y in zip(a_list, b_list):
        self.match_expected(
            math_ops.floordiv(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            math_ops.floordiv(y, x), reverse_expected_val, expected_dtype
        )
        # math_ops.floordiv and math_ops.floor_div are used interchangeably.
        self.match_expected(
            math_ops.floor_div(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            math_ops.floor_div(y, x), reverse_expected_val, expected_dtype
        )
        if at_least_one_tensor_type(x, y):
          self.match_expected(x // y, expected_val, expected_dtype)
          self.match_expected(y // x, reverse_expected_val, expected_dtype)

    run_test_floor_div(a=124, b=123)
    run_test_floor_div(a=41, b=20)
    run_test_floor_div(a=2, b=6)

  def test_weak_tensor_real_div(self, a_dtype, b_dtype, expected_dtype):
    def run_test_real_div(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)

      # Skip if provided dtype is not a valid input dtype for the op.
      if not output_dtype_supported_in_op("real_div", expected_dtype[0]):
        return

      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = math_ops.real_div(a_tensor, b_tensor)
      reverse_expected_val = math_ops.real_div(b_tensor, a_tensor)
      for x, y in zip(a_list, b_list):
        self.match_expected(
            math_ops.realdiv(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            math_ops.realdiv(y, x), reverse_expected_val, expected_dtype
        )
        # math_ops.realdiv and gen_math_ops.real_div are used interchangeably.
        self.match_expected(
            gen_math_ops.real_div(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            gen_math_ops.real_div(y, x), reverse_expected_val, expected_dtype
        )

    run_test_real_div(a=124, b=123)
    run_test_real_div(a=41, b=20)
    run_test_real_div(a=2, b=6)

  def test_weak_tensor_truncate_div(self, a_dtype, b_dtype, expected_dtype):
    def run_test_truncate_div(a, b):
      # Skip if provided dtype is not a valid input dtype for the op.
      if not output_dtype_supported_in_op("truncate_div", expected_dtype[0]):
        return

      a, b = maybe_to_positive_input(a, b, a_dtype, b_dtype, expected_dtype)
      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = math_ops.truncatediv(a_tensor, b_tensor)
      reverse_expected_val = math_ops.truncatediv(b_tensor, a_tensor)

      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)
      for x, y in zip(a_list, b_list):
        self.match_expected(
            math_ops.truncatediv(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            math_ops.truncatediv(y, x), reverse_expected_val, expected_dtype
        )
        # math_ops.truncatediv and gen_math_ops.truncate_div are used
        # interchangeably.
        self.match_expected(
            gen_math_ops.truncate_div(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            gen_math_ops.truncate_div(y, x),
            reverse_expected_val,
            expected_dtype,
        )

    run_test_truncate_div(a=124, b=123)
    run_test_truncate_div(a=41, b=20)
    run_test_truncate_div(a=2, b=6)
    run_test_truncate_div(a=-7, b=5)
    run_test_truncate_div(a=1, b=-2)
    run_test_truncate_div(a=-100, b=-50)

  def test_weak_tensor_truncate_mod(self, a_dtype, b_dtype, expected_dtype):
    def run_test_truncate_mod(a, b):
      # Skip if provided dtype is not a valid input dtype for the op.
      if not output_dtype_supported_in_op("truncate_mod", expected_dtype[0]):
        return

      a, b = maybe_to_positive_input(a, b, a_dtype, b_dtype, expected_dtype)
      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = math_ops.truncatemod(a_tensor, b_tensor)
      reverse_expected_val = math_ops.truncatemod(b_tensor, a_tensor)

      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)
      for x, y in zip(a_list, b_list):
        self.match_expected(
            math_ops.truncatemod(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            math_ops.truncatemod(y, x), reverse_expected_val, expected_dtype
        )
        # math_ops.truncatemod and gen_math_ops.truncate_mod are used
        # interchangeably.
        self.match_expected(
            gen_math_ops.truncate_mod(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            gen_math_ops.truncate_mod(y, x),
            reverse_expected_val,
            expected_dtype,
        )

    run_test_truncate_mod(a=124, b=123)
    run_test_truncate_mod(a=41, b=20)
    run_test_truncate_mod(a=2, b=6)
    run_test_truncate_mod(a=-7, b=5)
    run_test_truncate_mod(a=1, b=-1)
    run_test_truncate_mod(a=-100, b=-50)

  def test_weak_tensor_scalar_mul(self, a_dtype, b_dtype, expected_dtype):
    def run_test_scalar_mul(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)
      # Expected dtype = second arg's dtype.
      _ = expected_dtype
      if not a_dtype[0].is_compatible_with(b_dtype[0]):
        return
      expected_val = np.multiply(a, b)
      for x, y in zip(a_list, b_list):
        self.match_expected(math_ops.scalar_mul(x, y), expected_val, b_dtype)
        self.match_expected(math_ops.scalar_mul(y, x), expected_val, a_dtype)

    run_test_scalar_mul(a=4, b=1)
    run_test_scalar_mul(a=41, b=2)
    run_test_scalar_mul(a=2, b=0)

  def test_weak_tensor_mat_mul(self, a_dtype, b_dtype, expected_dtype):
    def run_test_mat_mul(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)

      # Skip if provided dtype is not a valid input dtype for the op.
      if not output_dtype_supported_in_op("matmul", expected_dtype[0]):
        return

      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = math_ops.matmul(a_tensor, b_tensor)
      expected_val_reverse = math_ops.matmul(b_tensor, a_tensor)
      for x, y in zip(a_list, b_list):
        self.match_expected(math_ops.matmul(x, y), expected_val, expected_dtype)
        self.match_expected(
            math_ops.matmul(y, x), expected_val_reverse, expected_dtype
        )

    run_test_mat_mul(a=[[2, 1], [3, 4]], b=[[1, 2], [3, 4]])
    run_test_mat_mul(a=[[[3]]], b=[[[2]]])

  def test_weak_tensor_truediv(self, a_dtype, b_dtype, expected_dtype):
    def run_test_truediv(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)
      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = a_tensor / b_tensor
      reverse_expected_val = b_tensor / a_tensor
      for x, y in zip(a_list, b_list):
        # Truediv has a dtype conversion orthagonal to our change. Therefore,
        # we compare our result dtype to Tensor truediv.
        expected_result_dtype = expected_val.dtype
        self.match_expected(
            math_ops.truediv(x, y),
            expected_val,
            (expected_result_dtype, expected_dtype[1]),
        )
        self.match_expected(
            math_ops.truediv(y, x),
            reverse_expected_val,
            (expected_result_dtype, expected_dtype[1]),
        )
        # truediv, divide, and divide dunder method all use Python 3 division
        # semantics.
        self.match_expected(
            math_ops.divide(x, y),
            expected_val,
            (expected_result_dtype, expected_dtype[1]),
        )
        self.match_expected(
            math_ops.divide(y, x),
            reverse_expected_val,
            (expected_result_dtype, expected_dtype[1]),
        )
        if at_least_one_tensor_type(x, y):
          self.match_expected(
              x / y, expected_val, (expected_result_dtype, expected_dtype[1])
          )
          self.match_expected(
              y / x,
              reverse_expected_val,
              (expected_result_dtype, expected_dtype[1]),
          )

    run_test_truediv(a=4, b=2)
    run_test_truediv(a=41, b=3)
    run_test_truediv(a=2, b=6)

  def test_weak_tensor_div_no_nan(self, a_dtype, b_dtype, expected_dtype):
    def run_test_div_no_nan(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)
      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = math_ops.div_no_nan(a_tensor, b_tensor)
      reverse_expected_val = math_ops.div_no_nan(b_tensor, a_tensor)
      # The behavior of div_no_nan is same as truediv in most cases, except
      # for when it divides by nan or 0.
      expected_result_dtype = expected_val.dtype
      for x, y in zip(a_list, b_list):
        self.match_expected(
            math_ops.div_no_nan(x, y),
            expected_val,
            (expected_result_dtype, expected_dtype[1]),
        )
        self.match_expected(
            math_ops.div_no_nan(y, x),
            reverse_expected_val,
            (expected_result_dtype, expected_dtype[1]),
        )

    run_test_div_no_nan(a=4, b=2)
    run_test_div_no_nan(a=41, b=40)
    run_test_div_no_nan(a=2, b=6)

    # Test div_no_nan(x, 0) = 0 even if x is NaN or Inf.
    x = np.NaN
    y = 0
    self.match_expected(math_ops.div_no_nan(x, y), 0, (dtypes.float32, True))

    x = np.Inf
    self.match_expected(math_ops.div_no_nan(x, y), 0, (dtypes.float32, True))

  def test_weak_tensor_multiply_no_nan(self, a_dtype, b_dtype, expected_dtype):
    def run_test_multiply_no_nan(a, b):
      a_list = _get_test_input_for_binary_op(a, a_dtype)
      b_list = _get_test_input_for_binary_op(b, b_dtype)

      # Skip if provided dtype is not a valid input dtype for the op.
      if not output_dtype_supported_in_op("multiply_no_nan", expected_dtype[0]):
        return

      a_tensor = constant_op.constant(a, expected_dtype[0])
      b_tensor = constant_op.constant(b, expected_dtype[0])
      expected_val = math_ops.multiply_no_nan(a_tensor, b_tensor)
      for x, y in zip(a_list, b_list):
        self.match_expected(
            math_ops.multiply_no_nan(x, y), expected_val, expected_dtype
        )
        self.match_expected(
            math_ops.multiply_no_nan(y, x), expected_val, expected_dtype
        )

    run_test_multiply_no_nan(a=4, b=2)
    run_test_multiply_no_nan(a=41, b=10)
    run_test_multiply_no_nan(a=2, b=6)

    # Test multiply_no_nan(x, 0) = 0 even if x is NaN or Inf.
    x = np.NaN
    y = 0
    self.match_expected(
        math_ops.multiply_no_nan(x, y), 0, (dtypes.float32, True)
    )

    x = np.Inf
    self.match_expected(
        math_ops.multiply_no_nan(x, y), 0, (dtypes.float32, True)
    )


@parameterized.parameters(safe_mode_unallowed_promos_list)
class WeakTensorBinaryOpsTestSafeMode(
    test_util.TensorFlowTestCase, parameterized.TestCase
):

  def test_weak_tensor_add(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv("safe"):
      a_list = _get_test_input_for_binary_op(1, a_dtype)
      b_list = _get_test_input_for_binary_op(1, b_dtype)
      for x, y in zip(a_list, b_list):
        with self.assertRaises(TypeError):
          _ = math_ops.add(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.add(y, x)
        if at_least_one_tensor_type(x, y):
          with self.assertRaises(TypeError):
            _ = x + y
          with self.assertRaises(TypeError):
            _ = y + x

  def test_weak_tensor_sub(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv("safe"):
      a_list = _get_test_input_for_binary_op(1, a_dtype)
      b_list = _get_test_input_for_binary_op(1, b_dtype)
      for x, y in zip(a_list, b_list):
        with self.assertRaises(TypeError):
          _ = math_ops.subtract(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.subtract(y, x)
        if at_least_one_tensor_type(x, y):
          with self.assertRaises(TypeError):
            _ = x - y
          with self.assertRaises(TypeError):
            _ = y - x

  def test_weak_tensor_mul(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv("safe"):
      a_list = _get_test_input_for_binary_op(1, a_dtype)
      b_list = _get_test_input_for_binary_op(1, b_dtype)
      for x, y in zip(a_list, b_list):
        with self.assertRaises(TypeError):
          _ = math_ops.multiply(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.multiply(y, x)
        if at_least_one_tensor_type(x, y):
          with self.assertRaises(TypeError):
            _ = x * y
          with self.assertRaises(TypeError):
            _ = y * x

  def test_weak_tensor_pow(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv("safe"):
      a_list = _get_test_input_for_binary_op(1, a_dtype)
      b_list = _get_test_input_for_binary_op(1, b_dtype)
      for x, y in zip(a_list, b_list):
        with self.assertRaises(TypeError):
          _ = math_ops.pow(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.pow(y, x)
        if at_least_one_tensor_type(x, y):
          with self.assertRaises(TypeError):
            _ = x**y
          with self.assertRaises(TypeError):
            _ = y**x

  def test_weak_tensor_mod(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv("safe"):
      a_list = _get_test_input_for_binary_op(1, a_dtype)
      b_list = _get_test_input_for_binary_op(1, b_dtype)
      for x, y in zip(a_list, b_list):
        with self.assertRaises(TypeError):
          _ = math_ops.mod(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.mod(y, x)
        with self.assertRaises(TypeError):
          _ = gen_math_ops.floor_mod(x, y)
        with self.assertRaises(TypeError):
          _ = gen_math_ops.floor_mod(y, x)
        if at_least_one_tensor_type(x, y):
          with self.assertRaises(TypeError):
            _ = x % y
          with self.assertRaises(TypeError):
            _ = y % x

  def test_weak_tensor_floor_div(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv("safe"):
      a_list = _get_test_input_for_binary_op(1, a_dtype)
      b_list = _get_test_input_for_binary_op(1, b_dtype)
      for x, y in zip(a_list, b_list):
        with self.assertRaises(TypeError):
          _ = math_ops.floordiv(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.floordiv(y, x)
        with self.assertRaises(TypeError):
          _ = gen_math_ops.floor_div(x, y)
        with self.assertRaises(TypeError):
          _ = gen_math_ops.floor_div(y, x)
        if at_least_one_tensor_type(x, y):
          with self.assertRaises(TypeError):
            _ = x // y
          with self.assertRaises(TypeError):
            _ = y // x

  def test_weak_tensor_real_div(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv("safe"):
      a_list = _get_test_input_for_binary_op(1, a_dtype)
      b_list = _get_test_input_for_binary_op(1, b_dtype)
      for x, y in zip(a_list, b_list):
        with self.assertRaises(TypeError):
          _ = math_ops.realdiv(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.realdiv(y, x)
        with self.assertRaises(TypeError):
          _ = gen_math_ops.real_div(x, y)
        with self.assertRaises(TypeError):
          _ = gen_math_ops.real_div(y, x)

  def test_weak_tensor_truncate_mod(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv("safe"):
      a_list = _get_test_input_for_binary_op(1, a_dtype)
      b_list = _get_test_input_for_binary_op(1, b_dtype)
      for x, y in zip(a_list, b_list):
        with self.assertRaises(TypeError):
          _ = math_ops.truncatemod(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.truncatemod(y, x)
        with self.assertRaises(TypeError):
          _ = gen_math_ops.truncate_mod(x, y)
        with self.assertRaises(TypeError):
          _ = gen_math_ops.truncate_mod(y, x)

  def test_weak_tensor_truncate_div(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv("safe"):
      a_list = _get_test_input_for_binary_op(1, a_dtype)
      b_list = _get_test_input_for_binary_op(1, b_dtype)
      for x, y in zip(a_list, b_list):
        with self.assertRaises(TypeError):
          _ = math_ops.truncatediv(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.truncatediv(y, x)
        with self.assertRaises(TypeError):
          _ = gen_math_ops.truncate_div(x, y)
        with self.assertRaises(TypeError):
          _ = gen_math_ops.truncate_div(y, x)

  def test_weak_tensor_mat_mul(self, a_dtype, b_dtype):
    with DtypeConversionTestEnv("safe"):
      a_list = _get_test_input_for_binary_op([[1]], a_dtype)
      b_list = _get_test_input_for_binary_op([[1]], b_dtype)
      for x, y in zip(a_list, b_list):
        with self.assertRaises(TypeError):
          _ = math_ops.matmul(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.matmul(y, x)
        with self.assertRaises(TypeError):
          _ = math_ops.matmul(x, y)
        with self.assertRaises(TypeError):
          _ = math_ops.matmul(y, x)


def at_least_one_tensor_type(a, b):
  """Returns True if at least one of the inputs is a Tensor/WeakTensor."""
  if isinstance(a, tensor.Tensor) or isinstance(a, WeakTensor):
    return True
  if isinstance(b, tensor.Tensor) or isinstance(b, WeakTensor):
    return True
  return False


def maybe_to_positive_input(a, b, a_dtype, b_dtype, expected_dtype):
  """Converts inputs to positive inputs if the provided dtypes are unsigned."""
  unsigned_types = [dtypes.uint8, dtypes.uint16, dtypes.uint32, dtypes.uint64]
  if a < 0 and (
      a_dtype[0] in unsigned_types or expected_dtype[0] in unsigned_types
  ):
    a = a * (-1)
  if b < 0 and (
      b_dtype[0] in unsigned_types or expected_dtype[0] in unsigned_types
  ):
    b = b * (-1)
  return a, b


def output_dtype_supported_in_op(op_name, input_dtype):
  real_dtypes = [
      dtypes.int8,
      dtypes.int16,
      dtypes.int32,
      dtypes.int64,
      dtypes.uint8,
      dtypes.uint16,
      dtypes.uint32,
      dtypes.uint64,
      dtypes.bfloat16,
      dtypes.half,
      dtypes.float32,
      dtypes.float64,
  ]
  # Valid dtypes for the given op in Eager Mode.
  valid_dtypes_in_eager = {
      "pow": [
          dtypes.float16,
          dtypes.float32,
          dtypes.float64,
          dtypes.int32,
          dtypes.int64,
          dtypes.complex64,
          dtypes.complex128,
      ],
      "mod": real_dtypes,
      "floor_div": real_dtypes,
      "real_div": [
          dtypes.bfloat16,
          dtypes.float16,
          dtypes.float32,
          dtypes.float64,
          dtypes.complex64,
          dtypes.complex128,
      ],
      "truncate_div": real_dtypes,
      "truncate_mod": [
          dtypes.int32,
          dtypes.int64,
          dtypes.float32,
          dtypes.float64,
      ],
      "matmul": [
          dtypes.bfloat16,
          dtypes.float16,
          dtypes.float32,
          dtypes.float64,
          dtypes.int32,
          dtypes.int64,
          dtypes.complex64,
          dtypes.complex128,
      ],
      "multiply_no_nan": [dtypes.float32, dtypes.float64],
  }
  return input_dtype in valid_dtypes_in_eager[op_name]


# TODO(b/289333658): Add tf.constant(x) with no dtype arg as a "weak" input
# after adding WeakTensor construction logic to tf.constant.
def get_test_input_for_unary_op(op):
  if op in _TF_UNARY_APIS_WITH_INT_INPUT:
    return (
        _get_weak_tensor(5, dtypes.int32),
        5,
        constant_op.constant(5),
        constant_op.constant(5, dtypes.int32),
        np.array(5),
    )
  elif op in _TF_UNARY_APIS_WITH_2D_INPUT:
    return (
        _get_weak_tensor([[1, 2], [3, 4]], dtypes.int32),
        [[1, 2], [3, 4]],
        constant_op.constant([[1, 2], [3, 4]]),
        constant_op.constant([[1, 2], [3, 4]], dtypes.int32),
        np.array([[1, 2], [3, 4]]),
    )
  else:
    return (
        _get_weak_tensor([1.0, 2.0, 3.0], dtype=dtypes.float32),
        [1.0, 2.0, 3.0],
        constant_op.constant([1.0, 2.0, 3.0]),
        constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32),
        np.array([1.0, 2.0, 3.0]),
    )


if __name__ == "__main__":
  ops.enable_eager_execution()
  # Enabling numpy behavior adds some NumPy methods to the Tensor class, which
  # TF-NumPy ops depend on.
  np_config.enable_numpy_behavior(dtype_conversion_mode="all")
  googletest.main()
