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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_bitwise_ops
from tensorflow.python.ops import image_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import weak_tensor_ops
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_config
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import dispatch


_get_weak_tensor = weak_tensor_test_util.get_weak_tensor
_convert_to_input_type = weak_tensor_test_util.convert_to_input_type


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
    image_ops_impl.random_brightness,
    image_ops_impl.stateless_random_brightness,
    image_ops_impl.adjust_brightness,
    clip_ops.clip_by_value,
    np_array_ops.expand_dims,
    np_array_ops.full_like,
    np_array_ops.moveaxis,
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


class MyTensor(extension_type.ExtensionType):
  value: tensor.Tensor


class WeakTensorOpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  # Test unary ops with one input.
  @parameterized.named_parameters(
      (api.__module__ + "." + api.__name__, api)
      for api in set(_TF_UNARY_APIS) - set(_TF_UNARY_APIS_WITH_MULT_INPUT)
  )
  def test_unary_ops_return_weak_tensor(self, unary_api):
    weak_tensor_input, python_input, tensor_input, numpy_input = (
        _get_test_input(unary_api)
    )

    # Check that WeakTensor input outputs a WeakTensor.
    res = unary_api(weak_tensor_input)
    self.assertIsInstance(res, WeakTensor)
    expected_result = unary_api(weak_tensor_input.tensor)
    # Check that the actual result is correct.
    self.assertAllEqual(res, expected_result)

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
      ("NumPy", np.float32, tensor.Tensor),
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
        image_ops_impl.random_brightness(test_input, 0.2), result_type
    )
    self.assertIsInstance(
        image_ops_impl.stateless_random_brightness(
            image=test_input, max_delta=0.2, seed=(1, 2)
        ),
        result_type,
    )
    self.assertIsInstance(
        image_ops_impl.adjust_brightness(test_input, delta=0.2), result_type
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

    tensor_input = constant_op.constant([1.0, 2.0, 3.0], dtypes.float32)
    res = unary_api_specific_dtype(tensor_input)
    self.assertIsInstance(res, tensor.Tensor)

    tensor_input = np.array([1.0, 2.0, 3.0])
    res = unary_api_specific_dtype(tensor_input)
    self.assertIsInstance(res, tensor.Tensor)

  # Test unary ops with optional dtype arg.
  @parameterized.parameters(
      ("WeakTensor", dtypes.float32, WeakTensor),
      ("Python", None, WeakTensor),
      ("NumPy", np.float32, tensor.Tensor),
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
      ("NumPy", None, np.int32, tensor.Tensor),
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
    self.assertAllEqual(wt_np_result, t_np_result)


# TODO(b/289333658): Add tf.constant(x) with no dtype arg as a "weak" input
# after adding WeakTensor construction logic to tf.constant.
def _get_test_input(op):
  if op in _TF_UNARY_APIS_WITH_INT_INPUT:
    return (
        _get_weak_tensor(5, dtypes.int32),
        5,
        constant_op.constant(5, dtypes.int32),
        np.array(5),
    )
  elif op in _TF_UNARY_APIS_WITH_2D_INPUT:
    return (
        _get_weak_tensor([[1, 2], [3, 4]], dtypes.int32),
        [[1, 2], [3, 4]],
        constant_op.constant([[1, 2], [3, 4]], dtypes.int32),
        np.array([[1, 2], [3, 4]]),
    )
  else:
    return (
        _get_weak_tensor([1.0, 2.0, 3.0], dtype=dtypes.float32),
        [1.0, 2.0, 3.0],
        constant_op.constant([1.0, 2.0, 3.0], dtype=dtypes.float32),
        np.array([1.0, 2.0, 3.0]),
    )


if __name__ == "__main__":
  ops.enable_eager_execution()
  # Enabling numpy behavior adds some NumPy methods to the Tensor class, which
  # TF-NumPy ops depend on.
  np_config.enable_numpy_behavior(dtype_conversion_mode="all")
  googletest.main()
