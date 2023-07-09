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
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow.lite.kernels.variants.py import register_list_ops_py
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import googletest
from tensorflow.python.types.core import Callable


def make_reserve_set_stack(shape: tf.TensorShape) -> Callable:
  @tf.function(input_signature=[tf.TensorSpec(shape=shape, dtype=tf.float32)])
  def reserve_set_stack(x: tf.Tensor) -> tf.Tensor:
    l = list_ops.tensor_list_reserve(
        element_shape=shape,
        element_dtype=tf.float32,
        num_elements=2,
    )

    l1 = list_ops.tensor_list_set_item(l, 0, x)
    l2 = list_ops.tensor_list_set_item(l1, 1, x)
    return list_ops.tensor_list_stack(l2, tf.float32)

  return reserve_set_stack


def make_from_stack(shape: tf.TensorShape) -> Callable:
  @tf.function(input_signature=[tf.TensorSpec(shape=shape, dtype=tf.float32)])
  def from_stack(x: tf.Tensor) -> tf.Tensor:
    l = list_ops.tensor_list_from_tensor(x, element_shape=x.shape[1:])
    return list_ops.tensor_list_stack(l, tf.float32)

  return from_stack


def make_reserve_set_get(shape: tf.TensorShape, get_empty: bool) -> Callable:
  @tf.function(input_signature=[tf.TensorSpec(shape=shape, dtype=tf.float32)])
  def reserve_set_get(x: tf.Tensor) -> tf.Tensor:
    l = list_ops.tensor_list_reserve(
        element_shape=shape,
        element_dtype=tf.float32,
        num_elements=2,
    )

    l = list_ops.tensor_list_set_item(l, 0, x)
    return list_ops.tensor_list_get_item(
        l, 1 if get_empty else 0, tf.float32, shape
    )

  return reserve_set_get


class ListOpsTest(parameterized.TestCase):
  """Checks for tfl custom TensorList kernels.

  * Validate conversion through python api with
    `legalize_custom_tensor_list_ops = True`.
  * Validate python bindings for registration through
    `InterpreterWithCustomOps`.
  * Check tensorflow and tensorflow lite output the same data.
  """

  def _get_interpreter_from_c_func(self, func):
    concrete_function_list = [func.get_concrete_function()]

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        concrete_function_list
    )
    # Don't allow flex ops.
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.allow_custom_ops = True
    converter.legalize_custom_tensor_list_ops = True

    tfl_model = converter.convert()

    # Instantiate interpreter with custom tensor list ops.
    interpreter = _interpreter.InterpreterWithCustomOps(
        model_content=tfl_model,
        custom_op_registerers=[register_list_ops_py.TFLRegisterListOps],
    )
    return interpreter

  @parameterized.named_parameters(
      ("ReserveSetStackStatic", make_reserve_set_stack(tf.TensorShape([2, 2]))),
      ("FromStackStatic", make_from_stack(tf.TensorShape([2, 2]))),
      (
          "ReserveSetStackGetStatic_GetEmpty1D",
          make_reserve_set_get(tf.TensorShape([2]), True),
      ),
      (
          "ReserveSetStackGetStatic_GetEmpty2D",
          make_reserve_set_get(tf.TensorShape([2, 3]), True),
      ),
      (
          "ReserveSetStackGetStatic_GetPresent1D",
          make_reserve_set_get(tf.TensorShape([2]), False),
      ),
      (
          "ReserveSetStackGetStatic_GetPresent2D",
          make_reserve_set_get(tf.TensorShape([2, 3]), False),
      ),
  )
  def test_register_and_invoke_static_shape(self, tf_func):
    interpreter = self._get_interpreter_from_c_func(tf_func)

    interpreter.allocate_tensors()

    detail = interpreter.get_input_details()[0]
    input_tensor = np.ndarray(detail["shape"], detail["dtype"])

    input_tensor.fill(1.0)
    interpreter.set_tensor(detail["index"], input_tensor)

    interpreter.invoke()

    output_tensor = interpreter.get_tensor(
        interpreter.get_output_details()[0]["index"]
    )

    tf_out = tf_func(input_tensor)

    self.assertEqual(tf_out.dtype, output_tensor.dtype)
    self.assertEqual(tf_out.shape, output_tensor.shape)
    self.assertTrue((tf_out == output_tensor).numpy().all())

  @parameterized.named_parameters(
      (
          "ReserveSetStackDynamic",
          make_reserve_set_stack(tf.TensorShape(None)),
          [2, 2],
      ),
      (
          "FromStackDynamic",
          make_reserve_set_stack(tf.TensorShape(None)),
          [2, 3],
      ),
      (
          "ReserveSetStackGetDynamic_GetEmpty",
          make_reserve_set_get(tf.TensorShape(None), True),
          [3, 2],
      ),
      (
          "ReserveSetStackGetDynamic_GetPresent",
          make_reserve_set_get(tf.TensorShape(None), False),
          [4],
      ),
  )
  def test_register_list_ops_and_invoke_dynamic_shape(
      self, tf_func, input_shape
  ):
    interpreter = self._get_interpreter_from_c_func(tf_func)

    # Save some typing.
    detail = lambda k: interpreter.get_input_details()[0][k]

    interpreter.resize_tensor_input(detail("index"), input_shape)

    interpreter.allocate_tensors()

    input_tensor = np.ndarray(detail("shape"), detail("dtype"))
    input_tensor.fill(1.0)
    interpreter.set_tensor(detail("index"), input_tensor)

    interpreter.invoke()

    output_tensor = interpreter.get_tensor(
        interpreter.get_output_details()[0]["index"]
    )

    tf_out = tf_func(input_tensor)

    self.assertEqual(tf_out.dtype, output_tensor.dtype)
    self.assertEqual(tf_out.shape, output_tensor.shape)
    self.assertTrue((tf_out == output_tensor).numpy().all())

  @parameterized.named_parameters(
      ("ZeroElements_ScalarStackShape", [], 0),
      ("NonZeroElements_ScalarStackShape", [], 2),
      ("NonZeroElements_ZeroStackShape", [0], 2),
      ("ZeroElements_ZeroStackShape", [0], 0),
      ("ZeroElements_2DZeroStackShape", [0, 2], 0),
      ("NonZeroElements_2DZeroStackShape", [0, 2], 2),
  )
  def test_stack_empty_list(
      self, stack_element_shape: list[int], num_elements: int
  ):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.int32)
        ]
    )
    def reserve_stack(stack_element_shape) -> tf.Tensor:
      l = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(None),
          element_dtype=tf.float32,
          num_elements=num_elements,
      )
      return list_ops.tensor_list_stack(
          l, element_shape=stack_element_shape, element_dtype=tf.float32
      )

    interpreter = self._get_interpreter_from_c_func(reserve_stack)

    input_index = interpreter.get_input_details()[0]["index"]

    interpreter.resize_tensor_input(input_index, [len(stack_element_shape)])

    interpreter.allocate_tensors()

    input_tensor = np.array(stack_element_shape, dtype=np.int32)
    interpreter.set_tensor(input_index, input_tensor)

    interpreter.invoke()

    output_tensor = interpreter.get_tensor(
        interpreter.get_output_details()[0]["index"]
    )

    tf_out = reserve_stack(input_tensor)

    self.assertEqual(tf_out.dtype, output_tensor.dtype)
    self.assertEqual(tf_out.shape, output_tensor.shape)
    self.assertTrue((tf_out == output_tensor).numpy().all())


if __name__ == "__main__":
  googletest.main()
