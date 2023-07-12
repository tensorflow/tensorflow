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

from tensorflow.lite.kernels.variants.py import register_list_ops
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


class RegisterListOpsTest(parameterized.TestCase):
  """Checks for tfl custom TensorList kernels.

  * Validate conversion through python api with
    `legalize_custom_tensor_list_ops = True`.
  * Validate python bindings for registration through
    `InterpreterWithCustomOps`.
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
        custom_op_registerers=[register_list_ops.TFLRegisterListOps],
    )
    return interpreter

  @parameterized.named_parameters(
      ("ReserveSetStackStatic", make_reserve_set_stack(tf.TensorShape([2, 2]))),
      ("FromStackStatic", make_from_stack(tf.TensorShape([2, 2]))),
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
          [2, 2],
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


if __name__ == "__main__":
  googletest.main()
