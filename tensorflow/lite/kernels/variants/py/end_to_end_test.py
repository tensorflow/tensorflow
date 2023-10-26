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
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter.allow_custom_ops = True
    converter.legalize_custom_tensor_list_ops = True
    converter._experimental_lower_tensor_list_ops = False

    tfl_model = converter.convert()

    # Instantiate interpreter with custom tensor list ops.
    interpreter = _interpreter.InterpreterWithCustomOps(
        model_content=tfl_model,
        custom_op_registerers=[register_list_ops_py.TFLRegisterListOps],
    )
    return interpreter

  def _get_tfl_tf_outputs(self, tf_callable, *inp):
    interpreter = self._get_interpreter_from_c_func(tf_callable)
    get_inp_index = lambda i: interpreter.get_input_details()[i]["index"]
    for i, arr in enumerate(inp):
      interpreter.resize_tensor_input(get_inp_index(i), arr.shape)
    interpreter.allocate_tensors()
    for i, arr in enumerate(inp):
      interpreter.set_tensor(get_inp_index(i), arr)
    interpreter.invoke()

    tfl_out = interpreter.get_tensor(
        interpreter.get_output_details()[0]["index"]
    )
    tf_out = tf_callable(*inp)
    return tfl_out, tf_out

  @parameterized.named_parameters(
      ("1DFilledList", 2, 2, [2]),
      ("1DPartialList", 2, 1, [2]),
      ("2DFilledList", 2, 2, [2, 2]),
      ("2DPartialList", 2, 1, [2, 2]),
  )
  def test_reserve_set_stack(self, num_elements, num_set_ops, item_shape):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.float32)
        ]
    )
    def f(x: tf.Tensor) -> tf.Tensor:
      l = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(None),
          element_dtype=tf.float32,
          num_elements=num_elements,
      )
      # This will be unrolled.
      for i in range(num_set_ops):
        l = list_ops.tensor_list_set_item(l, i, x)
      return list_ops.tensor_list_stack(l, tf.float32)

    inp = np.ndarray(shape=item_shape, dtype=np.float32)
    inp.fill(1)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, inp)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("2DStatic", [2, 2], [2, 2]),
      ("3DStatic", [3, 3, 3], [3, 3, 3]),
      ("2DDynamic", None, [2, 2]),
      ("3DDynamic", None, [3, 3, 3]),
  )
  def test_from_tensor_stack(self, f_shape, t_shape):
    @tf.function(
        input_signature=[tf.TensorSpec(shape=f_shape, dtype=tf.float32)]
    )
    def f(x: tf.Tensor) -> tf.Tensor:
      l = list_ops.tensor_list_from_tensor(x, element_shape=x.shape[1:])
      return list_ops.tensor_list_stack(l, tf.float32)

    inp = np.ndarray(shape=t_shape, dtype=np.float32)
    inp.fill(1)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, inp)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("StaticReadPresent", [2, 2], [2, 2], 0, 0),
      ("StaticReadEmpty", [2, 2], [2, 2], 0, 1),
      ("DynamicReadPresent", None, [2, 2], 0, 0),
      ("DynamicReadEmpty", None, [2, 2], 0, 1),
  )
  def test_reserve_set_get(self, f_shape, t_shape, i_write, i_read):
    @tf.function(
        input_signature=[tf.TensorSpec(shape=f_shape, dtype=tf.float32)]
    )
    def f(x: tf.Tensor) -> tf.Tensor:
      l = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(None),
          element_dtype=tf.float32,
          num_elements=2,
      )
      l = list_ops.tensor_list_set_item(l, i_write, x)
      return list_ops.tensor_list_get_item(l, i_read, tf.float32)

    inp = np.ndarray(shape=t_shape, dtype=np.float32)
    inp.fill(1)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, inp)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

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
    def f(stack_element_shape) -> tf.Tensor:
      l = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(None),
          element_dtype=tf.float32,
          num_elements=num_elements,
      )
      return list_ops.tensor_list_stack(
          l, element_shape=stack_element_shape, element_dtype=tf.float32
      )

    inp = np.array(stack_element_shape, dtype=np.int32)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, inp)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("OneD", [4]),
      ("TwoD", [2, 2]),
      ("ThreeD", [2, 3, 4]),
  )
  def test_from_pop_back_unrolled(
      self,
      tensor_shape: list[int],
  ):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.int32)
        ]
    )
    def f(x: tf.Tensor) -> tf.Tensor:
      l = list_ops.tensor_list_from_tensor(x, element_shape=tensor_shape[1:])
      a = tf.constant(0, tf.int32, shape=tensor_shape[1:])
      for unused_i in range(tensor_shape[0]):
        l, e = list_ops.tensor_list_pop_back(l, tf.int32)
        a += e
      return a

    input_tensor = np.ndarray(shape=tensor_shape, dtype=np.int32)
    input_tensor.fill(1)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, input_tensor)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("OneD", [4]),
      ("TwoD", [2, 2]),
      ("ThreeD", [2, 3, 4]),
  )
  def test_from_pop_back_loop(
      self,
      tensor_shape: list[int],
  ):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.int32)
        ]
    )
    def f(x: tf.Tensor) -> tf.Tensor:
      l = list_ops.tensor_list_from_tensor(x, element_shape=tensor_shape[1:])
      a = tf.constant(0, tf.int32, shape=tensor_shape[1:])
      while len(l) > 0:  # pylint: disable=g-explicit-length-test
        l, e = list_ops.tensor_list_pop_back(l, tf.int32)
        a += e
      return a

    input_tensor = np.ndarray(shape=tensor_shape, dtype=np.int32)
    input_tensor.fill(1)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, input_tensor)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  def test_empty_tensorlist_set_stack(self):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.int32)
        ]
    )
    def f(x) -> tf.Tensor:
      l = list_ops.empty_tensor_list(tf.TensorShape(None), tf.int32)
      l2 = list_ops.tensor_list_set_item(l, 0, x, True)

      return list_ops.tensor_list_stack(l2, tf.int32)

    inp = np.ndarray(shape=[2, 2], dtype=np.int32)
    inp.fill(0)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, inp)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("Unranked", None),
      ("DynDim", [None]),
      ("DynMultiDim", [None, 2]),
      ("AllStatic", [2, 2]),
  )
  def test_reserve_element_shape(self, element_shape):
    @tf.function
    def f() -> tf.Tensor:
      l = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(element_shape),
          element_dtype=tf.int32,
          num_elements=10,
      )
      return list_ops.tensor_list_element_shape(l, tf.int32)

    tfl_out, tf_out = self._get_tfl_tf_outputs(f)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("ReserveLength0", 0),
      ("ReserveLength1", 1),
      ("ReserveLength5", 5),
  )
  def test_reserve_length(self, num_elements: int):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=tf.TensorShape([]), dtype=tf.int32)
        ]
    )
    def f(num_elements) -> tf.Tensor:
      l = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(None),
          element_dtype=tf.int32,
          num_elements=num_elements,
      )
      return list_ops.tensor_list_length(l)

    inp = np.array(num_elements, dtype=np.int32)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, inp)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("OneD", [2]),
      ("TwoD", [2, 2]),
      ("3D", [2, 3, 4]),
      ("Scalar", []),
  )
  def test_push_back_stack(self, element_shape):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=tf.TensorShape(element_shape), dtype=tf.int32)
        ]
    )
    def f(x) -> tf.Tensor:
      l = list_ops.empty_tensor_list(tf.TensorShape(None), tf.int32)
      for unused_i in range(10):
        l = list_ops.tensor_list_push_back(l, x)
      return list_ops.tensor_list_stack(l, tf.int32)

    input_tensor = np.ndarray(shape=element_shape, dtype=np.int32)
    input_tensor.fill(0)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, input_tensor)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("1D", [4]),
      ("2D", [2, 2]),
      ("3D", [2, 3, 4]),
  )
  def test_from_tensor_addn(self, element_shape):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.int32)
        ]
    )
    def f(x) -> tf.Tensor:
      l = list_ops.tensor_list_from_tensor(x, element_shape[1:])
      l2 = list_ops.tensor_list_from_tensor(x, element_shape[1:])
      l3 = tf.math.add_n([l, l2])
      return list_ops.tensor_list_stack(l3, tf.int32)

    inp = np.ndarray(shape=element_shape, dtype=np.int32)
    inp.fill(1)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, inp)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("StaticSameShape", [2, 2], [2, 2]),
      ("FullyDefinedAndUnranked", [2, 2], None),
      ("MergeToFullyDefined", [2, None], [None, 2]),
  )
  def test_reserve_addn(self, list1_shape, list2_shape):
    @tf.function
    def f() -> tf.Tensor:
      l = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(list1_shape),
          element_dtype=tf.int32,
          num_elements=10,
      )
      l2 = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(list2_shape),
          element_dtype=tf.int32,
          num_elements=10,
      )
      l3 = tf.math.add_n([l, l2])
      return list_ops.tensor_list_stack(l3, tf.int32)

    tfl_out, tf_out = self._get_tfl_tf_outputs(f)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("AllStaticShapesEqual", [2, 2], [2, 2], [2, 2]),
      ("ListShapesUnranked", None, None, [3, 3]),
  )
  def test_reserve_set_single_item_addn(
      self, list1_shape, list2_shape, input1_shape
  ):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.int32),
        ]
    )
    def f(x) -> tf.Tensor:
      l = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(list1_shape),
          element_dtype=tf.int32,
          num_elements=1,
      )
      l = list_ops.tensor_list_set_item(l, 0, x)

      l2 = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(list2_shape),
          element_dtype=tf.int32,
          num_elements=1,
      )

      m = tf.math.add_n([l, l2])
      return list_ops.tensor_list_get_item(m, 0, tf.int32)

    inp1 = np.ndarray(shape=input1_shape, dtype=np.int32)
    inp1.fill(1)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, inp1)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())

  @parameterized.named_parameters(
      ("AllStaticShapesEqual", [2, 2], [2, 2], [2, 2], [2, 2]),
      ("UrankedListShapes", None, None, [2, 2], [2, 2]),
  )
  def test_reserve_set_both_item_addn(
      self, list1_shape, list2_shape, input1_shape, input2_shape
  ):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.int32),
            tf.TensorSpec(shape=tf.TensorShape(None), dtype=tf.int32),
        ]
    )
    def f(x, y) -> tf.Tensor:
      l = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(list1_shape),
          element_dtype=tf.int32,
          num_elements=1,
      )
      l = list_ops.tensor_list_set_item(l, 0, x)

      l2 = list_ops.tensor_list_reserve(
          element_shape=tf.TensorShape(list2_shape),
          element_dtype=tf.int32,
          num_elements=1,
      )
      l2 = list_ops.tensor_list_set_item(l2, 0, y)

      m = tf.math.add_n([l, l2])
      return list_ops.tensor_list_get_item(m, 0, tf.int32)

    inp1 = np.ndarray(shape=input1_shape, dtype=np.int32)
    inp1.fill(1)
    inp2 = np.ndarray(shape=input2_shape, dtype=np.int32)
    inp2.fill(1)
    tfl_out, tf_out = self._get_tfl_tf_outputs(f, inp1, inp2)

    self.assertEqual(tf_out.dtype, tfl_out.dtype)
    self.assertEqual(tf_out.shape, tfl_out.shape)
    self.assertTrue((tf_out == tfl_out).numpy().all())


if __name__ == "__main__":
  googletest.main()
