# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for interop between TF ops, numpy_ops, and numpy methods."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
import tensorflow.compat.v2 as tf

import tensorflow.python.ops.numpy_ops as np


# Tests for code snippet put in README.md
class ReadmeTest(tf.test.TestCase):

  def testBroadcastAdd(self):
    x_np = np.ones([2, 1]) + np.ones([1, 2])
    x_onp = onp.ones([2, 1]) + onp.ones([1, 2])
    self.assertAllClose(x_onp, x_np)

  def testTypePromotion(self):
    x_np = np.ones([1, 2], dtype=np.int16) + np.ones([2, 1], dtype=np.uint8)
    x_onp = np.ones([1, 2], dtype=np.int16) + np.ones([2, 1], dtype=np.uint8)
    self.assertEqual(x_onp.dtype, x_np.dtype)
    self.assertAllClose(x_onp, x_np)

  def testTFInterop(self):
    x_np = np.sum(np.ones([1, 2]) + tf.ones([2, 1]))
    x_onp = onp.sum(onp.ones([1, 2]) + onp.ones([2, 1]))
    self.assertAllClose(x_onp, x_np)

  def testOnpInterop(self):
    x_np = onp.sum(np.ones([1, 2]) + onp.ones([2, 1]))
    x_onp = onp.sum(onp.ones([1, 2]) + onp.ones([2, 1]))
    self.assertAllClose(x_onp, x_np)

  def testDevice(self):
    if tf.test.is_gpu_available():
      with tf.device('GPU:0'):
        x = np.ones([1, 2])
      self.assertIn('GPU', tf.convert_to_tensor(x).device)
    with tf.device('CPU:0'):
      x = np.ones([1, 2])
    self.assertIn('CPU', tf.convert_to_tensor(x).device)

  def testFunction(self):

    @tf.function
    def f(x, y):
      return np.sum(x + y)

    x_np = f(np.ones([1, 2]), tf.ones([2, 1]))
    x_onp = onp.sum(onp.ones([1, 2]) + onp.ones([2, 1]))
    self.assertAllClose(x_onp, x_np)


class InteropTest(tf.test.TestCase):

  def setUp(self):
    super(InteropTest, self).setUp()
    physical_devices = tf.config.list_physical_devices('CPU')
    configs = tf.config.get_logical_device_configuration(physical_devices[0])
    if configs is None:
      logical_devices = [
          tf.config.LogicalDeviceConfiguration() for _ in range(3)
      ]
      tf.config.set_logical_device_configuration(physical_devices[0],
                                                 logical_devices)

  def testGradientTapeInterop(self):
    with tf.GradientTape() as t:
      x = np.asarray(3.0)
      y = np.asarray(2.0)

      t.watch([x, y])

      xx = 2 * x
      yy = 3 * y

    dx, dy = t.gradient([xx, yy], [x, y])

    self.assertIsInstance(dx, np.ndarray)
    self.assertIsInstance(dy, np.ndarray)
    self.assertAllClose(dx, 2.0)
    self.assertAllClose(dy, 3.0)

  def testCondInterop(self):
    x = np.asarray(3.0)

    def fn(x):
      x_plus_1 = tf.cond(x > 0, lambda: x + 1, lambda: x + 2)
      x_plus_2 = tf.cond(x < 0, lambda: x + 1, lambda: x + 2)

      return x_plus_1, x_plus_2

    raw_x_plus_1, raw_x_plus_2 = fn(x)
    fn_x_plus_1, fn_x_plus_2 = tf.function(fn)(x)

    self.assertAllClose(raw_x_plus_1, x + 1)
    self.assertAllClose(raw_x_plus_2, x + 2)

    self.assertAllClose(fn_x_plus_1, x + 1)
    self.assertAllClose(fn_x_plus_2, x + 2)

  def testWhileInterop(self):

    def fn():
      x = np.asarray(0)
      c = lambda x: x < 10000
      b = lambda x: [x + 1]
      return tf.while_loop(c, b, [x], parallel_iterations=20)

    self.assertEqual(10000, fn()[0])
    self.assertEqual(10000, tf.function(fn)()[0])

  def testTensorTFNPArrayInterop(self):
    arr = np.asarray(0.)
    t = tf.constant(10.)

    arr_plus_t = arr + t
    t_plus_arr = t + arr

    self.assertIsInstance(arr_plus_t, tf.Tensor)
    self.assertIsInstance(t_plus_arr, tf.Tensor)
    self.assertEqual(10., arr_plus_t.numpy())
    self.assertEqual(10., t_plus_arr.numpy())

  def testTensorTFNPOp(self):
    t = tf.constant(10.)

    sq = np.square(t)
    self.assertIsInstance(sq, np.ndarray)
    self.assertEqual(100., sq)

  def testTFNPArrayTFOpInterop(self):
    arr = np.asarray(10.)

    # TODO(nareshmodi): Test more ops.
    sq = tf.square(arr)
    self.assertIsInstance(sq, tf.Tensor)
    self.assertEqual(100., sq.numpy())

  def testTFNPArrayNPOpInterop(self):
    arr = np.asarray([10.])

    # TODO(nareshmodi): Test more ops.
    sq = onp.square(arr)
    self.assertIsInstance(sq, onp.ndarray)
    self.assertEqual(100., sq[0])

    # TODO(nareshmodi): Fails since the autopacking code doesn't use
    # nest.flatten.


#   def testAutopacking(self):
#     arr1 = np.asarray(1.)
#     arr2 = np.asarray(2.)
#     arr3 = np.asarray(3.)
#     t = ops.convert_to_tensor_v2([arr1, arr2, arr3])

#     self.assertEqual(t.numpy(), [1., 2., 3.])

  def testDistStratInterop(self):
    strategy = tf.distribute.MirroredStrategy(
        devices=['CPU:0', 'CPU:1', 'CPU:2'])

    multiplier = np.asarray(5.)

    @tf.function
    def run():
      ctx = tf.distribute.get_replica_context()
      val = np.asarray(ctx.replica_id_in_sync_group)
      return val * multiplier

    distributed_values = strategy.run(run)
    reduced = strategy.reduce(
        tf.distribute.ReduceOp.SUM, distributed_values, axis=None)

    values = strategy.experimental_local_results(distributed_values)

    # Note that this should match the number of virtual CPUs.
    self.assertLen(values, 3)
    self.assertIsInstance(values[0], np.ndarray)
    self.assertIsInstance(values[1], np.ndarray)
    self.assertIsInstance(values[2], np.ndarray)
    self.assertAllClose(values[0], 0)
    self.assertAllClose(values[1], 5)
    self.assertAllClose(values[2], 10)

    # "strategy.reduce" doesn't rewrap in ndarray.
    # self.assertIsInstance(reduced, np.ndarray)
    self.assertAllClose(reduced, 15)

  def testPyFuncInterop(self):
    def py_func_fn(a, b):
      return a + b

    @tf.function
    def fn(a, b):
      result = tf.py_function(py_func_fn, [a, b], a.dtype)
      return np.asarray(result)

    a = np.asarray(1.)
    b = np.asarray(2.)

    result = fn(a, b)
    self.assertIsInstance(result, np.ndarray)
    self.assertAllClose(result, 3.)

  def testDatasetInterop(self):
    values = [1, 2, 3, 4, 5, 6]
    values_as_array = np.asarray(values)

    # Tensor dataset
    dataset = tf.data.Dataset.from_tensors(values_as_array)

    for value, value_from_dataset in zip([values_as_array], dataset):
      self.assertIsInstance(value_from_dataset, np.ndarray)
      self.assertAllEqual(value_from_dataset, value)

    # Tensor slice dataset
    dataset = tf.data.Dataset.from_tensor_slices(values_as_array)

    for value, value_from_dataset in zip(values, dataset):
      self.assertIsInstance(value_from_dataset, np.ndarray)
      self.assertAllEqual(value_from_dataset, value)

    # # TODO(nareshmodi): as_numpy_iterator() doesn't work.
    # items = list(dataset.as_numpy_iterator())

    # Map over a dataset.
    dataset = dataset.map(lambda x: np.add(x, 1))

    for value, value_from_dataset in zip(values, dataset):
      self.assertIsInstance(value_from_dataset, np.ndarray)
      self.assertAllEqual(value_from_dataset, value + 1)

    # Batch a dataset.
    dataset = tf.data.Dataset.from_tensor_slices(values_as_array).batch(2)

    for value, value_from_dataset in zip([[1, 2], [3, 4], [5, 6]], dataset):
      self.assertIsInstance(value_from_dataset, np.ndarray)
      self.assertAllEqual(value_from_dataset, value)

  def testKerasInterop(self):
    # Return an ndarray from the model.
    inputs = tf.keras.layers.Input(shape=(10,))
    output_layer = tf.keras.layers.Lambda(np.square)(inputs)
    model = tf.keras.Model([inputs], output_layer)

    values = onp.arange(10, dtype=onp.float32)
    values_as_array = np.asarray(values)

    result = model(values)
    self.assertIsInstance(result, np.ndarray)
    self.assertAllClose(result, onp.square(values))

    result = model(values_as_array)
    self.assertIsInstance(result, np.ndarray)
    self.assertAllClose(result, onp.square(values))

  def testPForInterop(self):
    def outer_product(a):
      return np.tensordot(a, a, 0)

    batch_size = 100
    a = np.ones((batch_size, 32, 32))
    c = tf.vectorized_map(outer_product, a)

    # # TODO(nareshmodi): vectorized_map doesn't rewrap tensors in ndarray.
    # self.assertIsInstance(c, np.ndarray)
    self.assertEqual(c.shape, (batch_size, 32, 32, 32, 32))

  def testJacobian(self):
    with tf.GradientTape() as g:
      x = np.asarray([1., 2.])
      y = np.asarray([3., 4.])
      g.watch(x)
      g.watch(y)
      z = x * x * y

    jacobian = g.jacobian(z, [x, y])
    answer = [tf.linalg.diag(2 * x * y), tf.linalg.diag(x * x)]

    self.assertIsInstance(jacobian[0], np.ndarray)
    self.assertIsInstance(jacobian[1], np.ndarray)
    self.assertAllClose(jacobian, answer)


class FunctionTest(InteropTest):

  def testFunctionInterop(self):
    x = np.asarray(3.0)
    y = np.asarray(2.0)

    add = lambda x, y: x + y
    add_fn = tf.function(add)

    raw_result = add(x, y)
    fn_result = add_fn(x, y)

    self.assertIsInstance(raw_result, np.ndarray)
    self.assertIsInstance(fn_result, np.ndarray)
    self.assertAllClose(raw_result, fn_result)

  def testLen(self):

    @tf.function
    def f(x):
      # Note that shape of input to len is data dependent.
      return len(np.where(x)[0])

    t = np.asarray([True, False, True])
    with self.assertRaises(TypeError):
      f(t)

  def testIter(self):

    @tf.function
    def f(x):
      y, z = x
      return y, z

    with self.assertRaises(TypeError):
      f(np.asarray([3, 4]))

  def testIndex(self):

    @tf.function
    def f(x):
      return [0, 1][x]

    with self.assertRaises(TypeError):
      f(np.asarray([1]))


class VariableTest(InteropTest):

  def test(self):
    tf_var = tf.Variable(2.0)
    value = np.square(tf_var)
    self.assertIsInstance(value, np.ndarray)
    self.assertAllClose(4.0, value)
    with tf.control_dependencies([tf_var.assign_add(value)]):
      tf_var_value = tf_var.read_value()
    self.assertAllClose(6.0, tf_var_value)


if __name__ == '__main__':
  tf.compat.v1.enable_eager_execution()
  tf.test.main()
