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


from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test


class InteropTest(test.TestCase):

  def setUp(self):
    super(InteropTest, self).setUp()
    physical_devices = config.list_physical_devices('CPU')
    configs = config.get_logical_device_configuration(physical_devices[0])
    if configs is None:
      logical_devices = [
          context.LogicalDeviceConfiguration() for _ in range(3)
      ]
      config.set_logical_device_configuration(physical_devices[0],
                                              logical_devices)

  def testGradientTapeInterop(self):
    with backprop.GradientTape() as t:
      x = np_array_ops.asarray(3.0)
      y = np_array_ops.asarray(2.0)

      t.watch([x, y])

      xx = 2 * x
      yy = 3 * y

    dx, dy = t.gradient([xx, yy], [x, y])

    # # TODO(nareshmodi): Figure out a way to rewrap ndarray as tensors.
    # self.assertIsInstance(dx, np_arrays.ndarray)
    # self.assertIsInstance(dy, np_arrays.ndarray)
    self.assertAllClose(dx, 2.0)
    self.assertAllClose(dy, 3.0)

  def testFunctionInterop(self):
    x = np_array_ops.asarray(3.0)
    y = np_array_ops.asarray(2.0)

    add = lambda x, y: x + y
    add_fn = def_function.function(add)

    raw_result = add(x, y)
    fn_result = add_fn(x, y)

    self.assertIsInstance(raw_result, np_arrays.ndarray)
    self.assertIsInstance(fn_result, np_arrays.ndarray)
    self.assertAllClose(raw_result, fn_result)

  def testCondInterop(self):
    x = np_array_ops.asarray(3.0)

    def fn(x):
      x_plus_1 = control_flow_ops.cond(x > 0, lambda: x+1, lambda: x+2)
      x_plus_2 = control_flow_ops.cond(x < 0, lambda: x+1, lambda: x+2)

      return x_plus_1, x_plus_2

    raw_x_plus_1, raw_x_plus_2 = fn(x)
    fn_x_plus_1, fn_x_plus_2 = def_function.function(fn)(x)

    self.assertAllClose(raw_x_plus_1, x + 1)
    self.assertAllClose(raw_x_plus_2, x + 2)

    self.assertAllClose(fn_x_plus_1, x + 1)
    self.assertAllClose(fn_x_plus_2, x + 2)

  def testWhileInterop(self):
    def fn():
      x = np_array_ops.asarray(0)
      c = lambda x: x < 10000
      b = lambda x: [x + 1]
      return control_flow_ops.while_loop_v2(c, b, [x], parallel_iterations=20)

    self.assertEqual(10000, fn()[0])
    self.assertEqual(10000, def_function.function(fn)()[0])

  def testTensorTFNPArrayInterop(self):
    arr = np_array_ops.asarray(0.)
    t = constant_op.constant(10.)

    arr_plus_t = arr + t
    t_plus_arr = t + arr

    self.assertIsInstance(arr_plus_t, ops.Tensor)
    self.assertIsInstance(t_plus_arr, ops.Tensor)
    self.assertEqual(10., arr_plus_t.numpy())
    self.assertEqual(10., t_plus_arr.numpy())

  def testTensorTFNPOp(self):
    t = constant_op.constant(10.)

    sq = np_math_ops.square(t)
    self.assertIsInstance(sq, np_arrays.ndarray)
    self.assertEqual(100., sq)

  def testTFNPArrayTFOpInterop(self):
    arr = np_array_ops.asarray(10.)

    # TODO(nareshmodi): Test more ops.
    sq = math_ops.square(arr)
    self.assertIsInstance(sq, ops.Tensor)
    self.assertEqual(100., sq.numpy())

  def testTFNPArrayNPOpInterop(self):
    arr = np_array_ops.asarray([10.])

    # TODO(nareshmodi): Test more ops.
    sq = onp.square(arr)
    self.assertIsInstance(sq, onp.ndarray)
    self.assertEqual(100., sq[0])

    # TODO(nareshmodi): Fails since the autopacking code doesn't use
    # nest.flatten.
#   def testAutopacking(self):
#     arr1 = np_array_ops.asarray(1.)
#     arr2 = np_array_ops.asarray(2.)
#     arr3 = np_array_ops.asarray(3.)
#     t = ops.convert_to_tensor_v2([arr1, arr2, arr3])

#     self.assertEqual(t.numpy(), [1., 2., 3.])

  def testDistStratInterop(self):
    strategy = mirrored_strategy.MirroredStrategy(
        devices=['CPU:0', 'CPU:1', 'CPU:2'])

    multiplier = np_array_ops.asarray(5.)

    with strategy.scope():
      @def_function.function
      def run():
        ctx = distribution_strategy_context.get_replica_context()
        val = np_array_ops.asarray(ctx.replica_id_in_sync_group)
        return val * multiplier

      distributed_values = strategy.run(run)
      reduced = strategy.reduce(reduce_util.ReduceOp.SUM,
                                distributed_values, axis=None)

    values = distributed_values.values

    # Note that this should match the number of virtual CPUs.
    self.assertLen(values, 3)
    self.assertIsInstance(values[0], np_arrays.ndarray)
    self.assertIsInstance(values[1], np_arrays.ndarray)
    self.assertIsInstance(values[2], np_arrays.ndarray)
    self.assertAllClose(values[0], 0)
    self.assertAllClose(values[1], 5)
    self.assertAllClose(values[2], 10)

    # "strategy.reduce" doesn't rewrap in ndarray.
    # self.assertIsInstance(reduced, np_arrays.ndarray)
    self.assertAllClose(reduced, 15)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
