# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for `tf.data.MultiDeviceIterator`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl.testing import parameterized
import six

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.experimental.ops import optimization
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


# memory_profiler might not be available in the OSS version of TensorFlow.
try:
  import memory_profiler  # pylint:disable=g-import-not-at-top
except ImportError:
  memory_profiler = None


@test_util.run_all_in_graph_and_eager_modes
class MultiDeviceIteratorTest(test_base.DatasetTestBase,
                              parameterized.TestCase):

  def assertNotIncreasingMemory(self,
                                f,
                                num_iters=100000,
                                increase_threshold_absolute_mb=10):
    """Assert memory usage doesn't increase beyond given threshold for f."""

    with context.eager_mode():
      # Warm up.
      f()

      # Wait for background threads to start up and take over memory.
      # FIXME: The nature of this test leaves few other options. Maybe there
      # is a better way to do this.
      time.sleep(4)
      initial = memory_profiler.memory_usage(-1)[0]
      for _ in six.moves.range(num_iters):
        f()
      increase = memory_profiler.memory_usage(-1)[0] - initial
      assert increase < increase_threshold_absolute_mb, (
          "Increase is too high. Initial memory usage: %f MB. Increase: %f MB. "
          "Maximum allowed increase: %f") % (initial, increase,
                                             increase_threshold_absolute_mb)

  @parameterized.parameters(0, 1, 42,)
  @test_util.run_v1_only("b/121264236")
  def testInitOnly(self, num_inits):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      for _ in range(num_inits):
        self.evaluate(multi_device_iterator.initializer)

  @test_util.run_v1_only("b/121264236")
  def testBasic(self):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @test_util.run_v1_only("b/121264236")
  def testEagerNoMemoryLeak(self):
    if not context.executing_eagerly():
      self.skipTest("Only eager mode test")
    if memory_profiler is None:
      self.skipTest("memory_profiler required to run this test")

    def f():
      dataset = dataset_ops.Dataset.range(10)
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          dataset, ["/cpu:1", "/cpu:2"])
      self.evaluate(multi_device_iterator.get_next())
      del multi_device_iterator
      del dataset

    self.assertNotIncreasingMemory(
        f, num_iters=100, increase_threshold_absolute_mb=175)

  @test_util.run_v1_only("b/121264236")
  def testOneOnSameDevice(self):
    with ops.device("/cpu:0"):
      dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:0", "/cpu:1"])

    config = config_pb2.ConfigProto(device_count={"CPU": 2})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @test_util.run_v1_only("b/121264236")
  def testRepeatDevices(self):
    with ops.device("/cpu:0"):
      dataset = dataset_ops.Dataset.range(20)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2", "/cpu:1", "/cpu:2"])

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 20, 4):
        elements = multi_device_iterator.get_next()
        elem_on_1, elem_on_2, elem_on_3, elem_on_4 = elements
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
        self.assertEqual(i + 2, self.evaluate(elem_on_3))
        self.assertEqual(i + 3, self.evaluate(elem_on_4))
      with self.assertRaises(errors.OutOfRangeError):
        elements = multi_device_iterator.get_next()
        elem_on_1, elem_on_2, elem_on_3, elem_on_4 = elements
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)
        self.evaluate(elem_on_3)
        self.evaluate(elem_on_4)

  @test_util.run_v1_only("b/121264236")
  def testNotFullyDivisible(self):
    dataset = dataset_ops.Dataset.range(9)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 8, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      elem_on_1 = multi_device_iterator.get_next("/cpu:1")
      self.assertEqual(8, self.evaluate(elem_on_1))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @test_util.run_v1_only("b/121264236")
  def testGetNextAsOptional(self):
    if context.executing_eagerly():
      return

    dataset = dataset_ops.Dataset.range(9)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])
    elem_on_1, elem_on_2 = multi_device_iterator.get_next_as_optional()
    elem_on_1_has_value_t = elem_on_1.has_value()
    elem_on_1_t = elem_on_1.get_value()
    elem_on_2_has_value_t = elem_on_2.has_value()
    elem_on_2_t = elem_on_2.get_value()

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config) as sess:
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 8, 2):
        elem_on_1_has_value, elem_on_1_value = sess.run(
            [elem_on_1_has_value_t, elem_on_1_t])
        self.assertTrue(elem_on_1_has_value)
        self.assertEqual(i, elem_on_1_value)
        elem_on_2_has_value, elem_on_2_value = sess.run(
            [elem_on_2_has_value_t, elem_on_2_t])
        self.assertTrue(elem_on_2_has_value)
        self.assertEqual(i + 1, elem_on_2_value)
      elem_on_1_has_value, elem_on_1_value = sess.run(
          [elem_on_1_has_value_t, elem_on_1_t])
      self.assertTrue(elem_on_1_has_value)
      self.assertEqual(8, elem_on_1_value)
      self.assertFalse(self.evaluate(elem_on_1_has_value_t))
      self.assertFalse(self.evaluate(elem_on_2_has_value_t))
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(elem_on_1_t)
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(elem_on_2_t)

  @test_util.run_v1_only("b/121264236")
  def testUneven(self):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"], max_buffer_size=4)

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1 = multi_device_iterator.get_next("/cpu:1")
        self.assertEqual(i, self.evaluate(elem_on_1))
      for i in range(0, 10, 2):
        elem_on_2 = multi_device_iterator.get_next("/cpu:2")
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @test_util.run_v1_only("b/121264236")
  def testMultipleInitializationsGraph(self):
    if context.executing_eagerly():
      return

    with ops.device("/cpu:0"):
      epoch = array_ops.placeholder(dtypes.int64, shape=[])
      dataset1 = dataset_ops.Dataset.from_tensors(epoch).repeat(1000)
      dataset2 = dataset_ops.Dataset.range(1000)
      dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"], prefetch_buffer_size=4)
    elem_on_1, elem_on_2 = multi_device_iterator.get_next()
    init_op = multi_device_iterator.initializer

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config) as sess:
      for i in range(1000):
        sess.run(init_op, feed_dict={epoch: i})
        self.assertEqual([(i, 0), (i, 1)], self.evaluate([elem_on_1,
                                                          elem_on_2]))

  @test_util.run_v1_only("b/121264236")
  def testMultipleInitializationsEager(self):
    if not context.executing_eagerly():
      return

    with ops.device("/cpu:0"):
      dataset1 = dataset_ops.Dataset.range(1000)
      dataset2 = dataset_ops.Dataset.range(1000)
      dataset = dataset_ops.Dataset.zip((dataset1, dataset2))

    for _ in range(5):
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          dataset, ["/cpu:1", "/cpu:2"], prefetch_buffer_size=4)
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual([(0, 0), (1, 1)], self.evaluate([elem_on_1, elem_on_2]))

  @test_util.run_v1_only("b/121264236")
  def testBasicGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/gpu:0"])

    config = config_pb2.ConfigProto(device_count={"CPU": 2, "GPU": 1})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @test_util.run_v1_only("b/121264236")
  def testUnevenGpu(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")

    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/gpu:0"], max_buffer_size=4)

    config = config_pb2.ConfigProto(device_count={"CPU": 2, "GPU": 1})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1 = multi_device_iterator.get_next("/cpu:1")
        self.assertEqual(i, self.evaluate(elem_on_1))
      for i in range(0, 10, 2):
        elem_on_2 = multi_device_iterator.get_next("/gpu:0")
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)

  @test_util.run_v1_only("b/121264236")
  def testGetNextAsOptionalGpu(self):
    if not test_util.is_gpu_available() or context.executing_eagerly():
      self.skipTest("No GPU available")

    dataset = dataset_ops.Dataset.range(9)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/gpu:0"])
    elem_on_1, elem_on_2 = multi_device_iterator.get_next_as_optional()
    elem_on_1_has_value_t = elem_on_1.has_value()
    elem_on_1_t = elem_on_1.get_value()
    elem_on_2_has_value_t = elem_on_2.has_value()
    elem_on_2_t = elem_on_2.get_value()

    config = config_pb2.ConfigProto(device_count={"CPU": 2, "GPU": 1})
    with self.test_session(config=config) as sess:
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 8, 2):
        elem_on_1_has_value, elem_on_1_value = sess.run(
            [elem_on_1_has_value_t, elem_on_1_t])
        self.assertTrue(elem_on_1_has_value)
        self.assertEqual(i, elem_on_1_value)
        elem_on_2_has_value, elem_on_2_value = sess.run(
            [elem_on_2_has_value_t, elem_on_2_t])
        self.assertTrue(elem_on_2_has_value)
        self.assertEqual(i + 1, elem_on_2_value)
      elem_on_1_has_value, elem_on_1_value = sess.run(
          [elem_on_1_has_value_t, elem_on_1_t])
      self.assertTrue(elem_on_1_has_value)
      self.assertEqual(8, elem_on_1_value)
      self.assertFalse(self.evaluate(elem_on_1_has_value_t))
      self.assertFalse(self.evaluate(elem_on_2_has_value_t))
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(elem_on_1_t)
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(elem_on_2_t)

  @test_util.run_v1_only("b/121264236")
  def testOptimization(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(optimization.assert_next(["MemoryCacheImpl"]))
    dataset = dataset.skip(0)  # this should be optimized away
    dataset = dataset.cache()

    options = dataset_ops.Options()
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)

    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, ["/cpu:1", "/cpu:2"])

    config = config_pb2.ConfigProto(device_count={"CPU": 3})
    with self.test_session(config=config):
      self.evaluate(multi_device_iterator.initializer)
      for i in range(0, 10, 2):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.assertEqual(i, self.evaluate(elem_on_1))
        self.assertEqual(i + 1, self.evaluate(elem_on_2))
      with self.assertRaises(errors.OutOfRangeError):
        elem_on_1, elem_on_2 = multi_device_iterator.get_next()
        self.evaluate(elem_on_1)
        self.evaluate(elem_on_2)


if __name__ == "__main__":
  ops.enable_eager_execution(
      config=config_pb2.ConfigProto(device_count={"CPU": 3, "GPU": 1}))
  test.main()
