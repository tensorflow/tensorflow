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
"""Tests for the `MultiDeviceIterator` and `OwnedMultiDeviceIterator` API."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import executor
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


cls_combination = combinations.combine(cls=[
    combinations.NamedObject("MultiDeviceIterator",
                             multi_device_iterator_ops.MultiDeviceIterator),
    combinations.NamedObject("OwnedMultiDeviceIterator",
                             multi_device_iterator_ops.OwnedMultiDeviceIterator)
])


class MultiDeviceIteratorCommonTest(test_base.DatasetTestBase,
                                    parameterized.TestCase):
  """Tests that are common to MultiDeviceIterator and OwnedMultiDeviceIterator."""

  def setUp(self):
    super().setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(3)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(), cls_combination))
  def testCancelGetNextWithDevice(self, cls):
    ping = data_flow_ops.FIFOQueue(capacity=2, dtypes=dtypes.int64)
    pong = data_flow_ops.FIFOQueue(capacity=2, dtypes=dtypes.int64)

    @def_function.function
    def map_fn(v):
      ball = ping.dequeue()
      with ops.control_dependencies([pong.enqueue(ball)]):
        return v + ping.dequeue()

    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.map(map_fn)

    # We need to set prefetch_buffer_size=0 so that we can cancel the
    # MultiDeviceIteratorGetNextFromShardOp from eager. If
    # prefetch_buffer_size>0, that op runs in the background threads of the
    # prefetch and can only be cancelled by deleting the iterator.
    multi_device_iterator = cls(
        dataset, [self._devices[1], self._devices[2]], prefetch_buffer_size=0)

    @def_function.function
    def get_next_device1():
      return multi_device_iterator.get_next(self._devices[1])

    async_executor = executor.new_executor(enable_async=True)
    with context.executor_scope(async_executor):
      cancel_mgr = cancellation.CancellationManager()
      cancel_mgr.get_cancelable_function(
          get_next_device1.get_concrete_function())()
    # Make sure we cancel in the middle of get_next.
    ping.enqueue(0)
    pong.dequeue()
    cancel_mgr.start_cancel()
    with self.assertRaises(errors.CancelledError):
      async_executor.wait()
    # Note that fetching from upstream iterator is not cancelled with the
    # cancellation of get_next.
    ping.enqueue(0)

    # Cancelling a get_next on one device shouldn't cancel the
    # multi_device_iterator and iterators on other devices.
    ping.enqueue(0)
    ping.enqueue(0)
    self.assertEqual(1,
                     multi_device_iterator.get_next(self._devices[2]).numpy())
    # FIXME(b/209534797): Workaround an asan error caused by this test.
    # Remove the dangling reference from tf.function to ensure queue objects
    # are not freed before they are flushed.
    import gc  # pylint: disable=g-import-not-at-top
    del get_next_device1
    gc.collect()

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(), cls_combination))
  def testEmptyDataset(self, cls):
    dataset = dataset_ops.Dataset.range(0)
    multi_device_iterator = cls(
        dataset, devices=[self._devices[1], self._devices[2]])
    with self.assertRaises(errors.OutOfRangeError):
      multi_device_iterator.get_next()

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(), cls_combination))
  def testEmptyDeviceList(self, cls):
    dataset = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Length for attr 'devices' of 0 must be at least minimum 1"):
      cls(dataset, devices=[])


class MultiDeviceIteratorTest(test_base.DatasetTestBase,
                              parameterized.TestCase):

  def setUp(self):
    super(MultiDeviceIteratorTest, self).setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(3)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_inits=[0, 1, 42])))
  def testInitOnly(self, num_inits):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])

    for _ in range(num_inits):
      self.evaluate(multi_device_iterator.initializer)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              max_buffer_size=[0, 1, 10], prefetch_buffer_size=[0, 1, 10])))
  def testBasic(self, prefetch_buffer_size, max_buffer_size):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]],
        max_buffer_size=max_buffer_size,
        prefetch_buffer_size=prefetch_buffer_size)

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual(i, self.evaluate(elem_on_1))
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.default_test_combinations())
  def testOneOnSameDevice(self):
    dataset = dataset_ops.Dataset.range(12)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[0], self._devices[1], self._devices[2]])

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 12, 3):
      elem_on_0, elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual(i, self.evaluate(elem_on_0))
      self.assertEqual(i + 1, self.evaluate(elem_on_1))
      self.assertEqual(i + 2, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_0, elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_0)
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatDevices(self):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[1]])

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elements = multi_device_iterator.get_next()
      elem_on_1, elem_on_2 = elements
      self.assertEqual(i, self.evaluate(elem_on_1))
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elements = multi_device_iterator.get_next()
      elem_on_1, elem_on_2 = elements
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.default_test_combinations())
  def testNotFullyDivisible(self):
    dataset = dataset_ops.Dataset.range(9)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 8, 2):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual(i, self.evaluate(elem_on_1))
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    elem_on_1 = multi_device_iterator.get_next(self._devices[1])
    self.assertEqual(8, self.evaluate(elem_on_1))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.default_test_combinations())
  def testGetNextAsOptional(self):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next_as_optional()
      has_elem_1, get_elem_1 = self.evaluate(
          [elem_on_1.has_value(), elem_on_1.get_value()])
      has_elem_2, get_elem_2 = self.evaluate(
          [elem_on_2.has_value(), elem_on_2.get_value()])
      self.assertTrue(has_elem_1)
      self.assertEqual(i, get_elem_1)
      self.assertTrue(has_elem_2)
      self.assertEqual(i + 1, get_elem_2)
    elem_on_1, elem_on_2 = multi_device_iterator.get_next_as_optional()
    has_elem_1 = elem_on_1.has_value()
    has_elem_2 = elem_on_2.has_value()
    self.assertFalse(self.evaluate(has_elem_1))
    self.assertFalse(self.evaluate(has_elem_2))
    with self.assertRaises(errors.InvalidArgumentError):
      elem_1 = elem_on_1.get_value()
      self.evaluate(elem_1)
    with self.assertRaises(errors.InvalidArgumentError):
      elem_2 = elem_on_2.get_value()
      self.evaluate(elem_2)

  @combinations.generate(test_base.default_test_combinations())
  def testUneven(self):
    dataset = dataset_ops.Dataset.range(10)
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]], max_buffer_size=4)

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elem_on_1 = multi_device_iterator.get_next(self._devices[1])
      self.assertEqual(i, self.evaluate(elem_on_1))
    for i in range(0, 10, 2):
      elem_on_2 = multi_device_iterator.get_next(self._devices[2])
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)

  @combinations.generate(test_base.graph_only_combinations())
  def testMultipleInitializationsGraph(self):
    dataset1 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset_ops.Dataset.range(1000)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))
    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]], prefetch_buffer_size=4)
    elem_on_1, elem_on_2 = multi_device_iterator.get_next()

    for _ in range(5):
      self.evaluate(multi_device_iterator.initializer)
      self.assertEqual([(0, 0), (1, 1)], self.evaluate([elem_on_1, elem_on_2]))

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleInitializationsEager(self):
    dataset1 = dataset_ops.Dataset.range(1000)
    dataset2 = dataset_ops.Dataset.range(1000)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))

    for _ in range(5):
      multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
          dataset, [self._devices[1], self._devices[2]], prefetch_buffer_size=4)
      self.evaluate(multi_device_iterator.initializer)
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual([(0, 0), (1, 1)], self.evaluate([elem_on_1, elem_on_2]))

  @combinations.generate(test_base.default_test_combinations())
  def testOptimization(self):
    dataset = dataset_ops.Dataset.range(10)
    dataset = dataset.apply(testing.assert_next(["MemoryCacheImpl"]))
    dataset = dataset.skip(0)  # this should be optimized away
    dataset = dataset.cache()

    options = options_lib.Options()
    options.experimental_optimization.noop_elimination = True
    dataset = dataset.with_options(options)

    multi_device_iterator = multi_device_iterator_ops.MultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]])

    self.evaluate(multi_device_iterator.initializer)
    for i in range(0, 10, 2):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.assertEqual(i, self.evaluate(elem_on_1))
      self.assertEqual(i + 1, self.evaluate(elem_on_2))
    with self.assertRaises(errors.OutOfRangeError):
      elem_on_1, elem_on_2 = multi_device_iterator.get_next()
      self.evaluate(elem_on_1)
      self.evaluate(elem_on_2)


class OwnedMultiDeviceIteratorTest(test_base.DatasetTestBase,
                                   parameterized.TestCase):

  def setUp(self):
    super(OwnedMultiDeviceIteratorTest, self).setUp()
    self._devices = self.configureDevicesForMultiDeviceTest(3)

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(
              max_buffer_size=[0, 1, 10], prefetch_buffer_size=[0, 1, 10])))
  def testBasic(self, max_buffer_size, prefetch_buffer_size):
    dataset = dataset_ops.Dataset.range(1000)

    mdi = multi_device_iterator_ops.OwnedMultiDeviceIterator(
        dataset, [self._devices[1], self._devices[2]],
        max_buffer_size=max_buffer_size,
        prefetch_buffer_size=prefetch_buffer_size)

    for i, el in enumerate(mdi):
      self.assertEqual([i * 2, i * 2 + 1], [el[0].numpy(), el[1].numpy()])

  @combinations.generate(test_base.eager_only_combinations())
  def testBasicFunction(self):
    queue = data_flow_ops.FIFOQueue(10, dtypes.int64)

    @def_function.function
    def fn():
      with ops.device(self._devices[0]):
        dataset = dataset_ops.Dataset.range(10)
      iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(
          dataset, [self._devices[1], self._devices[2]])
      for _ in range(5):
        el0, el1 = next(iterator)
        queue.enqueue(el0)
        queue.enqueue(el1)

    fn()

    for i in range(10):
      self.assertEqual(queue.dequeue().numpy(), i)

  @combinations.generate(test_base.eager_only_combinations())
  def testFunctionError(self):
    # In this test we verify that a function that raises an error ends up
    # properly deallocating the iterator resource.

    queue = data_flow_ops.FIFOQueue(10, dtypes.int64)
    queue.enqueue(0)

    def init_fn(n):
      return n

    def next_fn(_):
      ds = dataset_ops.Dataset.range(0)
      return next(iter(ds))

    def finalize_fn(n):
      queue.enqueue(0)
      return n

    @def_function.function
    def fn():
      dataset = dataset_ops._GeneratorDataset(
          1,
          init_fn,
          next_fn,
          finalize_fn,
          output_signature=tensor_spec.TensorSpec([], dtypes.int64))
      iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(
          dataset, [self._devices[1], self._devices[2]])
      next(iterator)

    with self.assertRaises(errors.OutOfRangeError):
      fn()

    self.assertEqual(queue.size().numpy(), 2)

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleInitializations(self):
    dataset = dataset_ops.Dataset.range(1000)

    for _ in range(5):
      multi_device_iterator = (
          multi_device_iterator_ops.OwnedMultiDeviceIterator(
              dataset, [self._devices[1], self._devices[2]]))
      for i, el in enumerate(multi_device_iterator):
        self.assertEqual([i * 2, i * 2 + 1], [el[0].numpy(), el[1].numpy()])

  @combinations.generate(test_base.eager_only_combinations())
  def testLimitedRetracing(self):
    trace_count = [0]

    @def_function.function
    def f(iterator):
      trace_count[0] += 1
      counter = np.int64(0)
      for _ in range(5):
        elem = next(iterator)
        counter += elem[0]
        counter += elem[1]
      return counter

    dataset = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(20)

    for _ in range(10):
      multi_device_iterator = (
          multi_device_iterator_ops.OwnedMultiDeviceIterator(
              dataset, [self._devices[1], self._devices[2]]))
      self.assertEqual(self.evaluate(f(multi_device_iterator)), 45)
      multi_device_iterator2 = (
          multi_device_iterator_ops.OwnedMultiDeviceIterator(
              dataset2, [self._devices[1], self._devices[2]]))
      self.assertEqual(self.evaluate(f(multi_device_iterator2)), 45)
      self.assertEqual(trace_count[0], 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testMissingDevices(self):
    dataset = dataset_ops.Dataset.range(1000)
    with self.assertRaisesRegex(ValueError, "`devices` must be provided."):
      multi_device_iterator_ops.OwnedMultiDeviceIterator(dataset)

  @combinations.generate(test_base.eager_only_combinations())
  def testMissingInput(self):
    with self.assertRaisesRegex(
        ValueError,
        "When `dataset` is not provided, both `components` and `element_spec` "
        "must be specified."):
      multi_device_iterator_ops.OwnedMultiDeviceIterator(
          dataset=None, devices=[self._devices[1], self._devices[2]])

  @combinations.generate(test_base.eager_only_combinations())
  def testExtraElementSpecInput(self):
    dataset = dataset_ops.Dataset.range(1000)
    with self.assertRaisesRegex(
        ValueError,
        "When `dataset` is provided, `element_spec` and `components` must "
        "not be specified."):
      multi_device_iterator_ops.OwnedMultiDeviceIterator(
          dataset, devices=[self._devices[1], self._devices[2]],
          element_spec=dataset.element_spec)

  @combinations.generate(test_base.graph_only_combinations())
  def testGraphMode(self):
    dataset = dataset_ops.Dataset.range(1000)
    with self.assertRaisesRegex(
        RuntimeError,
        "OwnedMultiDeviceIterator is only supported inside of tf.function or "
        "when eager execution is enabled."):
      multi_device_iterator_ops.OwnedMultiDeviceIterator(
          dataset, devices=[self._devices[1], self._devices[2]])


if __name__ == "__main__":
  test.main()
