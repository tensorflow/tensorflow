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
"""Tests for the input_lib library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.contrib.distribute.python import combinations
from tensorflow.contrib.distribute.python import multi_worker_test_base
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.util import nest


class PerReplicaDatasetTest(test.TestCase):

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  def _test_iterator(self, devices, dataset, expected_values):
    device_map = values.ReplicaDeviceMap(devices)
    input_workers = input_lib.InputWorkers(device_map)
    per_replica_dataset = input_lib.PerReplicaDataset(dataset, input_workers, 0)
    if context.executing_eagerly():
      iterator = per_replica_dataset.make_one_shot_iterator()
    else:
      iterator = per_replica_dataset.make_initializable_iterator()
      self.evaluate([iterator.initializer])

    for expected_value in expected_values:
      next_element = iterator.get_next_as_list()
      computed_value = self.evaluate(next_element)
      self.assertEqual(expected_value, computed_value)

    with self.assertRaises(errors.OutOfRangeError):
      next_element = iterator.get_next_as_list()
      self.evaluate(next_element)

  @test_util.run_in_graph_and_eager_modes
  def testOneDevice(self):
    devices = ["/device:CPU:0"]
    dataset = dataset_ops.Dataset.range(10)

    expected_values = [[i] for i in range(10)]

    self._test_iterator(devices, dataset, expected_values)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testMultipleDevices(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    devices = ["/device:CPU:0", "/device:GPU:0"]
    dataset = dataset_ops.Dataset.range(10)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    self._test_iterator(devices, dataset, expected_values)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testTupleDataset(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    devices = ["/device:CPU:0", "/device:GPU:0"]
    dataset1 = dataset_ops.Dataset.range(10)
    dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x**2)
    dataset = dataset_ops.Dataset.zip((dataset1, dataset2))

    expected_values = [[(i, i**2), (i+1, (i+1)**2)] for i in range(0, 10, 2)]

    self._test_iterator(devices, dataset, expected_values)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testUnevenDatasetBatches(self):
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    devices = ["/device:CPU:0", "/device:GPU:0"]
    dataset = dataset_ops.Dataset.range(11)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]
    self._test_iterator(devices, dataset, expected_values)

  def testInitializableIterator(self):
    with context.graph_mode():
      devices = ["/device:CPU:0"]
      # Using random input since that is only allowed with initializable
      # iterator.
      dataset = dataset_ops.Dataset.from_tensor_slices(
          random_ops.random_uniform((10,)))

      device_map = values.ReplicaDeviceMap(devices)
      input_workers = input_lib.InputWorkers(device_map)
      per_replica_dataset = input_lib.PerReplicaDataset(
          dataset, input_workers, 0)
      iterator = per_replica_dataset.make_initializable_iterator()

      self.evaluate(iterator.initializer)
      next_element = iterator.get_next_as_list()
      for _ in range(10):
        self.evaluate(next_element)

      # Should fail after the input is finished.
      with self.assertRaises(errors.OutOfRangeError):
        self.evaluate(next_element)

      # After re-initializing the iterator, should be able to iterate again.
      self.evaluate(iterator.initializer)
      for _ in range(10):
        self.evaluate(next_element)


class MultiWorkerDatasetTest(multi_worker_test_base.MultiWorkerTestBase):

  def _test_iterator(self, sess, iterator, devices, expected_values):
    next_element = iterator.get_next()
    for r, device in enumerate(devices):
      v = values.select_replica(r, next_element)
      # The `v` here can be a tuple.
      for element in nest.flatten(v):
        self.assertTrue(element.device in device)

    for expected_value in expected_values:
      t = [values.select_replica(r, next_element) for r in range(len(devices))]
      actual = sess.run(t)
      self.assertEqual(expected_value, actual)

    with self.assertRaises(errors.OutOfRangeError):
      sess.run([values.select_replica(r, next_element)
                for r in range(len(devices))])

  def _test_dataset(self, dataset_fn, worker_devices, devices,
                    expected_values):
    device_map = values.ReplicaDeviceMap(devices)
    input_workers = input_lib.InputWorkers(device_map, worker_devices)
    multi_worker_dataset = input_lib.MultiWorkerDataset(
        dataset_fn, input_workers)
    multi_worker_iterator = multi_worker_dataset.make_initializable_iterator()
    with self.cached_session() as sess:
      sess.run(multi_worker_iterator.initializer)
      self._test_iterator(sess, multi_worker_iterator, devices, expected_values)

  def _cpu_devices(self):
    worker_devices = (
        ("/job:worker/replica:0/task:0",
         ["/job:worker/replica:0/task:0/device:CPU:0"]),
        ("/job:worker/replica:0/task:1",
         ["/job:worker/replica:0/task:1/device:CPU:0"])
    )
    devices = [
        "/job:worker/replica:0/task:0/device:CPU:0",
        "/job:worker/replica:0/task:1/device:CPU:0"
    ]
    return worker_devices, devices

  def _cpu_and_one_gpu_devices(self):
    worker_devices = (
        ("/job:worker/replica:0/task:0", (
            "/job:worker/replica:0/task:0/device:GPU:0",
            "/job:worker/replica:0/task:0/device:CPU:0"
        )),
        ("/job:worker/replica:0/task:1", (
            "/job:worker/replica:0/task:1/device:GPU:0",
            "/job:worker/replica:0/task:1/device:CPU:0"
        ))
    )
    devices = [
        "/job:worker/replica:0/task:0/device:GPU:0",
        "/job:worker/replica:0/task:0/device:CPU:0",
        "/job:worker/replica:0/task:1/device:GPU:0",
        "/job:worker/replica:0/task:1/device:CPU:0"
    ]
    return worker_devices, devices

  def testDataDistributionOneDevicePerWorker(self):
    worker_devices, devices = self._cpu_devices()
    with context.graph_mode():
      dataset_fn = lambda: dataset_ops.Dataset.range(8)
      self._test_dataset(
          dataset_fn, worker_devices, devices,
          [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])

  def testDataDistributionTwoDevicePerWorker(self):
    if context.num_gpus() < 1:
      self.skipTest("A GPU is not available for this test.")
    worker_devices, devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode():
      dataset_fn = lambda: dataset_ops.Dataset.range(8)
      self._test_dataset(
          dataset_fn, worker_devices, devices,
          [[0, 1, 0, 1], [2, 3, 2, 3], [4, 5, 4, 5], [6, 7, 6, 7]])

  def testTupleDataset(self):
    worker_devices, devices = self._cpu_devices()

    with context.graph_mode():

      def dataset_fn():
        dataset1 = dataset_ops.Dataset.range(8)
        dataset2 = dataset_ops.Dataset.range(8).map(lambda x: x**2)
        return dataset_ops.Dataset.zip((dataset1, dataset2))

      expected_values = [[(i, i**2), (i, i**2)] for i in range(8)]
      self._test_dataset(dataset_fn, worker_devices, devices,
                         expected_values)

  def testInitializableIterator(self):
    worker_devices, devices = self._cpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      dataset_fn = lambda: dataset_ops.Dataset.range(8)
      device_map = values.ReplicaDeviceMap(devices)
      input_workers = input_lib.InputWorkers(device_map, worker_devices)
      multi_worker_dataset = input_lib.MultiWorkerDataset(
          dataset_fn, input_workers)
      multi_worker_iterator = multi_worker_dataset.make_initializable_iterator()

      sess.run(multi_worker_iterator.initializer)
      self._test_iterator(
          sess, multi_worker_iterator, devices,
          [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])

      # After re-initializing the iterator, should be able to iterate again.
      sess.run(multi_worker_iterator.initializer)
      self._test_iterator(
          sess, multi_worker_iterator, devices,
          [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]])

  def testValueErrorForIterator(self):
    # Incompatiable arguments.
    d1 = "/device:GPU:0"
    d2 = "/device:GPU:1"
    device_map = values.ReplicaDeviceMap([d1, d2])
    input_workers = input_lib.InputWorkers(
        device_map, (("w1", (d1,)), ("w2", (d2,))))
    with self.assertRaises(ValueError):
      input_lib.MultiWorkerDataIterator([("w1", None)], input_workers)

  def testDuplicateDevices(self):
    _, devices = self._cpu_devices()
    devices.append("/job:worker/replica:0/task:0/device:CPU:0")
    with self.assertRaises(ValueError):
      _ = values.ReplicaDeviceMap(devices)


class InputIteratorTestBase(test.TestCase):

  def _test_iterator(self, input_type, dataset_fn, worker_device_pairs,
                     expected_values, sess=None, split_batch_by=None):
    devices = nest.flatten([ds for _, ds in worker_device_pairs])
    device_map = values.ReplicaDeviceMap(devices)
    input_workers = input_lib.InputWorkers(device_map, worker_device_pairs)

    if input_type == "input_fn":
      input_contexts = [
          distribute_lib.InputContext() for _ in worker_device_pairs]
      input_fn = lambda _: dataset_fn()
      iterator = input_lib.InputFunctionIterator(
          input_fn, input_workers, input_contexts)
    else:
      iterator = input_lib.DatasetIterator(
          dataset_fn(), input_workers, split_batch_by)

    evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)

    evaluate(control_flow_ops.group(iterator.initialize()))

    for expected_value in expected_values:
      next_element = iterator.get_next()
      computed_value = evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])
      self.assertAllEqual(expected_value, computed_value)

    with self.assertRaises(errors.OutOfRangeError):
      next_element = iterator.get_next()
      evaluate([values.select_replica(r, next_element)
                for r in range(len(devices))])

    # After re-initializing the iterator, should be able to iterate again.
    evaluate(control_flow_ops.group(iterator.initialize()))

    for expected_value in expected_values:
      next_element = iterator.get_next()
      computed_value = evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])
      self.assertAllEqual(expected_value, computed_value)


class InputIteratorSingleWorkerTest(InputIteratorTestBase,
                                    parameterized.TestCase):

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"]))
  def testOneDeviceCPU(self, input_type):
    worker_device_pairs = [("", ["/device:CPU:0"])]
    dataset_fn = lambda: dataset_ops.Dataset.range(10)

    expected_values = [[i] for i in range(10)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testTwoDevicesOneGPUOneCPU(self, input_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    dataset_fn = lambda: dataset_ops.Dataset.range(10)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testTupleDataset(self, input_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    def dataset_fn():
      dataset1 = dataset_ops.Dataset.range(10)
      dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x**2)
      return dataset_ops.Dataset.zip((dataset1, dataset2))

    expected_values = [[(i, i**2), (i+1, (i+1)**2)] for i in range(0, 10, 2)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testUnevenDatasetBatches(self, input_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    dataset_fn = lambda: dataset_ops.Dataset.range(11)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]
    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["dataset"],
      split_batch_by=[None, 2],
      required_gpus=1))
  def testBatchSplitting(self, input_type, split_batch_by):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    batch_size = 10
    dataset_fn = lambda: dataset_ops.Dataset.range(100).batch(batch_size)

    updated_batch_size = (
        batch_size // split_batch_by if split_batch_by else batch_size)
    expected_values = [[range(i, i+updated_batch_size),
                        range(i+updated_batch_size, i+2*updated_batch_size)]
                       for i in range(0, 100, updated_batch_size*2)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values, sess=None,
                        split_batch_by=split_batch_by)


class InputIteratorMultiWorkerTest(
    multi_worker_test_base.MultiWorkerTestBase, InputIteratorTestBase,
    parameterized.TestCase):

  def _cpu_devices(self):
    return [
        ("/job:worker/replica:0/task:0",
         ["/job:worker/replica:0/task:0/device:CPU:0"]),
        ("/job:worker/replica:0/task:1",
         ["/job:worker/replica:0/task:1/device:CPU:0"])]

  def _cpu_and_one_gpu_devices(self):
    return [
        ("/job:worker/replica:0/task:0", [
            "/job:worker/replica:0/task:0/device:GPU:0",
            "/job:worker/replica:0/task:0/device:CPU:0"
        ]),
        ("/job:worker/replica:0/task:1", [
            "/job:worker/replica:0/task:1/device:GPU:0",
            "/job:worker/replica:0/task:1/device:CPU:0"
        ])
    ]

  @combinations.generate(combinations.combine(
      mode=["graph"],
      input_type=["input_fn", "dataset"]))
  def testOneDevicePerWorker(self, input_type):
    worker_devices = self._cpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      dataset_fn = lambda: dataset_ops.Dataset.range(4)
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          [[0, 0], [1, 1], [2, 2], [3, 3]], sess)

  @combinations.generate(combinations.combine(
      mode=["graph"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testTwoDevicesPerWorker(self, input_type):
    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      dataset_fn = lambda: dataset_ops.Dataset.range(4)
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          [[0, 1, 0, 1], [2, 3, 2, 3]], sess)

  @combinations.generate(combinations.combine(
      mode=["graph"],
      input_type=["input_fn", "dataset"]))
  def testTupleDataset(self, input_type):
    worker_devices = self._cpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      def dataset_fn():
        dataset1 = dataset_ops.Dataset.range(4)
        dataset2 = dataset_ops.Dataset.range(4).map(lambda x: x**2)
        return dataset_ops.Dataset.zip((dataset1, dataset2))

      expected_values = [[(i, i**2), (i, i**2)] for i in range(0, 4)]
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          expected_values, sess)


class SplitDatasetBatchTest(test.TestCase):

  def testBatchDataset(self):
    dataset = dataset_ops.Dataset.range(100).batch(20)
    split_batch_by = 2
    result_dataset = input_lib._split_dataset_batch(dataset, split_batch_by)
    expected_values = [range(i, i+10) for i in range(0, 100, 10)]
    result = [self.evaluate(el) for el in result_dataset]
    self.assertAllEqual(expected_values, result)

  def testMapAndBatchDataset(self):
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.apply(batching.map_and_batch(lambda x: x, 20))
    split_batch_by = 2
    result_dataset = input_lib._split_dataset_batch(dataset, split_batch_by)
    expected_values = [range(i, i+10) for i in range(0, 100, 10)]
    result = [self.evaluate(el) for el in result_dataset]
    self.assertAllEqual(expected_values, result)

  def testPrefetchDataset(self):
    dataset = dataset_ops.Dataset.range(100).batch(20).prefetch(1)
    split_batch_by = 2
    result_dataset = input_lib._split_dataset_batch(dataset, split_batch_by)
    expected_values = [range(i, i+10) for i in range(0, 100, 10)]
    result = [self.evaluate(el) for el in result_dataset]
    self.assertAllEqual(expected_values, result)


if __name__ == "__main__":
  test.main()
