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
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import errors
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest


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
