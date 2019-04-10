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
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import errors
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import nest


class InputIteratorTestBase(test.TestCase):

  def _create_iterator(self, input_type, dataset_fn, worker_device_pairs,
                       devices, split_batch_by,
                       enable_get_next_as_optional):
    device_map = values.ReplicaDeviceMap(devices)
    input_workers = input_lib.InputWorkers(device_map, worker_device_pairs)

    if input_type == "input_fn":
      input_contexts = []
      for i in range(input_workers.num_workers):
        input_contexts.append(
            distribute_lib.InputContext(
                num_input_pipelines=input_workers.num_workers,
                input_pipeline_id=i,
                num_replicas_in_sync=len(devices)))

      iterator = input_lib.InputFunctionIterator(
          dataset_fn, input_workers, input_contexts,
          _enable_get_next_as_optional=enable_get_next_as_optional)
    else:
      iterator = input_lib.DatasetIterator(
          dataset_fn(distribute_lib.InputContext()), input_workers,
          split_batch_by,
          _enable_get_next_as_optional=enable_get_next_as_optional)
    return iterator

  def _test_iterator(self,
                     input_type,
                     dataset_fn,
                     worker_device_pairs,
                     expected_values,
                     sess=None,
                     split_batch_by=None,
                     enable_get_next_as_optional=False):
    devices = nest.flatten([ds for _, ds in worker_device_pairs])
    iterator = self._create_iterator(
        input_type, dataset_fn, worker_device_pairs, devices, split_batch_by,
        enable_get_next_as_optional)

    evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)
    evaluate(control_flow_ops.group(iterator.initialize()))

    for expected_value in expected_values:
      next_element = iterator.get_next()
      computed_value = evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])
      self.assertEqual(len(expected_value), len(computed_value))
      for i in range(len(expected_value)):
        self.assertAllEqual(expected_value[i], computed_value[i])

    with self.assertRaises(errors.OutOfRangeError):
      next_element = iterator.get_next()
      evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])

    # After re-initializing the iterator, should be able to iterate again.
    evaluate(control_flow_ops.group(iterator.initialize()))

    for expected_value in expected_values:
      next_element = iterator.get_next()
      computed_value = evaluate(
          [values.select_replica(r, next_element) for r in range(len(devices))])
      self.assertEqual(len(expected_value), len(computed_value))
      for i in range(len(expected_value)):
        self.assertAllEqual(expected_value[i], computed_value[i])


class InputIteratorSingleWorkerTest(InputIteratorTestBase,
                                    parameterized.TestCase):

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"]))
  def testOneDeviceCPU(self, input_type):
    worker_device_pairs = [("", ["/device:CPU:0"])]
    dataset_fn = lambda _: dataset_ops.Dataset.range(10)

    expected_values = [[i] for i in range(10)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testTwoDevicesOneGPUOneCPU(self, input_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    dataset_fn = lambda _: dataset_ops.Dataset.range(10)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testTupleDataset(self, input_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]

    def dataset_fn(ctx):
      del ctx
      dataset1 = dataset_ops.Dataset.range(10)
      dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x**2)
      return dataset_ops.Dataset.zip((dataset1, dataset2))

    expected_values = [[(i, i**2), (i+1, (i+1)**2)] for i in range(0, 10, 2)]

    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values)

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          required_gpus=1))
  def testUnevenDatasetBatches(self, input_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    dataset_fn = lambda _: dataset_ops.Dataset.range(9).batch(2)

    # The last global batch only contains data for one replica.
    expected_values = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8], []]]
    self._test_iterator(input_type, dataset_fn, worker_device_pairs,
                        expected_values, enable_get_next_as_optional=True)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["dataset"],
      split_batch_by=[None, 2],
      required_gpus=1))
  def testBatchSplitting(self, input_type, split_batch_by):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    batch_size = 10
    dataset_fn = lambda _: dataset_ops.Dataset.range(100).batch(batch_size)

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
      dataset_fn = lambda _: dataset_ops.Dataset.range(4)
      if input_type == "dataset":
        # Autosharded
        expected_values = [[0, 1], [2, 3]]
      else:
        expected_values = [[0, 0], [1, 1], [2, 2], [3, 3]]
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          expected_values, sess)

  @combinations.generate(combinations.combine(
      mode=["graph"],
      input_type=["input_fn", "dataset"],
      required_gpus=1))
  def testTwoDevicesPerWorker(self, input_type):
    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      dataset_fn = lambda _: dataset_ops.Dataset.range(4)
      if input_type == "dataset":
        # Autosharded
        expected_values = [[0, 2, 1, 3]]
      else:
        expected_values = [[0, 1, 0, 1], [2, 3, 2, 3]]
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          expected_values, sess)

  @combinations.generate(combinations.combine(
      mode=["graph"],
      input_type=["input_fn", "dataset"]))
  def testTupleDataset(self, input_type):
    worker_devices = self._cpu_devices()
    with context.graph_mode(), self.cached_session() as sess:

      def dataset_fn(ctx):
        del ctx
        dataset1 = dataset_ops.Dataset.range(4)
        dataset2 = dataset_ops.Dataset.range(4).map(lambda x: x**2)
        return dataset_ops.Dataset.zip((dataset1, dataset2))

      if input_type == "dataset":
        # Autosharded
        expected_values = [[(0, 0), (1, 1)], [(2, 4), (3, 9)]]
      else:
        expected_values = [[(i, i**2), (i, i**2)] for i in range(0, 4)]
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          expected_values, sess)

  @combinations.generate(
      combinations.combine(
          mode=["graph"], input_type=["input_fn", "dataset"], required_gpus=1))
  def testUnevenDatasetBatches(self, input_type):
    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      dataset_fn = lambda _: dataset_ops.Dataset.range(9).batch(2)
      if input_type == "dataset":
        # Autosharded
        expected_values = [[[0, 1], [4, 5], [2, 3], [6, 7]], [[8], [], [], []]]
      else:
        expected_values = [[[0, 1], [2, 3], [0, 1], [2, 3]],
                           [[4, 5], [6, 7], [4, 5], [6, 7]], [[8], [], [8], []]]
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          expected_values, sess,
                          enable_get_next_as_optional=True)

  @combinations.generate(
      combinations.combine(
          mode=["graph"], input_type=["input_fn"], required_gpus=1))
  def testDifferentDatasets(self, input_type):
    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), self.cached_session() as sess:

      def dataset_fn(ctx):
        if ctx.input_pipeline_id == 0:
          return dataset_ops.Dataset.range(8).batch(2)
        else:
          return dataset_ops.Dataset.range(9).batch(2)

      expected_values = [[[0, 1], [2, 3], [0, 1], [2, 3]],
                         [[4, 5], [6, 7], [4, 5], [6, 7]], [[], [], [8], []]]
      self._test_iterator(input_type, dataset_fn, worker_devices,
                          expected_values, sess,
                          enable_get_next_as_optional=True)


if __name__ == "__main__":
  test.main()
