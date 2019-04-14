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
from tensorflow.python import tf2
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


class DistributedIteratorTestBase(test.TestCase):

  def _wrap_iterator(self, input_type, dataset_fn, input_workers, devices,
                     split_batch_by, enable_get_next_as_optional):
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

  def _wrap_dataset(self, input_type, dataset, input_workers,
                    split_batch_by, enable_get_next_as_optional):
    if isinstance(dataset, dataset_ops.Dataset):
      return input_lib.DistributedDatasetV1(
          dataset, input_workers,
          split_batch_by,
          _enable_get_next_as_optional=enable_get_next_as_optional)
    else:
      return input_lib.DistributedDataset(
          dataset, input_workers,
          split_batch_by,
          _enable_get_next_as_optional=enable_get_next_as_optional)

  def _test_input_iteration(self,
                            input_type,
                            api_type,
                            iteration_type,
                            dataset_fn,
                            worker_device_pairs,
                            expected_values,
                            sess=None,
                            split_batch_by=None,
                            enable_get_next_as_optional=False):
    if iteration_type == "for_loop" and not context.executing_eagerly():
      self.skipTest("unsupported test combination.")

    if api_type == "wrap_into_iterator" and iteration_type == "for_loop":
      self.skipTest("unsupported test combination.")

    if api_type == "wrap_into_dataset" and input_type == "input_fn":
      self.skipTest("unsupported test combination.")

    devices = nest.flatten([ds for _, ds in worker_device_pairs])
    device_map = values.ReplicaDeviceMap(devices)
    input_workers = input_lib.InputWorkers(device_map, worker_device_pairs)

    if api_type == "wrap_into_iterator":
      iterator = self._wrap_iterator(
          input_type, dataset_fn, input_workers, devices, split_batch_by,
          enable_get_next_as_optional)
    else:
      # wrapping into a dataset:
      given_dataset = dataset_fn(distribute_lib.InputContext())
      dataset = self._wrap_dataset(input_type, given_dataset, input_workers,
                                   split_batch_by, enable_get_next_as_optional)

      if context.executing_eagerly():
        iterator = iter(dataset)
      else:
        # The dataset can be a tf.data.DatasetV1Adapter instance since we wrap
        # tf.data.DatasetV1 as a tf.data.DatasetV1Adapter instance when we
        # autoshard the dataset.
        if not isinstance(dataset, (dataset_ops.DatasetV1,
                                    dataset_ops.DatasetV1Adapter)):
          iterator = iter(dataset)
        else:
          iterator = dataset.make_one_shot_iterator()

    if iteration_type == "get_next":
      evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)
      if isinstance(iterator, input_lib.DistributedIteratorV1):
        evaluate(control_flow_ops.group(iterator.initialize()))
      else:
        evaluate(control_flow_ops.group(iterator._initializer))

      for expected_value in expected_values:
        next_element = iterator.get_next()
        computed_value = evaluate(
            [values.select_replica(r,
                                   next_element) for r in range(len(devices))])
        self.assertEqual(len(expected_value), len(computed_value))
        for i in range(len(expected_value)):
          self.assertAllEqual(expected_value[i], computed_value[i])

      with self.assertRaises(errors.OutOfRangeError):
        next_element = iterator.get_next()
        evaluate(
            [values.select_replica(r,
                                   next_element) for r in range(len(devices))])

      # After re-initializing the iterator, should be able to iterate again.
      if isinstance(iterator, input_lib.DistributedIteratorV1):
        evaluate(control_flow_ops.group(iterator.initialize()))
      else:
        evaluate(control_flow_ops.group(iterator._initializer))

      for expected_value in expected_values:
        next_element = iterator.get_next()
        computed_value = evaluate(
            [values.select_replica(r,
                                   next_element) for r in range(len(devices))])
        self.assertEqual(len(expected_value), len(computed_value))
        for i in range(len(expected_value)):
          self.assertAllEqual(expected_value[i], computed_value[i])

    if iteration_type == "for_loop" and context.executing_eagerly():
      actual_values = []
      for x in dataset:
        computed_value = self.evaluate(
            [values.select_replica(r, x) for r in range(len(devices))])
        actual_values.append(computed_value)
      for i, expected_value in enumerate(expected_values):
        self.assertEqual(len(expected_value), len(actual_values[i]))
        for j in range(len(expected_value)):
          self.assertAllEqual(expected_value[j], actual_values[i][j])


class DistributedIteratorSingleWorkerTest(DistributedIteratorTestBase,
                                          parameterized.TestCase):

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      api_type=["wrap_into_iterator", "wrap_into_dataset"],
      iteration_type=["get_next", "for_loop"]))
  def testOneDeviceCPU(self, input_type, api_type, iteration_type):
    worker_device_pairs = [("", ["/device:CPU:0"])]
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(10)

    expected_values = [[i] for i in range(10)]

    self._test_input_iteration(input_type, api_type, iteration_type, dataset_fn,
                               worker_device_pairs, expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      api_type=["wrap_into_iterator", "wrap_into_dataset"],
      iteration_type=["get_next", "for_loop"],
      required_gpus=1))
  def testTwoDevicesOneGPUOneCPU(self, input_type, api_type, iteration_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(10)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    self._test_input_iteration(input_type, api_type, iteration_type, dataset_fn,
                               worker_device_pairs, expected_values)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["input_fn", "dataset"],
      api_type=["wrap_into_iterator", "wrap_into_dataset"],
      iteration_type=["get_next", "for_loop"],
      required_gpus=1))
  def testTupleDataset(self, input_type, api_type, iteration_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]

    def dataset_fn(ctx):
      del ctx
      if tf2.enabled():
        dataset1 = dataset_ops.Dataset.range(10)
        dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x**2)
        return dataset_ops.Dataset.zip((dataset1, dataset2))
      else:
        dataset1 = dataset_ops.DatasetV2.range(10)
        dataset2 = dataset_ops.DatasetV2.range(10).map(lambda x: x**2)
        return dataset_ops.DatasetV2.zip((dataset1, dataset2))

    expected_values = [[(i, i**2), (i+1, (i+1)**2)] for i in range(0, 10, 2)]

    self._test_input_iteration(input_type, api_type, iteration_type, dataset_fn,
                               worker_device_pairs, expected_values)

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          required_gpus=1))
  def testUnevenDatasetBatches(self, input_type, api_type, iteration_type):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(9).batch(2)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(9).batch(2)

    # The last global batch only contains data for one replica.
    expected_values = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8], []]]
    self._test_input_iteration(input_type, api_type, iteration_type, dataset_fn,
                               worker_device_pairs, expected_values,
                               enable_get_next_as_optional=True)

  @combinations.generate(combinations.combine(
      mode=["graph", "eager"],
      input_type=["dataset"],
      api_type=["wrap_into_iterator", "wrap_into_dataset"],
      iteration_type=["get_next", "for_loop"],
      split_batch_by=[None, 2],
      required_gpus=1))
  def testBatchSplitting(self, input_type, api_type, iteration_type,
                         split_batch_by):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    batch_size = 10
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(100).batch(batch_size)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(100).batch(batch_size)

    updated_batch_size = (
        batch_size // split_batch_by if split_batch_by else batch_size)
    expected_values = [[range(i, i+updated_batch_size),
                        range(i+updated_batch_size, i+2*updated_batch_size)]
                       for i in range(0, 100, updated_batch_size*2)]

    self._test_input_iteration(input_type, api_type, iteration_type, dataset_fn,
                               worker_device_pairs, expected_values, sess=None,
                               split_batch_by=split_batch_by)


class DistributedIteratorMultiWorkerTest(
    multi_worker_test_base.MultiWorkerTestBase, DistributedIteratorTestBase,
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
      input_type=["input_fn", "dataset"],
      api_type=["wrap_into_iterator", "wrap_into_dataset"],
      iteration_type=["get_next", "for_loop"]))
  def testOneDevicePerWorker(self, input_type, api_type, iteration_type):
    worker_devices = self._cpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      if tf2.enabled():
        dataset_fn = lambda _: dataset_ops.DatasetV2.range(4)
      else:
        dataset_fn = lambda _: dataset_ops.Dataset.range(4)

      if input_type == "dataset":
        # Autosharded
        expected_values = [[0, 1], [2, 3]]
      else:
        expected_values = [[0, 0], [1, 1], [2, 2], [3, 3]]
      self._test_input_iteration(input_type, api_type, iteration_type,
                                 dataset_fn, worker_devices,
                                 expected_values, sess)

  @combinations.generate(combinations.combine(
      mode=["graph"],
      input_type=["input_fn", "dataset"],
      api_type=["wrap_into_iterator", "wrap_into_dataset"],
      iteration_type=["get_next", "for_loop"],
      required_gpus=1))
  def testTwoDevicesPerWorker(self, input_type, api_type, iteration_type):
    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      if tf2.enabled():
        dataset_fn = lambda _: dataset_ops.DatasetV2.range(4)
      else:
        dataset_fn = lambda _: dataset_ops.Dataset.range(4)

      if input_type == "dataset":
        # Autosharded
        expected_values = [[0, 2, 1, 3]]
      else:
        expected_values = [[0, 1, 0, 1], [2, 3, 2, 3]]
      self._test_input_iteration(input_type, api_type, iteration_type,
                                 dataset_fn, worker_devices,
                                 expected_values, sess)

  @combinations.generate(combinations.combine(
      mode=["graph"],
      input_type=["input_fn", "dataset"],
      api_type=["wrap_into_iterator", "wrap_into_dataset"],
      iteration_type=["get_next", "for_loop"]))
  def testTupleDataset(self, input_type, api_type, iteration_type):
    worker_devices = self._cpu_devices()
    with context.graph_mode(), self.cached_session() as sess:

      def dataset_fn(ctx):
        del ctx
        if tf2.enabled():
          dataset1 = dataset_ops.DatasetV2.range(4)
          dataset2 = dataset_ops.DatasetV2.range(4).map(lambda x: x**2)
          return dataset_ops.DatasetV2.zip((dataset1, dataset2))
        else:
          dataset1 = dataset_ops.Dataset.range(4)
          dataset2 = dataset_ops.Dataset.range(4).map(lambda x: x**2)
          return dataset_ops.Dataset.zip((dataset1, dataset2))

      if input_type == "dataset":
        # Autosharded
        expected_values = [[(0, 0), (1, 1)], [(2, 4), (3, 9)]]
      else:
        expected_values = [[(i, i**2), (i, i**2)] for i in range(0, 4)]
      self._test_input_iteration(input_type, api_type, iteration_type,
                                 dataset_fn, worker_devices, expected_values,
                                 sess)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          required_gpus=1))
  def testUnevenDatasetBatches(self, input_type, api_type, iteration_type):
    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      if tf2.enabled():
        dataset_fn = lambda _: dataset_ops.DatasetV2.range(9).batch(2)
      else:
        dataset_fn = lambda _: dataset_ops.Dataset.range(9).batch(2)
      if input_type == "dataset":
        # Autosharded
        expected_values = [[[0, 1], [4, 5], [2, 3], [6, 7]], [[8], [], [], []]]
      else:
        expected_values = [[[0, 1], [2, 3], [0, 1], [2, 3]],
                           [[4, 5], [6, 7], [4, 5], [6, 7]], [[8], [], [8], []]]
      self._test_input_iteration(input_type, api_type, iteration_type,
                                 dataset_fn, worker_devices, expected_values,
                                 sess, enable_get_next_as_optional=True)

  @combinations.generate(
      combinations.combine(
          mode=["graph"], input_type=["input_fn"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          required_gpus=1))
  def testDifferentDatasets(self, input_type, api_type, iteration_type):
    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), self.cached_session() as sess:

      def dataset_fn(ctx):
        if ctx.input_pipeline_id == 0:
          return dataset_ops.Dataset.range(8).batch(2)
        else:
          return dataset_ops.Dataset.range(9).batch(2)

      expected_values = [[[0, 1], [2, 3], [0, 1], [2, 3]],
                         [[4, 5], [6, 7], [4, 5], [6, 7]], [[], [], [8], []]]
      self._test_input_iteration(input_type, api_type, iteration_type,
                                 dataset_fn, worker_devices, expected_values,
                                 sess, enable_get_next_as_optional=True)

if __name__ == "__main__":
  test.main()
