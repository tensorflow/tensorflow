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

import collections
import json
import threading

from absl.testing import parameterized
import numpy as np

from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops.distribute_options import AutoShardPolicy
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_tensor as ragged_tensor_lib
from tensorflow.python.util import nest


class DistributedIteratorTestBase(test.TestCase):

  # The passed input_context is to create a sharded dataset in between-graph
  # case.
  def _wrap_iterator(self,
                     input_type,
                     dataset_or_input_fn,
                     input_workers,
                     devices,
                     split_batch_by,
                     strategy,
                     input_context=None):
    # The `input_context` passed in is to shard dataset for
    # MultiWorkerMirroredStrategy. It doesn't apply to in-graph case where
    # multiple InputContexts are needed.
    if input_type == "input_fn":
      self.assertIsNone(
          input_context,
          msg=("`The input_context` arg is only used to shard dataset in "
               "`MultiWorkerMirroredStrategy` when the input type is dataset."))

      input_contexts = []
      for i in range(input_workers.num_workers):
        input_contexts.append(
            distribute_lib.InputContext(
                # Note: `input_workers.num_workers` is always 1 in between-graph
                # case.
                num_input_pipelines=input_workers.num_workers,
                input_pipeline_id=i,
                num_replicas_in_sync=len(devices)))

      iterator = input_lib.InputFunctionIterator(
          dataset_or_input_fn,
          input_workers,
          input_contexts,
          strategy)
    else:
      iterator = input_lib.DatasetIterator(
          dataset_or_input_fn,
          input_workers,
          strategy,
          split_batch_by=split_batch_by,
          input_context=input_context)
    return iterator

  def _wrap_dataset(self,
                    input_type,
                    dataset,
                    input_workers,
                    split_batch_by,
                    strategy,
                    input_context=None):
    if isinstance(dataset, (dataset_ops.Dataset, dataset_ops.DatasetV1Adapter)):
      return input_lib.DistributedDatasetV1(
          dataset,
          input_workers,
          strategy,
          split_batch_by=split_batch_by,
          input_context=input_context)
    elif input_type == "dataset":
      return input_lib.DistributedDataset(
          dataset,
          input_workers,
          strategy,
          split_batch_by=split_batch_by,
          input_context=input_context)
    else:
      return strategy.experimental_distribute_datasets_from_function(dataset)

  def _test_input_iteration(self,
                            input_type,
                            api_type,
                            iteration_type,
                            dataset_or_input_fn,
                            worker_device_pairs,
                            expected_values,
                            strategy,
                            sess=None,
                            split_batch_by=None,
                            input_context=None):
    if iteration_type == "for_loop" and not context.executing_eagerly():
      self.skipTest("unsupported test combination.")

    if api_type == "wrap_into_iterator" and iteration_type == "for_loop":
      self.skipTest("unsupported test combination.")

    devices = nest.flatten([ds for _, ds in worker_device_pairs])
    input_workers = input_lib.InputWorkers(worker_device_pairs)

    if api_type == "wrap_into_iterator":
      iterator = self._wrap_iterator(
          input_type,
          dataset_or_input_fn,
          input_workers,
          devices,
          split_batch_by,
          strategy,
          input_context=input_context)
    else:
      # wrapping into a dataset:
      dataset = self._wrap_dataset(
          input_type,
          dataset_or_input_fn,
          input_workers,
          split_batch_by,
          strategy,
          input_context=input_context)

      if context.executing_eagerly():
        iterator = iter(dataset)
      else:
        if isinstance(dataset, input_lib.DistributedDatasetV1):
          iterator = dataset.make_initializable_iterator()
        else:
          self.skipTest("unsupported test combination")

    if iteration_type == "get_next":
      evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)
      if isinstance(iterator, input_lib.DistributedIteratorV1):
        evaluate(control_flow_ops.group(iterator.initializer))

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
        evaluate(control_flow_ops.group(iterator.initializer))
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

  def _create_dataset_or_input_fn(self, input_type, input_fn):
    if input_type == "input_fn":
      return input_fn
    else:
      return input_fn(distribute_lib.InputContext())


class DistributedIteratorSingleWorkerTest(DistributedIteratorTestBase,
                                          parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu
          ]))
  def testMultiDeviceIterInitialize(self, distribution):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    dataset_fn = lambda _: dataset_ops.DatasetV1.range(10)

    input_workers = input_lib.InputWorkers(worker_device_pairs)

    dist_dataset = input_lib.get_distributed_dataset(
        dataset_fn(distribute_lib.InputContext()), input_workers, distribution)

    iterator = dataset_ops.make_one_shot_iterator(dist_dataset)

    @def_function.function
    def init_func_for_iter():
      self.evaluate(iterator.initializer)

    init_func_for_iter()

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu
          ]))
  def testDatasetV2IterError(self, distribution):
    worker_device_pairs = [("", ["/device:CPU:0"])]
    input_workers = input_lib.InputWorkers(worker_device_pairs)
    dataset_fn = lambda _: dataset_ops.DatasetV2.range(10).batch(2)

    dist_dataset = input_lib.get_distributed_dataset(
        dataset_fn(distribute_lib.InputContext()), input_workers, distribution)

    with self.assertRaisesRegexp(RuntimeError,
                                 "or when eager execution is enabled"):
      iter(dist_dataset)

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu
          ],
          enable_get_next_as_optional=[True, False]))
  def testOneDeviceCPU(self, input_type, api_type, iteration_type, distribution,
                       enable_get_next_as_optional):
    worker_device_pairs = [("", ["/device:CPU:0"])]
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(10)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    expected_values = [[i] for i in range(10)]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution)

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu
          ],
          enable_get_next_as_optional=[True, False]))
  def testTwoDevicesOneGPUOneCPU(self, input_type, api_type, iteration_type,
                                 distribution, enable_get_next_as_optional):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(10)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    expected_values = [[i, i+1] for i in range(0, 10, 2)]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution)

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[strategy_combinations.tpu_strategy],
          enable_get_next_as_optional=[True, False]))
  def testTPU(self, input_type, api_type, iteration_type, distribution,
              enable_get_next_as_optional):
    worker_device_pairs = collections.OrderedDict()
    for tpu_device in distribution.extended.worker_devices:
      host_device = device_util.get_host_for_device(tpu_device)
      worker_device_pairs.setdefault(host_device, [])
      worker_device_pairs[host_device].append(tpu_device)
    worker_device_pairs = worker_device_pairs.items()
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(10)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    expected_values = [[i, i + 1] for i in range(0, 10, 2)]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution)

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu
          ],
          enable_get_next_as_optional=[True, False]))
  def testTupleDataset(self, input_type, api_type, iteration_type, distribution,
                       enable_get_next_as_optional):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]

    def dataset_fn(ctx):
      del ctx
      if tf2.enabled():
        dataset1 = dataset_ops.DatasetV2.range(10)
        dataset2 = dataset_ops.DatasetV2.range(10).map(lambda x: x**2)
        return dataset_ops.DatasetV2.zip((dataset1, dataset2))
      else:
        dataset1 = dataset_ops.Dataset.range(10)
        dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x**2)
        return dataset_ops.Dataset.zip((dataset1, dataset2))
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    expected_values = [[(i, i**2), (i+1, (i+1)**2)] for i in range(0, 10, 2)]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu
          ]))
  def testIterableIterator(self, distribution):
    worker_device_pairs = [("", ["/device:CPU:0"])]
    input_workers = input_lib.InputWorkers(worker_device_pairs)

    dataset = dataset_ops.DatasetV2.range(10)
    dist_dataset = input_lib.get_distributed_dataset(dataset, input_workers,
                                                     distribution)

    iterator = iter(dist_dataset)
    for i, element in enumerate(iterator):
      self.assertEqual(i, element.numpy())

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          drop_remainder=[True, False],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu
          ]))
  def testUnevenDatasetBatches(self, input_type, api_type, iteration_type,
                               drop_remainder, distribution):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(9).batch(  # pylint: disable=g-long-lambda
          2, drop_remainder=drop_remainder)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(9).batch(  # pylint: disable=g-long-lambda
          2, drop_remainder=drop_remainder)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    # The last global batch only contains data for one replica.
    if drop_remainder:
      expected_values = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    else:
      expected_values = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8], []]]
    distribution.extended.experimental_enable_get_next_as_optional = True
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution)

  @combinations.generate(
      combinations.combine(
          mode=["graph", "eager"],
          input_type=["dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          split_batch_by=[None, 2],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu
          ],
          enable_get_next_as_optional=[True, False]))
  def testBatchSplitting(self, input_type, api_type, iteration_type,
                         split_batch_by, distribution,
                         enable_get_next_as_optional):
    worker_device_pairs = [("", ["/device:GPU:0", "/device:CPU:0"])]
    batch_size = 10
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(100).batch(batch_size)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(100).batch(batch_size)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    updated_batch_size = (
        batch_size // split_batch_by if split_batch_by else batch_size)
    expected_values = [[range(i, i+updated_batch_size),
                        range(i+updated_batch_size, i+2*updated_batch_size)]
                       for i in range(0, 100, updated_batch_size*2)]

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    self._test_input_iteration(
        input_type,
        api_type,
        iteration_type,
        dataset_or_input_fn,
        worker_device_pairs,
        expected_values,
        distribution,
        sess=None,
        split_batch_by=split_batch_by)


class DistributedIteratorTensorTypeTest(DistributedIteratorTestBase,
                                        parameterized.TestCase):
  """Tests for DistributedDataset with non-dense tensors."""

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
          ],
          input_type=["dataset", "input_fn"],
          drop_remainder=[False, True],
          defun_type=["lambda", "tf_function"],
      ))
  def testRaggedSparse(self, distribution, input_type, drop_remainder,
                       defun_type):
    """Test with `RaggedTensor`s and `SparseTensor`s."""
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")

    defun = {"lambda": lambda f: f,
             "tf_function": def_function.function}[defun_type]
    distribution.extended.experimental_enable_get_next_as_optional = True
    global_batch_size = 8

    def dataset_fn(ctx=None):
      ctx = ctx or distribute_lib.InputContext()
      batch_size = ctx.get_per_replica_batch_size(global_batch_size)
      # Use 20 which isn't divisible by 8 to test partial batch behavior.
      row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
      ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
          np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
      dataset = dataset_ops.DatasetV2.from_tensor_slices({
          "dense": ragged_tensor.to_tensor(),
          "ragged": ragged_tensor,
          "sparse": ragged_tensor.to_sparse(),
      })
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
      return dataset.batch(batch_size, drop_remainder=drop_remainder)

    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)
    dataset = self._wrap_dataset(input_type, dataset_or_input_fn,
                                 distribution.extended._input_workers,
                                 len(distribution.extended.worker_devices),
                                 distribution)
    # Assert that the tensors are rebatched and sparsity is preserved.
    per_replica_batch = defun(lambda x: next(iter(x)))(dataset)
    self.assertAllEqual(
        values.select_replica(0, per_replica_batch["dense"]),
        [[0., 0., 0.], [1., 0., 0.], [2., 2., 0.], [3., 3., 3.]])
    self.assertAllEqual(
        values.select_replica(1, per_replica_batch["dense"]),
        [[0., 0., 0.], [5., 0., 0.], [6., 6., 0.], [7., 7., 7.]])
    # Transitively check the ragged and sparse tensors by densification.
    for i in range(2):
      self.assertLen(
          values.select_replica(i, per_replica_batch["ragged"]).values, 6)
      self.assertAllEqual(
          values.select_replica(i, per_replica_batch["ragged"]).to_tensor(),
          values.select_replica(i, per_replica_batch["dense"]))
      self.assertLen(
          values.select_replica(i, per_replica_batch["sparse"]).indices, 6)
      self.assertAllEqual(
          sparse_ops.sparse_tensor_to_dense(
              values.select_replica(i, per_replica_batch["sparse"])),
          values.select_replica(i, per_replica_batch["dense"]))
    # Iterate through all the batches and sum them up.
    def sum_batch(per_replica_features):
      """Sums the `PerReplica` values in the `per_replica_features` map."""

      def map_fn(per_replica_values):
        per_replica_sums = distribution.run(
            (lambda x: math_ops.reduce_sum(x.values)) if all(
                map(sparse_tensor.is_sparse, per_replica_values.values)) else
            math_ops.reduce_sum, (per_replica_values,))
        return distribution.reduce(
            reduce_util.ReduceOp.SUM, per_replica_sums, axis=None)

      return nest.map_structure(map_fn, per_replica_features)

    def _reduce(state, batch):
      sums = sum_batch(batch)
      return {name: value + sums[name] for name, value in state.items()}

    def sum_for_loop(dataset):
      sums = {"dense": 0., "ragged": 0., "sparse": 0.}
      for batch in dataset:
        sums = _reduce(sums, batch)
      return sums

    def sum_while_loop(iterator, reduce_fn):
      sums = {"dense": 0., "ragged": 0., "sparse": 0.}
      while True:
        try:
          sums = reduce_fn(sums, iterator)
        except (StopIteration, errors.OutOfRangeError):
          return sums

    while_sums = sum_while_loop(
        iter(dataset),
        defun(lambda state, iterator: _reduce(state, next(iterator))))
    self.assertAllEqual(
        nest.flatten(while_sums),
        # When there's no partial batch, the sum is smaller.
        [200. if drop_remainder else 310.] * 3)
    for_sums = defun(sum_for_loop)(dataset)
    # For loops always call get next as optional inside tf functions, so we
    # expect 310 here when using an input function (as there are 5 batches of
    # size 4 round robined over 2 replicas.
    expected_for_sum = 200.
    if (not drop_remainder or (
        defun_type == "tf_function" and input_type == "input_fn")):
      expected_for_sum = 310.
    self.assertAllEqual(nest.flatten(for_sums), [expected_for_sum] * 3)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu
          ],
          input_type=["dataset", "input_fn"],
          drop_remainder=[False, True],
          tensor_type=["sparse", "ragged"],
          enable_get_next_as_optional=[True, False]
      ))
  def testRaggedSparseGetNextAsOptional(
      self, distribution, input_type, drop_remainder, tensor_type,
      enable_get_next_as_optional):
    """Test with `RaggedTensor`s and `SparseTensor`s."""
    if not tf2.enabled():
      self.skipTest("Only V2 is supported.")

    distribution.extended.experimental_enable_get_next_as_optional = (
        enable_get_next_as_optional)
    global_batch_size = 8

    def dataset_fn(ctx=None):
      ctx = ctx or distribute_lib.InputContext()
      batch_size = ctx.get_per_replica_batch_size(global_batch_size)
      # Use 20 which isn't divisible by 8 to test partial batch behavior.
      row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
      ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(
          np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
      dataset = dataset_ops.DatasetV2.from_tensor_slices({
          tensor_type: (ragged_tensor if tensor_type == "ragged" else
                        ragged_tensor.to_sparse()),
      })
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
      return dataset.batch(batch_size, drop_remainder=drop_remainder)

    if input_type == "dataset":
      ds = distribution.experimental_distribute_dataset(
          dataset_fn(distribute_lib.InputContext()))
    else:
      ds = distribution.experimental_distribute_datasets_from_function(
          dataset_fn)
    iterator = iter(ds)

    self.assertEqual(iterator._enable_get_next_as_optional,
                     (not drop_remainder) and enable_get_next_as_optional)


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
      input_type=["dataset"],
      api_type=["wrap_into_iterator", "wrap_into_dataset"],
      iteration_type=["get_next", "for_loop"],
      auto_shard_policy=[AutoShardPolicy.AUTO, AutoShardPolicy.OFF]))
  def testAutoshardingOption(self, input_type, api_type, iteration_type,
                             auto_shard_policy):
    ds_option = dataset_ops.Options()
    ds_option.experimental_distribute.auto_shard_policy = auto_shard_policy
    if tf2.enabled():
      dataset_fn = (
          lambda _: dataset_ops.DatasetV2.range(4).with_options(ds_option))
    else:
      dataset_fn = (
          lambda _: dataset_ops.Dataset.range(4).with_options(ds_option))
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    strategy = mirrored_strategy.MirroredStrategy(
        devices=(self._cpu_devices()[0][1] + self._cpu_devices()[1][1]),
        cross_device_ops=cross_device_ops_lib.MultiWorkerAllReduce(
            ["/job:worker/task:0", "/job:worker/task:1"], 1))
    worker_devices = self._cpu_devices()
    with context.graph_mode(), self.cached_session() as sess:
      if auto_shard_policy == AutoShardPolicy.AUTO:
        expected_values = [[0, 1], [2, 3]]
      else:
        expected_values = [[0, 0], [1, 1], [2, 2], [3, 3]]
      self._test_input_iteration(input_type, api_type, iteration_type,
                                 dataset_or_input_fn, worker_devices,
                                 expected_values, strategy, sess)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          enable_get_next_as_optional=[True, False]))
  def testOneDevicePerWorker(self, input_type, api_type, iteration_type,
                             enable_get_next_as_optional):
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(4)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(4)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    strategy = mirrored_strategy.MirroredStrategy(
        devices=(self._cpu_devices()[0][1] + self._cpu_devices()[1][1]),
        cross_device_ops=cross_device_ops_lib.MultiWorkerAllReduce(
            ["/job:worker/task:0", "/job:worker/task:1"], 1))
    worker_devices = self._cpu_devices()
    with context.graph_mode(), strategy.scope(), self.cached_session() as sess:

      if input_type == "dataset":
        # Autosharded
        expected_values = [[0, 1], [2, 3]]
      else:
        expected_values = [[0, 0], [1, 1], [2, 2], [3, 3]]
      strategy.extended.experimental_enable_get_next_as_optional = (
          enable_get_next_as_optional)
      self._test_input_iteration(
          input_type,
          api_type,
          iteration_type,
          dataset_or_input_fn,
          worker_devices,
          expected_values,
          strategy,
          sess=sess)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          enable_get_next_as_optional=[True, False],
          required_gpus=1))
  def testTwoDevicesPerWorker(self, input_type, api_type, iteration_type,
                              enable_get_next_as_optional):
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(4)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(4)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    strategy = mirrored_strategy.MirroredStrategy(
        devices=(self._cpu_and_one_gpu_devices()[0][1] +
                 self._cpu_and_one_gpu_devices()[1][1]),
        cross_device_ops=cross_device_ops_lib.MultiWorkerAllReduce(
            ["/job:worker/task:0", "/job:worker/task:1"], 2))
    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), strategy.scope(), self.cached_session() as sess:

      if input_type == "dataset":
        # Autosharded
        expected_values = [[0, 2, 1, 3]]
      else:
        expected_values = [[0, 1, 0, 1], [2, 3, 2, 3]]
      strategy.extended.experimental_enable_get_next_as_optional = (
          enable_get_next_as_optional)
      self._test_input_iteration(
          input_type,
          api_type,
          iteration_type,
          dataset_or_input_fn,
          worker_devices,
          expected_values,
          strategy,
          sess=sess)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          enable_get_next_as_optional=[True, False]))
  def testTupleDataset(self, input_type, api_type, iteration_type,
                       enable_get_next_as_optional):
    strategy = mirrored_strategy.MirroredStrategy(
        devices=(self._cpu_devices()[0][1] + self._cpu_devices()[1][1]),
        cross_device_ops=cross_device_ops_lib.MultiWorkerAllReduce(
            ["/job:worker/task:0", "/job:worker/task:1"], 1))
    worker_devices = self._cpu_devices()

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
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    with context.graph_mode(), strategy.scope(), self.cached_session() as sess:

      if input_type == "dataset":
        # Autosharded
        expected_values = [[(0, 0), (1, 1)], [(2, 4), (3, 9)]]
      else:
        expected_values = [[(i, i**2), (i, i**2)] for i in range(0, 4)]
      strategy.extended.experimental_enable_get_next_as_optional = (
          enable_get_next_as_optional)
      self._test_input_iteration(
          input_type,
          api_type,
          iteration_type,
          dataset_or_input_fn,
          worker_devices,
          expected_values,
          strategy,
          sess=sess)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          required_gpus=1))
  def testUnevenDatasetBatches(self, input_type, api_type, iteration_type):
    strategy = mirrored_strategy.MirroredStrategy(
        devices=(self._cpu_and_one_gpu_devices()[0][1] +
                 self._cpu_and_one_gpu_devices()[1][1]),
        cross_device_ops=cross_device_ops_lib.MultiWorkerAllReduce(
            ["/job:worker/task:0", "/job:worker/task:1"], 2))
    if tf2.enabled():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(9).batch(2)
    else:
      dataset_fn = lambda _: dataset_ops.Dataset.range(9).batch(2)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), strategy.scope(), self.cached_session() as sess:
      if input_type == "dataset":
        # Autosharded
        expected_values = [[[0, 1], [4, 5], [2, 3], [6, 7]], [[8], [], [], []]]
      else:
        expected_values = [[[0, 1], [2, 3], [0, 1], [2, 3]],
                           [[4, 5], [6, 7], [4, 5], [6, 7]], [[8], [], [8], []]]
      strategy.extended.experimental_enable_get_next_as_optional = True
      self._test_input_iteration(
          input_type,
          api_type,
          iteration_type,
          dataset_or_input_fn,
          worker_devices,
          expected_values,
          strategy,
          sess=sess)

  @combinations.generate(
      combinations.combine(
          mode=["graph"],
          input_type=["input_fn", "dataset"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next"],
          strategy_cls=[
              collective_all_reduce_strategy.CollectiveAllReduceStrategy,
              parameter_server_strategy.ParameterServerStrategy,
          ],
          required_gpus=0))
  def testUnevenDatasetBatchesBetweenGraph(self, input_type, api_type,
                                           iteration_type, strategy_cls):
    self.skipTest("broken test to be fixed")
    if api_type == "wrap_into_dataset" and input_type == "input_fn":
      self.skipTest("unsupported test combination.")
    if tf2.enabled():
      # The V2 tests are skipped since we don't support creating an
      # iterator for DistributedDataset in graph mode.
      self.skipTest("unsupported test combination")
    # Environment variable is global, we need locking when patching TF_CONFIG.
    lock = threading.Lock()

    def _worker_fn(task_type, task_id, num_gpus):
      del num_gpus
      tf_config = {
          "cluster": self._cluster_spec,
          "task": {
              "type": task_type,
              "index": task_id
          }
      }
      with context.graph_mode(), lock, test.mock.patch.dict(
          "os.environ", {"TF_CONFIG": json.dumps(tf_config)}):
        strategy = strategy_cls()
      with context.graph_mode(), strategy.scope(), self.cached_session(
          target="grpc://" + self._cluster_spec[task_type][task_id]) as sess:
        if tf2.enabled():
          dataset_fn = lambda _: dataset_ops.DatasetV2.range(5).batch(2)
        else:
          dataset_fn = lambda _: dataset_ops.Dataset.range(5).batch(2)
        dataset_or_input_fn = self._create_dataset_or_input_fn(
            input_type, dataset_fn)
        if (input_type == "dataset" and strategy_cls is
            collective_all_reduce_strategy.CollectiveAllReduceStrategy):
          # Autosharded
          if task_id == 0:
            expected_values = [[[0, 1]], [[4]]]
          else:
            expected_values = [[[2, 3]], [[]]]

          # input_context is for between-graph auto-sharding.
          input_context = distribute_lib.InputContext(
              num_input_pipelines=2,
              input_pipeline_id=task_id,
              num_replicas_in_sync=2)
        else:
          expected_values = [[[0, 1]], [[2, 3]], [[4]]]
          input_context = None

        strategy.extended.experimental_enable_get_next_as_optional = True
        self._test_input_iteration(
            input_type,
            api_type,
            iteration_type,
            dataset_or_input_fn,
            [("/job:%s/task:%d" %
              (task_type, task_id), strategy.extended.worker_devices)],
            expected_values,
            strategy,
            sess=sess,
            input_context=input_context)

    self._run_between_graph_clients(_worker_fn, self._cluster_spec, 0)

  @combinations.generate(
      combinations.combine(
          mode=["graph"], input_type=["input_fn"],
          api_type=["wrap_into_iterator", "wrap_into_dataset"],
          iteration_type=["get_next", "for_loop"],
          required_gpus=1))
  def testDifferentDatasets(self, input_type, api_type, iteration_type):
    def dataset_fn(ctx):
      if ctx.input_pipeline_id == 0:
        return dataset_ops.Dataset.range(8).batch(2)
      else:
        return dataset_ops.Dataset.range(9).batch(2)
    dataset_or_input_fn = self._create_dataset_or_input_fn(
        input_type, dataset_fn)

    strategy = mirrored_strategy.MirroredStrategy(
        devices=(self._cpu_and_one_gpu_devices()[0][1] +
                 self._cpu_and_one_gpu_devices()[1][1]),
        cross_device_ops=cross_device_ops_lib.MultiWorkerAllReduce(
            ["/job:worker/task:0", "/job:worker/task:1"], 2))
    worker_devices = self._cpu_and_one_gpu_devices()
    with context.graph_mode(), strategy.scope(), self.cached_session() as sess:

      expected_values = [[[0, 1], [2, 3], [0, 1], [2, 3]],
                         [[4, 5], [6, 7], [4, 5], [6, 7]], [[], [], [8], []]]
      strategy.extended.experimental_enable_get_next_as_optional = True
      self._test_input_iteration(
          input_type,
          api_type,
          iteration_type,
          dataset_or_input_fn,
          worker_devices,
          expected_values,
          strategy,
          sess=sess)


class InputTypeSpecTest(test.TestCase, parameterized.TestCase):

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
          input_type=["dataset", "dataset_fn"],
      ))
  def testInputSignatureForPerReplicaValues(self, distribution, input_type):
    def dataset_fn(ctx):
      del ctx  # unused
      return dataset_ops.DatasetV2.from_tensor_slices(
          np.ones([10, 12]).astype(np.float32)).batch(4)

    if input_type == "dataset":
      ds = distribution.experimental_distribute_dataset(
          dataset_fn(distribute_lib.InputContext()))
      type_spec = ds.element_spec
    else:
      ds = distribution.experimental_distribute_datasets_from_function(
          dataset_fn)
      iterator = iter(ds)
      type_spec = iterator.element_spec

    @def_function.function(input_signature=[type_spec])
    def process_inputs(inputs):
      distribution.run(lambda inputs: inputs, args=(inputs,))

    for x in ds:
      process_inputs(x)

  @combinations.generate(
      combinations.combine(
          mode=["eager"],
          distribution=[
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_one_cpu,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.central_storage_strategy_with_two_gpus,
          ],
      ))
  def testInputSignatureForNestedPerReplicaValues(self, distribution):
    a = np.ones((10, 2)) * 5
    b = np.ones((10, 3)) * 6
    dataset = dataset_ops.DatasetV2.from_tensor_slices((a, b)).batch(2)

    dist_dataset = distribution.experimental_distribute_dataset(dataset)

    @def_function.function(input_signature=[dist_dataset.element_spec])
    def process_inputs(inputs):
      distribution.run(lambda inputs: inputs, args=(inputs,))

    for x in dist_dataset:
      process_inputs(x)


if __name__ == "__main__":
  test.main()
