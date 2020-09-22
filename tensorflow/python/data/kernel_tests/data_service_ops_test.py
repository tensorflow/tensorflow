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
"""Tests for tf.data service ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
import time

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


# This will be resolved to a tmp directory by `start_dispatch_server`.
TMP_WORK_DIR = "tmp_work_dir_placeholder"
# `""` indicates not to use a work directory.
NO_WORK_DIR = ""


def _address_from_target(target):
  # Targets are in the format <protocol>://<address>
  return target.split("://")[1]


def _make_distributed_dataset(dataset,
                              dispatcher,
                              job_name=None,
                              max_outstanding_requests=None):
  return dataset.apply(
      data_service_ops._distribute(
          "parallel_epochs",
          dispatcher.target,
          job_name=job_name,
          max_outstanding_requests=max_outstanding_requests,
          task_refresh_interval_hint_ms=20))


def _all_cluster_configurations():
  with_work_dir = combinations.combine(
      work_dir=TMP_WORK_DIR, fault_tolerant_mode=[True, False])
  without_work_dir = combinations.combine(
      work_dir=NO_WORK_DIR, fault_tolerant_mode=False)
  return with_work_dir + without_work_dir


def _make_distributed_range_dataset(num_elements,
                                    dispatcher,
                                    job_name=None,
                                    max_outstanding_requests=None):
  """Creates a distributed dataset.

  Args:
    num_elements: The number of elements in the range dataset that will be
      distributed.
    dispatcher: The dispatcher to distribute to.
    job_name: Optional job name for the distributed dataset.
    max_outstanding_requests: Optional limit on the number of outstanding
      requests.

  Returns:
    The created dataset.
  """
  dataset = dataset_ops.Dataset.range(num_elements)
  return _make_distributed_dataset(dataset, dispatcher, job_name,
                                   max_outstanding_requests)


class DataServiceOpsTest(test_base.DatasetTestBase, parameterized.TestCase):

  def start_dispatch_server(self,
                            name="",
                            port=0,
                            work_dir=TMP_WORK_DIR,
                            fault_tolerant_mode=True,
                            job_gc_check_interval_ms=None,
                            job_gc_timeout_ms=None):
    # If a test starts multiple independent dispatch servers, it should give
    # them different `name` values.
    work_dir = os.path.join(self.get_temp_dir(), "work_dir_",
                            name) if work_dir is TMP_WORK_DIR else work_dir
    return server_lib.DispatchServer(
        server_lib.DispatcherConfig(
            port=port,
            work_dir=work_dir,
            fault_tolerant_mode=fault_tolerant_mode,
            job_gc_check_interval_ms=job_gc_check_interval_ms,
            job_gc_timeout_ms=job_gc_timeout_ms))

  def start_worker_server(self, dispatcher, port=0):
    return server_lib.WorkerServer(
        server_lib.WorkerConfig(
            dispatcher_address=_address_from_target(dispatcher.target),
            port=port,
            heartbeat_interval_ms=200))

  def restart_dispatcher(self, dispatcher):
    """Stops `dispatcher` and returns a new dispatcher with the same port."""
    port = int(_address_from_target(dispatcher.target).split(":")[1])
    dispatcher._stop()
    return self.start_dispatch_server(
        port=port,
        work_dir=dispatcher._config.work_dir,
        fault_tolerant_mode=dispatcher._config.fault_tolerant_mode)

  def restart_worker(self, worker, dispatcher, use_same_port=True):
    """Stops `worker` and returns a new worker."""
    port = 0
    if use_same_port:
      port = int(worker._address.split(":")[1])
    worker._stop()
    return self.start_worker_server(dispatcher, port)

  def start_cluster(self,
                    num_workers,
                    name="",
                    work_dir=TMP_WORK_DIR,
                    fault_tolerant_mode=True):
    """Creates and starts a tf.data service cluster."""
    dispatcher = self.start_dispatch_server(
        name=name, work_dir=work_dir, fault_tolerant_mode=fault_tolerant_mode)
    workers = [self.start_worker_server(dispatcher) for _ in range(num_workers)]
    return dispatcher, workers

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         _all_cluster_configurations()))
  def testDistributeBasic(self, work_dir, fault_tolerant_mode):
    dispatcher, workers = self.start_cluster(  # to avoid gcing workers, pylint: disable=unused-variable
        1,
        work_dir=work_dir,
        fault_tolerant_mode=fault_tolerant_mode)
    num_elements = 10
    ds = _make_distributed_range_dataset(10, dispatcher)
    results = [elem.numpy() for elem in ds]
    self.assertEqual(list(range(num_elements)), results)

  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherStop(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 100
    ds = _make_distributed_range_dataset(num_elements, dispatcher)
    iterator = iter(ds)
    results = []
    results.append(next(iterator).numpy())
    dispatcher._stop()
    # After the dispatcher dies, the worker should continue providing the rest
    # of the dataset's elements.
    for _ in range(num_elements - 1):
      results.append(next(iterator).numpy())
    self.assertEqual(results, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherRestartBeforeReading(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 100
    ds = _make_distributed_range_dataset(num_elements, dispatcher)
    dispatcher = self.restart_dispatcher(dispatcher)

    self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherRestartDuringReading(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 100
    ds = _make_distributed_range_dataset(num_elements, dispatcher)
    iterator = iter(ds)
    results = []
    for _ in range(num_elements // 2):
      results.append(next(iterator).numpy())
    dispatcher = self.restart_dispatcher(dispatcher)
    for elem in iterator:
      results.append(elem.numpy())

    self.assertEqual(list(range(num_elements)), results)

  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherRestartBetweenIterations(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 100
    ds = _make_distributed_range_dataset(100, dispatcher)
    self.assertDatasetProduces(ds, list(range(num_elements)))
    dispatcher = self.restart_dispatcher(dispatcher)
    self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherManyRestarts(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements_start = 10
    num_elements_end = 15
    datasets = []
    for num_elements in range(num_elements_start, num_elements_end):
      datasets.append(_make_distributed_range_dataset(num_elements, dispatcher))
      dispatcher = self.restart_dispatcher(dispatcher)
    for ds, num_elements in zip(datasets,
                                range(num_elements_start, num_elements_end)):
      self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testDispatcherAndWorkerRestart(self):
    dispatcher, [worker] = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 100
    ds = dataset_ops.Dataset.range(num_elements)

    def restart():
      return (self.restart_dispatcher(dispatcher),
              self.restart_worker(worker, dispatcher))

    ds = _make_distributed_dataset(ds, dispatcher)
    dispatcher, worker = restart()
    self.assertDatasetProduces(ds, list(range(num_elements)))
    dispatcher, worker = restart()
    self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeSparse(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    element = sparse_tensor.SparseTensor(
        indices=[[0]],
        values=constant_op.constant([0], dtype=dtypes.int32),
        dense_shape=[1])
    ds = dataset_ops.Dataset.from_tensors(element)
    ds = _make_distributed_dataset(ds, dispatcher)
    results = [sparse_ops.sparse_tensor_to_dense(elem) for elem in ds]
    self.assertAllEqual(results, [[0]])

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeRagged(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    ds = dataset_ops.Dataset.from_tensor_slices([1, 5, 3, 2, 8])
    ds = ds.map(math_ops.range)
    ds = ds.apply(batching.dense_to_ragged_batch(2))
    ds = _make_distributed_dataset(ds, dispatcher)
    results = [elem.to_tensor() for elem in ds]
    self.assertAllEqual(results[0], [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]])
    self.assertAllEqual(results[1], [[0, 1, 2], [0, 1, 0]])
    self.assertAllEqual(results[2], [[0, 1, 2, 3, 4, 5, 6, 7]])

  @combinations.generate(test_base.eager_only_combinations())
  def testDifferentShuffleOrders(self):
    random_seed.set_random_seed(None)
    num_elements = 100
    dispatcher, workers = self.start_cluster(2)  # to avoid gcing workers, pylint: disable=unused-variable
    ds = dataset_ops.Dataset.range(num_elements)
    ds = ds.shuffle(num_elements)
    ds = _make_distributed_dataset(ds, dispatcher)
    output = [elem.numpy() for elem in ds]

    # The output will be two sequences of range(num_elements)
    # non-deterministically interleaved together. If the orders of the elements
    # were the same, first_order and second_order computed below will be equal.
    first_order = {}
    second_order = {}
    for element in output:
      if element in first_order:
        second_order[element] = len(second_order)
      else:
        first_order[element] = len(first_order)
    self.assertNotEqual(first_order, second_order)

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleEpochs(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 3
    ds = _make_distributed_range_dataset(num_elements, dispatcher)
    for _ in range(10):
      self.assertEqual(list(range(num_elements)), [elem.numpy() for elem in ds])

  @combinations.generate(test_base.eager_only_combinations())
  def testRepeatedDataset(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 10
    num_repetitions = 5
    ds = _make_distributed_range_dataset(num_elements, dispatcher)
    ds = ds.repeat(num_repetitions)
    self.assertDatasetProduces(
        ds, expected_output=num_repetitions * list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testConcurrentEpoch(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 10
    num_datasets = 3
    iterators = []
    results = []
    for _ in range(num_datasets):
      ds = _make_distributed_range_dataset(num_elements, dispatcher)
      iterators.append(iter(ds))
      results.append([])

    for _ in range(num_elements):
      for dataset_ind in range(num_datasets):
        result = next(iterators[dataset_ind]).numpy()
        results[dataset_ind].append(result)
    for result in results:
      self.assertEqual(list(range(num_elements)), result)

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedEpoch(self):
    self.skipTest("Not yet implemented")
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 10
    num_iterators = 3
    ds = _make_distributed_range_dataset(num_elements, dispatcher)
    result = []
    iterators = []
    for _ in range(num_iterators):
      iterators.append(iter(ds))

    # Alternate reading between the iterators.
    for _ in range(2):
      for it in iterators:
        result.append(next(it).numpy())

    # Drain the rest of the elements.
    for it in iterators:
      for elem in it:
        result.append(elem.numpy())

    self.assertCountEqual(list(range(num_elements)), result)

  @combinations.generate(test_base.eager_only_combinations())
  def testMultiWorker(self):
    num_workers = 3
    dispatcher, workers = self.start_cluster(num_workers)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 10
    ds = _make_distributed_range_dataset(num_elements, dispatcher)
    results = [elem.numpy() for elem in ds]
    self.assertCountEqual(num_workers * list(range(num_elements)), results)

  @combinations.generate(test_base.eager_only_combinations())
  def testStartServersLate(self):
    # Test that the data service client performs retries instead of failing when
    # the dataset is created before the master and worker are started.
    try:
      import portpicker  # pylint: disable=g-import-not-at-top
      dispatcher_port = portpicker.pick_unused_port()
    except:
      raise self.skipTest("Flakes in portpicker library do not represent "
                          "TensorFlow errors.")
    dispatcher = server_lib.DispatchServer(
        server_lib.DispatcherConfig(port=dispatcher_port), start=False)
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(
            dispatcher_address=_address_from_target(dispatcher.target), port=0),
        start=False)

    def start_servers():
      time.sleep(1)
      dispatcher.start()
      worker.start()

    start_servers_thread = threading.Thread(target=start_servers, daemon=True)
    start_servers_thread.start()

    num_elements = 10
    ds = _make_distributed_range_dataset(num_elements, dispatcher)
    results = [elem.numpy() for elem in ds]
    self.assertEqual(list(range(num_elements)), results)
    start_servers_thread.join()

  @combinations.generate(test_base.eager_only_combinations())
  def testAddWorkerMidJob(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 100
    ds = _make_distributed_range_dataset(num_elements, dispatcher)
    iterator = iter(ds)
    results = []
    # Read halfway through the dataset.
    for _ in range(num_elements // 2):
      results.append(next(iterator).numpy())

    new_worker = self.start_worker_server(dispatcher)  # to avoid gcing workers, pylint: disable=unused-variable
    # Wait for the new worker to register with the dispatcher.
    while dispatcher._num_workers() < 2:
      time.sleep(10 / 1000)  # 10ms

    for elem in iterator:
      results.append(elem.numpy())

    self.assertCountEqual(2 * list(range(num_elements)), results)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(use_same_port=[True, False]),
                         _all_cluster_configurations()))
  def testRestartWorker(self, use_same_port, work_dir, fault_tolerant_mode):
    dispatcher, [worker] = self.start_cluster(
        1, work_dir=work_dir, fault_tolerant_mode=fault_tolerant_mode)
    num_elements = 100
    ds = _make_distributed_range_dataset(num_elements, dispatcher)
    iterator = iter(ds)
    # Read halfway through the dataset.
    midpoint = num_elements // 2
    for i in range(midpoint):
      self.assertEqual(i, next(iterator).numpy())

    # Stop the original worker and start a new one.
    worker = self.restart_worker(worker, dispatcher, use_same_port)

    # There may have been some elements prefetched from the first worker
    # before it was stopped.
    while True:
      val = next(iterator).numpy()
      if val == 0:
        break

    # The dataset starts over now that we read from the new worker.
    # TODO(b/157086991): Iterate until end of sequence when we support
    # detecting lost workers.
    for i in range(1, num_elements // 2):
      val = next(iterator).numpy()
      self.assertEqual(i, val)

  @combinations.generate(test_base.eager_only_combinations())
  def testMaxOutstandingRequests(self):
    num_workers = 3
    dispatcher, workers = self.start_cluster(num_workers)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 10
    ds = _make_distributed_range_dataset(
        num_elements, dispatcher, max_outstanding_requests=1)
    self.assertCountEqual(num_workers * list(range(num_elements)),
                          self.getDatasetOutput(ds))

  @combinations.generate(test_base.eager_only_combinations())
  def testInsideFunction(self):
    num_workers = 3
    dispatcher, workers = self.start_cluster(num_workers)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 10

    @def_function.function
    def f():
      ds = _make_distributed_range_dataset(num_elements, dispatcher)
      result = tensor_array_ops.TensorArray(
          dtypes.int64, size=num_workers * num_elements, dynamic_size=True)
      i = 0
      for elem in ds:
        result = result.write(i, elem)
        i += 1
      return result.stack()

    result = list(f().numpy())
    self.assertCountEqual(num_workers * list(range(num_elements)), result)

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedJobName(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 100

    def make_ds():
      return dataset_ops.Dataset.range(num_elements).shuffle(num_elements)

    ds1 = _make_distributed_dataset(make_ds(), dispatcher, job_name="job_name")
    ds2 = _make_distributed_dataset(make_ds(), dispatcher, job_name="job_name")
    iter1 = iter(ds1)
    iter2 = iter(ds2)
    results = []
    for _ in range(num_elements // 5):
      results.append(next(iter1).numpy())
      results.append(next(iter2).numpy())
    for elem in iter1:
      results.append(elem.numpy())
    for elem in iter2:
      results.append(elem.numpy())
    self.assertCountEqual(list(range(num_elements)), results)

  @combinations.generate(test_base.eager_only_combinations())
  def testDifferentJobNames(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    ds1 = _make_distributed_dataset(ds, dispatcher, job_name="job_name1")
    ds2 = _make_distributed_dataset(ds, dispatcher, job_name="job_name2")
    self.assertDatasetProduces(ds1, list(range(num_elements)))
    self.assertDatasetProduces(ds2, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedJobNameMultiIteration(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    ds1 = _make_distributed_dataset(ds, dispatcher, job_name="job_name")
    ds2 = _make_distributed_dataset(ds, dispatcher, job_name="job_name")
    # iteration 1
    self.assertDatasetProduces(ds1, list(range(num_elements)))
    self.assertDatasetProduces(ds2, [])
    # iteration 2
    self.assertDatasetProduces(ds2, list(range(num_elements)))
    self.assertDatasetProduces(ds1, [])

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedJobNameRepeat(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 100
    num_repetitions = 3
    ds = dataset_ops.Dataset.range(num_elements)
    ds1 = _make_distributed_dataset(ds, dispatcher, job_name="job_name")
    ds1 = ds1.repeat(num_repetitions)
    ds2 = _make_distributed_dataset(ds, dispatcher, job_name="job_name")
    ds2 = ds2.repeat(num_repetitions)
    results = []
    iter1 = iter(ds1)
    iter2 = iter(ds2)
    for _ in range((num_elements * num_repetitions) // 5):
      results.append(next(iter1).numpy())
    for _ in range((num_elements * num_repetitions) // 5):
      results.append(next(iter2).numpy())
    for elem in iter1:
      results.append(elem.numpy())
    for elem in iter2:
      results.append(elem.numpy())
    self.assertCountEqual(num_repetitions * list(range(num_elements)), results)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(job_name=[None, "test"])))
  def testGcUnusedJob(self, job_name):
    dispatcher = self.start_dispatch_server(
        job_gc_check_interval_ms=50, job_gc_timeout_ms=20)
    worker = self.start_worker_server(dispatcher)  # pylint: disable=unused-variable
    num_elements = 100
    ds = _make_distributed_range_dataset(
        num_elements, dispatcher, job_name=job_name)
    it = iter(ds)
    self.assertEqual(next(it).numpy(), 0)
    self.assertEqual(worker._num_tasks(), 1)
    del it
    while worker._num_tasks() > 0:
      time.sleep(0.1)

  @combinations.generate(test_base.eager_only_combinations())
  def testDontGcUsedJob(self):
    dispatcher = self.start_dispatch_server(
        job_gc_check_interval_ms=50, job_gc_timeout_ms=20)
    worker = self.start_worker_server(dispatcher)  # pylint: disable=unused-variable
    num_elements = 10
    it1 = iter(
        _make_distributed_range_dataset(
            num_elements, dispatcher, job_name="test1"))
    it2 = iter(
        _make_distributed_range_dataset(
            num_elements, dispatcher, job_name="test2"))
    it3 = iter(  # this iterator keeps the task alive. pylint: disable=unused-variable
        _make_distributed_range_dataset(
            num_elements, dispatcher, job_name="test2"))
    self.assertEqual(2, worker._num_tasks())
    del it1
    del it2
    # Check that only the first job is gced. The second job will not be gced
    # because there is still an outstanding iterator for it.
    while worker._num_tasks() > 1:
      time.sleep(0.1)
    self.assertEqual(1, worker._num_tasks())

  @combinations.generate(test_base.eager_only_combinations())
  def testApplyDeterminismOption(self):
    elements = list(range(10))
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable

    def dataset_fn(delay_ms):

      def interleave_fn(x):
        ds = dataset_ops.Dataset.from_tensors(x)
        if math_ops.equal(x, 0):
          ds = ds.apply(testing.sleep(delay_ms * 1000))
        else:
          ds = ds.apply(testing.sleep(0))
        return ds

      ds = dataset_ops.Dataset.from_tensor_slices(elements)
      ds = ds.interleave(interleave_fn, cycle_length=10, num_parallel_calls=10)
      opts = dataset_ops.Options()
      opts.experimental_deterministic = False
      ds = ds.with_options(opts)
      ds = _make_distributed_dataset(ds, dispatcher)
      return ds

    self.checkDeterminism(
        dataset_fn=dataset_fn,
        expect_determinism=False,
        expected_elements=elements)

  def run_stateful(self, external_state_policy):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements).map(
        lambda _: random_ops.random_uniform(()))

    options = dataset_ops.Options()
    options.experimental_external_state_policy = external_state_policy
    ds = ds.with_options(options)

    dispatcher, workers = self.start_cluster(3)  # to avoid gcing workers, pylint: disable=unused-variable
    ds = _make_distributed_dataset(ds, dispatcher)
    next(iter(ds))

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(external_state_policy=[
              distribute_options.ExternalStatePolicy.IGNORE,
              distribute_options.ExternalStatePolicy.WARN
          ])))
  def testStatefulNoError(self, external_state_policy):
    self.run_stateful(external_state_policy)

  @combinations.generate(test_base.eager_only_combinations())
  def testStatefulError(self):
    with self.assertRaises(errors.FailedPreconditionError):
      self.run_stateful(distribute_options.ExternalStatePolicy.FAIL)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpochTensorSlices(self):
    dispatcher, workers = self.start_cluster(2)  # to avoid gcing workers, pylint: disable=unused-variable
    vals = [5, 1, 2, 4]
    ds = dataset_ops.Dataset.from_tensor_slices(vals)
    ds = ds.apply(
        data_service_ops.distribute(
            processing_mode="distributed_epoch", service=dispatcher.target))
    self.assertDatasetProduces(ds, vals, assert_items_equal=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpochRepeat(self):
    dispatcher, workers = self.start_cluster(2)  # to avoid gcing workers, pylint: disable=unused-variable
    num_repeats = 5
    num_elements = 20
    ds = dataset_ops.Dataset.range(num_elements).repeat(num_repeats)
    ds = ds.apply(
        data_service_ops.distribute(
            processing_mode="distributed_epoch", service=dispatcher.target))
    self.assertDatasetProduces(
        ds, num_repeats * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpochShuffleAndRepeat(self):
    dispatcher, workers = self.start_cluster(2)  # to avoid gcing workers, pylint: disable=unused-variable
    num_repeats = 5
    num_elements = 20
    ds = dataset_ops.Dataset.range(num_elements).shuffle(num_elements).repeat(
        num_repeats)
    ds = ds.apply(
        data_service_ops.distribute(
            processing_mode="distributed_epoch", service=dispatcher.target))
    self.assertDatasetProduces(
        ds, num_repeats * list(range(num_elements)), assert_items_equal=True)

  def testDistributeFromInterleave(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    ds = dataset_ops.Dataset.range(2)

    def interleave_fn(_):
      dataset = dataset_ops.Dataset.range(2)
      _make_distributed_dataset(dataset, dispatcher)
      return dataset

    ds = ds.interleave(interleave_fn, cycle_length=2)
    self.assertDatasetProduces(ds, [0, 0, 1, 1])

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpoch(self):
    dispatcher, workers = self.start_cluster(2)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 100
    ds = dataset_ops.Dataset.range(num_elements)
    ds = ds.apply(
        data_service_ops.distribute(
            processing_mode="distributed_epoch", service=dispatcher.target))
    self.assertDatasetProduces(
        ds, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testChangeProcessingModeAfterRestart(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    num_elements = 100
    range_dataset = dataset_ops.Dataset.range(num_elements)
    ds = range_dataset.apply(
        data_service_ops.distribute(
            processing_mode="parallel_epochs",
            service=dispatcher.target,
            job_name="test"))
    iterator = iter(ds)
    for i in range(num_elements // 2):
      self.assertEqual(i, next(iterator).numpy())
    dispatcher = self.restart_dispatcher(dispatcher)
    ds = range_dataset.apply(
        data_service_ops.distribute(
            processing_mode="distributed_epoch",
            service=dispatcher.target,
            job_name="test"))
    with self.assertRaisesOpError("already an existing job with that name "
                                  "using processing mode <parallel_epochs>"):
      next(iter(ds)).numpy()

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeNonStringAddresses(self):
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(ValueError, "service must be a string"):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs", service=1))

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeEmptyAddress(self):
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesWithLiteralMatch(ValueError,
                                           "service must not be empty"):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs", service=""))

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeInvalidProcessingMode(self):
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(ValueError,
                                "invalid is not a valid processing mode"):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="invalid", service="grpc://localhost:5000"))

  @combinations.generate(test_base.eager_only_combinations())
  def testFromDatasetId(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    dataset_id = data_service_ops.register_dataset(dispatcher.target, ds)
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs", dispatcher.target, dataset_id, ds.element_spec)
    self.assertDatasetProduces(from_dataset_id_ds, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testFromDatasetIdMultipleComponents(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    ds = dataset_ops.Dataset.zip({"a": (ds, ds), "b": ds})
    dataset_id = data_service_ops.register_dataset(dispatcher.target, ds)
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs", dispatcher.target, dataset_id, ds.element_spec)
    output = self.getDatasetOutput(from_dataset_id_ds)
    for i in range(num_elements):
      self.assertEqual(i, output[i]["a"][0])
      self.assertEqual(i, output[i]["a"][1])
      self.assertEqual(i, output[i]["b"])

  @combinations.generate(test_base.eager_only_combinations())
  def testFromDatasetIdWrongElementSpec(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    dataset_id = data_service_ops.register_dataset(dispatcher.target, ds)
    wrong_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.variant)
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs", dispatcher.target, dataset_id, wrong_spec)
    with self.assertRaisesRegex(errors.FailedPreconditionError,
                                "Expected a tensor of type variant"):
      self.evaluate(self.getNext(from_dataset_id_ds)())

  @combinations.generate(test_base.eager_only_combinations())
  def testFromDatasetIdNotRegistered(self):
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable

    dataset_id = 0
    element_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.variant)
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs", dispatcher.target, dataset_id, element_spec)
    with self.assertRaisesRegex(errors.NotFoundError, "Dataset id"):
      self.evaluate(self.getNext(from_dataset_id_ds)())

  @combinations.generate(test_base.default_test_combinations())
  def testCancellation(self):
    self.skipTest("b/162521601")
    sleep_microseconds = int(1e6) * 1000

    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    # Create a dataset which produces the first element quickly, and the second
    # element slowly. Fetching the first element triggers prefetching of the
    # second element, which we should be able to cancel.
    slow = dataset_ops.Dataset.range(1)
    slow = slow.apply(testing.sleep(sleep_microseconds))
    ds = dataset_ops.Dataset.range(1).concatenate(slow)
    ds = _make_distributed_dataset(ds, dispatcher)
    ds = ds.prefetch(1)
    get_next = self.getNext(ds, requires_initialization=True)
    self.assertEqual(0, self.evaluate(get_next()))
    # Without properly implemented cancellation, we will hang here while trying
    # to garbage collect the dataset iterator.

  @combinations.generate(test_base.eager_only_combinations())
  def testRegisterEquivalentDatasets(self):
    ds_1 = dataset_ops.Dataset.range(10)
    ds_2 = dataset_ops.Dataset.range(10)
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    id_1 = data_service_ops.register_dataset(dispatcher.target, ds_1)
    id_2 = data_service_ops.register_dataset(dispatcher.target, ds_2)
    self.assertEqual(id_1.numpy(), id_2.numpy())

  @combinations.generate(test_base.eager_only_combinations())
  def testRegisterDifferentDatasets(self):
    ds_1 = dataset_ops.Dataset.range(10)
    ds_2 = dataset_ops.Dataset.range(20)
    dispatcher, workers = self.start_cluster(1)  # to avoid gcing workers, pylint: disable=unused-variable
    id_1 = data_service_ops.register_dataset(dispatcher.target, ds_1)
    id_2 = data_service_ops.register_dataset(dispatcher.target, ds_2)
    self.assertNotEqual(id_1.numpy(), id_2.numpy())

  @combinations.generate(test_base.eager_only_combinations())
  def testTwoLevelDistribute(self):
    cluster_1_size = 3
    dispatcher_1, workers_1 = self.start_cluster(  # to avoid gcing workers, pylint: disable=unused-variable
        cluster_1_size,
        name="cluster_1")
    dispatcher_2, workers_2 = self.start_cluster(1, name="cluster_2")  # to avoid gcing workers, pylint: disable=unused-variable
    num_sizes = 10
    size_repeats = 5
    strings = ["a" * i for i in range(num_sizes)] * size_repeats
    ds = dataset_ops.Dataset.from_tensor_slices(strings)
    ds = ds.shuffle(len(strings))
    ds = _make_distributed_dataset(ds, dispatcher_1)
    # Large enough so that all strings of the same size are windowed together.
    window_size = cluster_1_size * size_repeats
    batch_size = size_repeats

    def key_func(x):
      return math_ops.cast(string_ops.string_length_v2(x), dtypes.int64)

    ds = ds.apply(
        grouping.group_by_window(
            key_func=key_func,
            reduce_func=lambda _, x: x.batch(batch_size),
            window_size=window_size))
    ds = _make_distributed_dataset(ds, dispatcher_2)

    it = iter(ds)
    for _ in range(num_sizes):
      element = next(it).numpy()
      for _ in range(1, cluster_1_size):
        self.assertAllEqual(next(it).numpy(), element)
    self.assertEmpty(list(it))

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations()))
  def testDistributeLargeGraph(self):
    dispatcher, workers = self.start_cluster(  # to avoid gcing workers, pylint: disable=unused-variable
        1,
        work_dir=NO_WORK_DIR,
        fault_tolerant_mode=False)
    # Larger than default OSS grpc message size limit of 4MB.
    tensor = array_ops.ones((2, 1000, 1000), dtype=dtypes.float32)
    ds = dataset_ops.Dataset.from_tensors(tensor)
    ds = _make_distributed_dataset(ds, dispatcher)
    self.assertDatasetProduces(ds, [tensor])

  @combinations.generate(
      combinations.times(
          test_base.eager_only_combinations(),
          combinations.combine(work_dir=[TMP_WORK_DIR, NO_WORK_DIR])))
  def testDistributeLargeGraphThenRegisterWorker(self, work_dir):
    dispatcher = self.start_dispatch_server(
        work_dir=work_dir, fault_tolerant_mode=False)
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(
            dispatcher_address=_address_from_target(dispatcher.target), port=0),
        start=False)
    # Larger than default OSS grpc message size limit of 4MB.
    tensor = array_ops.ones((2, 1000, 1000), dtype=dtypes.float32)
    ds = dataset_ops.Dataset.from_tensors(tensor)
    ds = _make_distributed_dataset(ds, dispatcher)
    it = iter(ds)
    worker.start()
    self.assertAllEqual(next(it), tensor)


if __name__ == "__main__":
  test.main()
