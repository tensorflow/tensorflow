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

import time

from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.service import server_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test

PROTOCOL = "grpc"


def _make_distributed_dataset(dataset, service):
  """Creates a distributed dataset with a short task refresh interval."""
  return dataset.apply(
      data_service_ops._distribute(service, task_refresh_interval_hint_ms=20))


class DataServiceOpsTest(test_base.DatasetTestBase, parameterized.TestCase):

  def create_cluster(self, num_workers):
    """Creates a cluster of tf.data service servers.

    Args:
      num_workers: The number of workers in the cluster.

    Returns:
      A target for connecting to the service, e.g.
      "grpc+local://localhost:2000".
    """
    self._master = server_lib.MasterServer(PROTOCOL)
    master_address = self._master.target[len(PROTOCOL + "://"):]

    self._servers = []
    for _ in range(num_workers):
      self._servers.append(
          server_lib.WorkerServer(PROTOCOL, master_address=master_address))

    return self._master.target

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleEpochs(self):
    service = self.create_cluster(1)
    ds = dataset_ops.Dataset.range(3)
    ds = _make_distributed_dataset(ds, service)
    for _ in range(10):
      token = data_service_ops.create_job(ds, processing_mode="parallel_epochs")
      it = data_service_ops.create_iterator(ds, token)
      self.assertEqual(list(range(3)), [t.numpy() for t in it])

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeBasic(self):
    num_elements = 10
    service = self.create_cluster(1)
    ds = dataset_ops.Dataset.range(num_elements)
    ds = _make_distributed_dataset(ds, service)
    token = data_service_ops.create_job(ds, processing_mode="parallel_epochs")
    it = data_service_ops.create_iterator(ds, token)
    results = [t.numpy() for t in it]
    self.assertEqual(list(range(num_elements)), results)

  @combinations.generate(test_base.eager_only_combinations())
  def testConcurrentEpoch(self):
    num_elements = 10
    num_datasets = 3
    service = self.create_cluster(1)
    iterators = []
    results = []
    for _ in range(num_datasets):
      ds = dataset_ops.Dataset.range(num_elements)
      ds = _make_distributed_dataset(ds, service)
      token = data_service_ops.create_job(ds, processing_mode="parallel_epochs")
      it = data_service_ops.create_iterator(ds, token)
      iterators.append(it)
      results.append([])

    for _ in range(num_elements):
      for dataset_ind in range(num_datasets):
        result = next(iterators[dataset_ind]).numpy()
        results[dataset_ind].append(result)
    for result in results:
      self.assertEqual(list(range(num_elements)), result)

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedEpoch(self):
    num_elements = 10
    num_iterators = 3
    service = self.create_cluster(1)
    ds = dataset_ops.Dataset.range(num_elements)
    ds = _make_distributed_dataset(ds, service)
    result = []
    iterators = []
    token = data_service_ops.create_job(ds, processing_mode="parallel_epochs")
    for _ in range(num_iterators):
      iterators.append(data_service_ops.create_iterator(ds, token))

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
    num_elements = 10
    service = self.create_cluster(num_workers)
    ds = dataset_ops.Dataset.range(num_elements)
    ds = _make_distributed_dataset(ds, service)
    token = data_service_ops.create_job(ds, processing_mode="parallel_epochs")
    iterator = data_service_ops.create_iterator(ds, token)
    results = [elem.numpy() for elem in iterator]
    self.assertCountEqual(num_workers * list(range(num_elements)), results)

  @combinations.generate(test_base.eager_only_combinations())
  def testAddWorkerMidJob(self):
    self._master = server_lib.MasterServer(PROTOCOL)
    master_address = self._master.target[len(PROTOCOL + "://"):]
    self._worker = server_lib.WorkerServer(
        PROTOCOL, master_address=master_address)
    num_elements = 100
    ds = dataset_ops.Dataset.range(num_elements)
    ds = _make_distributed_dataset(ds, self._master.target)
    token = data_service_ops.create_job(ds, processing_mode="parallel_epochs")
    iterator = data_service_ops.create_iterator(ds, token)
    results = []
    # Read halfway through the dataset.
    for _ in range(num_elements // 2):
      results.append(next(iterator).numpy())

    self._new_worker = server_lib.WorkerServer(
        PROTOCOL, master_address=master_address)

    # Wait for the new worker to register with the master.
    while self._master.num_tasks() < 2:
      time.sleep(10 / 1000)  # 10ms

    for elem in iterator:
      results.append(elem.numpy())

    self.assertCountEqual(2 * list(range(num_elements)), results)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(use_same_port=[True, False])))
  def testRestartWorker(self, use_same_port):
    self._master = server_lib.MasterServer(PROTOCOL)
    master_address = self._master.target[len(PROTOCOL + "://"):]
    self._worker = server_lib.WorkerServer(
        PROTOCOL, master_address=master_address)
    num_elements = 100
    ds = dataset_ops.Dataset.range(num_elements)
    ds = _make_distributed_dataset(ds, self._master.target)
    token = data_service_ops.create_job(ds, processing_mode="parallel_epochs")
    iterator = data_service_ops.create_iterator(ds, token)
    # Read halfway through the dataset.
    midpoint = num_elements // 2
    for i in range(midpoint):
      self.assertEqual(i, next(iterator).numpy())

    # Stop the original worker and start a new one.
    port = 0
    if use_same_port:
      worker_address = self._worker.target[len(PROTOCOL + "://"):]
      port = int(worker_address.split(":")[1])
    self._worker.stop()
    self._new_worker = server_lib.WorkerServer(
        PROTOCOL, master_address=master_address, port=port)

    # The dataset starts over now that we read from the new worker.
    for i in range(num_elements):
      val = next(iterator).numpy()
      if val == midpoint and i != midpoint:
        # There may have been one last element prefetched from the first worker
        # before it was stopped.
        val = next(iterator).numpy()
      self.assertEqual(i, val)

  @combinations.generate(test_base.eager_only_combinations())
  def testInsideFunction(self):
    num_workers = 3
    num_elements = 10
    service = self.create_cluster(num_workers)

    @def_function.function
    def f():
      ds = dataset_ops.Dataset.range(num_elements)
      ds = _make_distributed_dataset(ds, service)
      token = data_service_ops.create_job(ds, processing_mode="parallel_epochs")
      it = data_service_ops.create_iterator(ds, token)
      result = tensor_array_ops.TensorArray(
          dtypes.int64, size=num_workers * num_elements, dynamic_size=True)
      i = 0
      for elem in it:
        result = result.write(i, elem)
        i += 1
      return result.stack()

    result = list(f().numpy())
    self.assertCountEqual(num_workers * list(range(num_elements)), result)

  def run_stateful(self, external_state_policy):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements).map(
        lambda _: random_ops.random_uniform(()))

    options = dataset_ops.Options()
    options.experimental_external_state_policy = external_state_policy
    ds = ds.with_options(options)

    service = self.create_cluster(3)
    ds = _make_distributed_dataset(ds, service)
    token = data_service_ops.create_job(ds, processing_mode="parallel_epochs")
    iterator = data_service_ops.create_iterator(ds, token)
    next(iterator)

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
  def testNoDistributeCalls(self):
    ds = dataset_ops.Dataset.range(1)
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "Dataset does not contain any distribute() transformations"):
      data_service_ops.create_job(ds, processing_mode="parallel_epochs")

  @combinations.generate(test_base.eager_only_combinations())
  def testMultipleDistributeCalls(self):
    service = self.create_cluster(1)
    ds1 = dataset_ops.Dataset.range(1)
    ds1 = _make_distributed_dataset(ds1, service)
    ds2 = dataset_ops.Dataset.range(1)
    ds2 = _make_distributed_dataset(ds2, service)
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    with self.assertRaisesWithLiteralMatch(
        ValueError, "Datasets containing multiple calls to .distribute(...) "
        "are not supported"):
      data_service_ops.create_job(ds, processing_mode="parallel_epochs")

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeFromInterleave(self):
    service = self.create_cluster(1)
    ds = dataset_ops.Dataset.range(2)

    def interleave_fn(_):
      ds = dataset_ops.Dataset.range(2)
      _make_distributed_dataset(ds, service)
      return ds

    with self.assertRaisesRegex(
        errors.InvalidArgumentError, r"The `.distribute\(...\)` dataset "
        "transformation is not supported within tf.data functions"):
      ds = ds.interleave(interleave_fn, cycle_length=2)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeNonStringAddresses(self):
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(ValueError, "service must be a string"):
      ds = ds.apply(data_service_ops.distribute(service=1))

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeEmptyAddress(self):
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesWithLiteralMatch(ValueError,
                                           "service must not be empty"):
      ds = ds.apply(data_service_ops.distribute(service=""))


if __name__ == "__main__":
  test.main()
