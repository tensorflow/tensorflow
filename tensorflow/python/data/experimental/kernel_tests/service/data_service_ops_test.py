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
import time

from absl.testing import parameterized

from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

TMP_WORK_DIR = data_service_test_base.TMP_WORK_DIR
NO_WORK_DIR = data_service_test_base.NO_WORK_DIR


class DataServiceOpsTest(data_service_test_base.TestBase,
                         parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         data_service_test_base.all_cluster_configurations()))
  def testDistributeBasic(self, work_dir, fault_tolerant_mode):
    cluster = data_service_test_base.TestCluster(
        num_workers=1,
        work_dir=work_dir,
        fault_tolerant_mode=fault_tolerant_mode)
    num_elements = 10
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(compression=[None, "AUTO"])))
  def testDistributeCompression(self, compression):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, compression=compression)
    self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeInvalidCompression(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    with self.assertRaisesRegex(ValueError, "Invalid `compression` argument"):
      self.make_distributed_range_dataset(10, cluster, compression="foo")

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeSparse(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    element = sparse_tensor.SparseTensor(
        indices=[[0]],
        values=constant_op.constant([0], dtype=dtypes.int32),
        dense_shape=[1])
    ds = dataset_ops.Dataset.from_tensors(element)
    ds = self.make_distributed_dataset(ds, cluster)
    results = [sparse_ops.sparse_tensor_to_dense(elem) for elem in ds]
    self.assertAllEqual(results, [[0]])

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeRagged(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    ds = dataset_ops.Dataset.from_tensor_slices([1, 5, 3, 2, 8])
    ds = ds.map(math_ops.range)
    ds = ds.apply(batching.dense_to_ragged_batch(2))
    ds = self.make_distributed_dataset(ds, cluster)
    results = [elem.to_tensor() for elem in ds]
    self.assertAllEqual(results[0], [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]])
    self.assertAllEqual(results[1], [[0, 1, 2], [0, 1, 0]])
    self.assertAllEqual(results[2], [[0, 1, 2, 3, 4, 5, 6, 7]])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              init_source=["textfile", "keyvaluetensor", "dataset"])))
  def testDistributeLookupTable(self, init_source):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    initializer = self.lookupTableInitializer(init_source, [10, 11])
    table = lookup_ops.StaticHashTable(initializer, -1)
    ds = dataset_ops.Dataset.range(3)
    ds = ds.map(table.lookup)
    ds = self.make_distributed_dataset(ds, cluster)
    self.evaluate(lookup_ops.tables_initializer())
    self.assertDatasetProduces(ds, [10, 11, -1], requires_initialization=True)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(value_rank=[0, 1])))
  def testDistributeMutableHashTable(self, value_rank):

    def value(v):
      for _ in range(value_rank):
        v = [v, v]
      return v

    v1 = value(10)
    v2 = value(11)
    default_value = value(-1)

    cluster = data_service_test_base.TestCluster(num_workers=1)
    table = lookup_ops.MutableHashTable(dtypes.int64, dtypes.int64,
                                        default_value)
    self.evaluate(table.insert([0, 1], [v1, v2]))
    ds = dataset_ops.Dataset.range(3)
    ds = ds.map(table.lookup)
    ds = self.make_distributed_dataset(ds, cluster)
    self.assertDatasetProduces(
        ds, [v1, v2, default_value], requires_initialization=True)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(shuffle_seed=[None, 10])))
  def testShuffleOrder(self, shuffle_seed):
    random_seed.set_random_seed(None)
    num_elements = 100
    cluster = data_service_test_base.TestCluster(num_workers=2)
    ds = dataset_ops.Dataset.range(num_elements)
    ds = ds.shuffle(num_elements, seed=shuffle_seed)
    ds = self.make_distributed_dataset(ds, cluster)
    output = self.getDatasetOutput(ds)

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
    if shuffle_seed is None:
      self.assertNotEqual(first_order, second_order)
    else:
      self.assertEqual(first_order, second_order)

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleEpochs(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 3
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    for _ in range(10):
      self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testRepeatedDataset(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 10
    num_repetitions = 5
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    ds = ds.repeat(num_repetitions)
    self.assertDatasetProduces(
        ds, expected_output=num_repetitions * list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testConcurrentEpoch(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 10
    num_datasets = 3
    get_nexts = []
    results = []
    for _ in range(num_datasets):
      ds = self.make_distributed_range_dataset(num_elements, cluster)
      get_nexts.append(self.getNext(ds))
      results.append([])

    for _ in range(num_elements):
      for dataset_ind in range(num_datasets):
        result = self.evaluate(get_nexts[dataset_ind]())
        results[dataset_ind].append(result)
    for result in results:
      self.assertEqual(list(range(num_elements)), result)

  @combinations.generate(test_base.default_test_combinations())
  def testMultiWorker(self):
    num_workers = 3
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    self.assertDatasetProduces(
        ds, num_workers * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testMaxOutstandingRequests(self):
    num_workers = 3
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, max_outstanding_requests=1)
    self.assertDatasetProduces(
        ds, num_workers * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testInsideFunction(self):
    num_workers = 3
    cluster = data_service_test_base.TestCluster(num_workers=num_workers)
    num_elements = 10

    @def_function.function
    def f():
      ds = self.make_distributed_range_dataset(num_elements, cluster)
      result = tensor_array_ops.TensorArray(
          dtypes.int64, size=num_workers * num_elements, dynamic_size=True)
      i = 0
      for elem in ds:
        result = result.write(i, elem)
        i += 1
      return result.stack()

    result = list(f().numpy())
    self.assertCountEqual(num_workers * list(range(num_elements)), result)

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyJobNameDistribute(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    with self.assertRaisesRegex(ValueError, "`job_name` must not be empty"):
      dataset_ops.Dataset.range(10).apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs",
              service=cluster.dispatcher.target,
              job_name=""))

  @combinations.generate(test_base.default_test_combinations())
  def testEmptyJobNameFromDatasetId(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset_ops.Dataset.range(10))
    with self.assertRaisesRegex(ValueError, "`job_name` must not be empty"):
      data_service_ops.from_dataset_id(
          dataset_id=dataset_id,
          processing_mode="parallel_epochs",
          service=cluster.dispatcher.target,
          job_name="")

  @combinations.generate(test_base.default_test_combinations())
  def testExplicitProtocolFromDatasetId(self):
    cluster = data_service_test_base.TestCluster(
        num_workers=1, data_transfer_protocol="grpc")
    range_ds = dataset_ops.Dataset.range(10)
    dataset_id = data_service_ops.register_dataset(cluster.dispatcher.target,
                                                   range_ds)
    ds = data_service_ops.from_dataset_id(
        dataset_id=dataset_id,
        processing_mode="parallel_epochs",
        element_spec=range_ds.element_spec,
        service=cluster.dispatcher.target,
        data_transfer_protocol="grpc")
    self.assertDatasetProduces(ds, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testNonStringJobNameDistribute(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    with self.assertRaisesRegex(ValueError, "`job_name` must be a string"):
      dataset_ops.Dataset.range(10).apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs",
              service=cluster.dispatcher.target,
              job_name=constant_op.constant("foo")))

  @combinations.generate(test_base.default_test_combinations())
  def testNonStringJobNameFromDatasetId(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset_id = data_service_ops.register_dataset(
        cluster.dispatcher.target, dataset_ops.Dataset.range(10))
    with self.assertRaisesRegex(ValueError, "`job_name` must be a string"):
      data_service_ops.from_dataset_id(
          dataset_id=dataset_id,
          processing_mode="parallel_epochs",
          service=cluster.dispatcher.target,
          job_name=constant_op.constant("foo"))

  @combinations.generate(test_base.default_test_combinations())
  def testSharedJobName(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 1000

    def make_ds():
      return dataset_ops.Dataset.range(num_elements).shuffle(num_elements)

    ds1 = self.make_distributed_dataset(make_ds(), cluster, job_name="job_name")
    ds2 = self.make_distributed_dataset(make_ds(), cluster, job_name="job_name")
    get_next_1 = self.getNext(ds1)
    get_next_2 = self.getNext(ds2)
    results = []
    for _ in range(num_elements // 5):
      results.append(self.evaluate(get_next_1()))
      results.append(self.evaluate(get_next_2()))
    results += self.getIteratorOutput(get_next_1)
    results += self.getIteratorOutput(get_next_2)
    self.assertCountEqual(list(range(num_elements)), results)

  @combinations.generate(test_base.default_test_combinations())
  def testDifferentJobNames(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 10
    ds1 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name1")
    ds2 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name2")
    self.assertDatasetProduces(ds1, list(range(num_elements)))
    self.assertDatasetProduces(ds2, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedJobNameMultiIteration(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 10
    ds1 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name")
    ds2 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name")
    # iteration 1
    self.assertDatasetProduces(ds1, list(range(num_elements)))
    self.assertDatasetProduces(ds2, [])
    # iteration 2
    self.assertDatasetProduces(ds2, list(range(num_elements)))
    self.assertDatasetProduces(ds1, [])

  @combinations.generate(test_base.default_test_combinations())
  def testSharedJobNameRepeat(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    num_repetitions = 3
    ds1 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name")
    ds1 = ds1.repeat(num_repetitions)
    ds2 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name")
    ds2 = ds2.repeat(num_repetitions)
    results = []
    get_next_1 = self.getNext(ds1)
    get_next_2 = self.getNext(ds2)
    for _ in range((num_elements * num_repetitions) // 5):
      results.append(self.evaluate(get_next_1()))
    for _ in range((num_elements * num_repetitions) // 5):
      results.append(self.evaluate(get_next_2()))
    results += self.getIteratorOutput(get_next_1)
    results += self.getIteratorOutput(get_next_2)
    self.assertCountEqual(num_repetitions * list(range(num_elements)), results)

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedJobNameMultipleEpochs(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = self.make_distributed_range_dataset(
        10, cluster, job_name="job_name")

    num_epochs = 5
    for _ in range(num_epochs):
      get_next = self.getNext(dataset)
      self.assertEqual(self.getIteratorOutput(get_next), list(range(10)))

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         combinations.combine(job_name=[None, "test"])))
  def testGcUnusedJob(self, job_name):
    cluster = data_service_test_base.TestCluster(
        num_workers=1, job_gc_check_interval_ms=50, job_gc_timeout_ms=20)
    num_elements = 100
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, job_name=job_name)
    it = iter(ds)
    self.assertEqual(next(it).numpy(), 0)
    self.assertEqual(cluster.workers[0].num_tasks(), 1)
    del it
    while cluster.workers[0].num_tasks() > 0:
      time.sleep(0.1)

  @combinations.generate(test_base.eager_only_combinations())
  def testDontGcUsedJob(self):
    cluster = data_service_test_base.TestCluster(
        num_workers=1, job_gc_check_interval_ms=50, job_gc_timeout_ms=20)
    num_elements = 10
    it1 = iter(
        self.make_distributed_range_dataset(
            num_elements, cluster, job_name="test1"))
    it2 = iter(
        self.make_distributed_range_dataset(
            num_elements, cluster, job_name="test2"))
    it3 = iter(  # this iterator keeps the task alive. pylint: disable=unused-variable
        self.make_distributed_range_dataset(
            num_elements, cluster, job_name="test2"))
    self.assertEqual(cluster.workers[0].num_tasks(), 2)
    del it1
    del it2
    # Check that only the first job is gced. The second job will not be gced
    # because there is still an outstanding iterator for it.
    while cluster.workers[0].num_tasks() > 1:
      time.sleep(0.1)
    self.assertEqual(cluster.workers[0].num_tasks(), 1)

  @combinations.generate(test_base.eager_only_combinations())
  def testGcAndRecreate(self):
    cluster = data_service_test_base.TestCluster(
        num_workers=3, job_gc_check_interval_ms=50, job_gc_timeout_ms=20)
    num_elements = 1000
    # Repeatedly create and garbage-collect the same job.
    for _ in range(3):
      ds = self.make_distributed_range_dataset(
          num_elements, cluster, job_name="test")
      it = iter(ds)
      for _ in range(50):
        next(it)
      del it
      # Wait for the task to be garbage-collected on all workers.
      while cluster.num_tasks_on_workers() > 0:
        time.sleep(0.1)

  @combinations.generate(test_base.eager_only_combinations())
  def testGcClient(self):
    dispatcher = server_lib.DispatchServer(
        service_config_pb2.DispatcherConfig(
            protocol="grpc",
            job_gc_check_interval_ms=50,
            job_gc_timeout_ms=20,
            client_timeout_ms=50))
    dispatcher_address = dispatcher.target.split("://")[1]
    _ = server_lib.WorkerServer(
        server_lib.WorkerConfig(
            dispatcher_address=dispatcher_address, heartbeat_interval_ms=100))

    num_elements = 1000
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.apply(
        data_service_ops._distribute(
            processing_mode=data_service_ops.ShardingPolicy.OFF,
            service=dispatcher.target,
            task_refresh_interval_hint_ms=10000))
    get_next = self.getNext(dataset)

    # The client does not heartbeat in 10 seconds. It will be garbage-collected.
    with self.assertRaisesRegex(errors.NotFoundError, "Unknown job client id"):
      self.evaluate(get_next())
      time.sleep(3)
      self.getIteratorOutput(get_next)

  @combinations.generate(test_base.eager_only_combinations())
  def testKeepClientAliveBeforeReading(self):
    dispatcher = server_lib.DispatchServer(
        service_config_pb2.DispatcherConfig(
            protocol="grpc",
            job_gc_check_interval_ms=50,
            job_gc_timeout_ms=20,
            client_timeout_ms=1000))
    dispatcher_address = dispatcher.target.split("://")[1]
    _ = server_lib.WorkerServer(
        server_lib.WorkerConfig(
            dispatcher_address=dispatcher_address, heartbeat_interval_ms=100))

    num_elements = 1000
    dataset = dataset_ops.Dataset.range(num_elements)
    dataset = dataset.apply(
        data_service_ops._distribute(
            processing_mode=data_service_ops.ShardingPolicy.OFF,
            service=dispatcher.target,
            task_refresh_interval_hint_ms=100))
    get_next = self.getNext(dataset)

    # The client regularly heartbeats in 100 milliseconds. It should not be
    # garbage-collected even if it does not start reading in 3 seconds.
    time.sleep(3)
    self.assertEqual(
        self.getIteratorOutput(get_next), list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testApplyDeterminismOption(self):
    elements = list(range(10))
    cluster = data_service_test_base.TestCluster(num_workers=1)

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
      opts = options_lib.Options()
      opts.deterministic = False
      ds = ds.with_options(opts)
      ds = self.make_distributed_dataset(ds, cluster)
      return ds

    self.checkDeterminism(
        dataset_fn=dataset_fn,
        expect_determinism=False,
        expected_elements=elements)

  def run_stateful(self, external_state_policy):
    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements).map(
        lambda _: random_ops.random_uniform(()))

    options = options_lib.Options()
    options.experimental_external_state_policy = external_state_policy
    ds = ds.with_options(options)

    cluster = data_service_test_base.TestCluster(num_workers=3)
    ds = self.make_distributed_dataset(ds, cluster)
    self.getDatasetOutput(ds)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(external_state_policy=[
              options_lib.ExternalStatePolicy.IGNORE,
              options_lib.ExternalStatePolicy.WARN
          ])))
  def testStatefulNoError(self, external_state_policy):
    self.run_stateful(external_state_policy)

  @combinations.generate(test_base.default_test_combinations())
  def testStatefulError(self):
    with self.assertRaises(errors.FailedPreconditionError):
      self.run_stateful(options_lib.ExternalStatePolicy.FAIL)

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeFromInterleave(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    ds = dataset_ops.Dataset.range(2)

    def interleave_fn(x):
      dataset = dataset_ops.Dataset.range(10 * x, 10 * x + 2)
      dataset = self.make_distributed_dataset(dataset, cluster)
      return dataset

    ds = ds.interleave(interleave_fn, cycle_length=2)
    self.assertDatasetProduces(ds, [0, 10, 1, 11])

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeNonStringAddresses(self):
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(ValueError, "`service` must be a string"):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs", service=1))

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeEmptyAddress(self):
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesWithLiteralMatch(ValueError,
                                           "`service` must not be empty"):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs", service=""))

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeExplicitProtocol(self):
    cluster = data_service_test_base.TestCluster(
        num_workers=1, data_transfer_protocol="grpc")
    ds = dataset_ops.Dataset.range(10)
    ds = ds.apply(
        data_service_ops.distribute(
            processing_mode="parallel_epochs",
            service="grpc://" + cluster.dispatcher_address()))
    self.assertDatasetProduces(ds, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testDistributeInvalidProtocol(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        errors.NotFoundError,
        "No credentials factory has been registered for protocol grp"):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="parallel_epochs",
              service="grp://" + cluster.dispatcher_address()))
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeInvalidProcessingMode(self):
    ds = dataset_ops.Dataset.range(10)
    with self.assertRaisesRegex(
        ValueError,
        "should be a `tf.data.experimental.service.ShardingPolicy`, "
        "`\"parallel_epochs\"`, or "
        "`\"distributed_epoch\"`. Got 'invalid'."):
      ds = ds.apply(
          data_service_ops.distribute(
              processing_mode="invalid", service="grpc://localhost:5000"))

  @combinations.generate(test_base.default_test_combinations())
  def testZipDifferentProcessingModesDatasets(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    ds1 = dataset_ops.Dataset.range(num_elements)
    ds1 = self.make_distributed_dataset(
        ds1, cluster, processing_mode="distributed_epoch")
    ds2 = dataset_ops.Dataset.range(num_elements)
    ds2 = self.make_distributed_dataset(
        ds2, cluster, processing_mode="parallel_epochs")
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    self.assertDatasetProduces(
        ds,
        list(zip(range(num_elements), range(num_elements))),
        assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testZipDifferentProcessingModesDatasetsSharedJobName(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    num_elements = 100
    ds1 = dataset_ops.Dataset.range(num_elements)
    ds1 = self.make_distributed_dataset(
        ds1, cluster, processing_mode="distributed_epoch", job_name="job_name")
    ds2 = dataset_ops.Dataset.range(num_elements)
    ds2 = self.make_distributed_dataset(
        ds2, cluster, processing_mode="parallel_epochs", job_name="job_name")
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "but found an existing job with diff"):
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetId(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    dataset_id = self.register_dataset(cluster.dispatcher_address(), ds)
    from_dataset_id_ds = self.from_dataset_id("parallel_epochs", cluster,
                                              dataset_id, ds.element_spec)
    self.assertDatasetProduces(from_dataset_id_ds, list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetIdSharedJobs(self):
    cluster = data_service_test_base.TestCluster(num_workers=2)

    datasets = [
        dataset_ops.Dataset.range(20, output_type=dtypes.int32),
        dataset_ops.Dataset.from_tensor_slices(list(range(20, 40)))
    ]
    dataset_ids = []

    for ds in datasets:
      dataset_id = self.register_dataset(cluster.dispatcher_address(), ds)
      dataset_ids.append(dataset_id)

    # Read from both jobs in parallel, with 2 consumers for each job.
    data_service_datasets = []
    for _ in range(2):
      for dataset, dataset_id in zip(datasets, dataset_ids):
        ds = self.from_dataset_id(
            "distributed_epoch",
            cluster,
            dataset_id,
            dataset.element_spec,
            job_name="shared_job")
        data_service_datasets.append(ds)
    ds = dataset_ops.Dataset.from_tensor_slices(data_service_datasets)
    ds = ds.interleave(lambda x: x, cycle_length=len(data_service_datasets))

    self.assertDatasetProduces(ds, list(range(40)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testRegisteringDatasetAsTfFunction(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    register_func = def_function.function(self.register_dataset)
    dataset_id = register_func(
        (constant_op.constant("grpc"),
         constant_op.constant(cluster.dispatcher_address())), ds)
    from_dataset_id_ds = self.from_dataset_id("parallel_epochs", cluster,
                                              dataset_id, ds.element_spec)
    self.assertDatasetProduces(from_dataset_id_ds, list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetIdMultipleComponents(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    ds = dataset_ops.Dataset.zip({"a": (ds, ds), "b": ds})
    dataset_id = self.register_dataset(cluster.dispatcher_address(), ds)
    from_dataset_id_ds = self.from_dataset_id("parallel_epochs", cluster,
                                              dataset_id, ds.element_spec)
    output = self.getDatasetOutput(from_dataset_id_ds)
    for i in range(num_elements):
      self.assertEqual(i, output[i]["a"][0])
      self.assertEqual(i, output[i]["a"][1])
      self.assertEqual(i, output[i]["b"])

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetIdWrongElementSpec(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    dataset_id = self.register_dataset(cluster.dispatcher_address(), ds)
    wrong_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.variant)
    from_dataset_id_ds = self.from_dataset_id("parallel_epochs", cluster,
                                              dataset_id, wrong_spec)

    if data_service_test_base.TRANSFER_PROTOCOL.value:
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Data type mismatch at component 0"):
        self.evaluate(self.getNext(from_dataset_id_ds)())
    else:
      with self.assertRaisesRegex(errors.FailedPreconditionError,
                                  "Expected a tensor of type variant"):
        self.evaluate(self.getNext(from_dataset_id_ds)())

  @combinations.generate(test_base.default_test_combinations())
  def testFromDatasetIdNotRegistered(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)

    dataset_id = 0
    element_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.variant)
    with self.assertRaisesRegex(errors.NotFoundError, "Dataset id 0 not found"):
      from_dataset_id_ds = self.from_dataset_id("parallel_epochs", cluster,
                                                dataset_id, element_spec)
      self.evaluate(self.getNext(from_dataset_id_ds)())

  @combinations.generate(test_base.default_test_combinations())
  def testCancellation(self):
    self.skipTest("b/162521601")
    sleep_microseconds = int(1e6) * 1000

    cluster = data_service_test_base.TestCluster(num_workers=1)
    # Create a dataset which produces the first element quickly, and the second
    # element slowly. Fetching the first element triggers prefetching of the
    # second element, which we should be able to cancel.
    slow = dataset_ops.Dataset.range(1)
    slow = slow.apply(testing.sleep(sleep_microseconds))
    ds = dataset_ops.Dataset.range(1).concatenate(slow)
    ds = self.make_distributed_dataset(ds, cluster)
    ds = ds.prefetch(1)
    get_next = self.getNext(ds)
    self.assertEqual(0, self.evaluate(get_next()))
    # Without properly implemented cancellation, we will hang here while trying
    # to garbage collect the dataset iterator.

  @combinations.generate(test_base.default_test_combinations())
  def testRegisterEquivalentDatasets(self):
    ds_1 = dataset_ops.Dataset.range(10)
    ds_2 = dataset_ops.Dataset.range(10)
    cluster = data_service_test_base.TestCluster(num_workers=1)
    id_1 = self.register_dataset(cluster.dispatcher_address(), ds_1)
    id_2 = self.register_dataset(cluster.dispatcher_address(), ds_2)
    self.assertEqual(self.evaluate(id_1), self.evaluate(id_2))

  @combinations.generate(test_base.default_test_combinations())
  def testRegisterDifferentDatasets(self):
    ds_1 = dataset_ops.Dataset.range(10)
    ds_2 = dataset_ops.Dataset.range(20)
    cluster = data_service_test_base.TestCluster(num_workers=1)
    id_1 = self.register_dataset(cluster.dispatcher_address(), ds_1)
    id_2 = self.register_dataset(cluster.dispatcher_address(), ds_2)
    self.assertNotEqual(self.evaluate(id_1), self.evaluate(id_2))

  @combinations.generate(test_base.default_test_combinations())
  def testDoubleDistribute(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    ds = self.make_distributed_range_dataset(num_elements=10, cluster=cluster)
    ds = self.make_distributed_dataset(dataset=ds, cluster=cluster)
    self.assertDatasetProduces(ds, list(range(10)))

  @combinations.generate(test_base.default_test_combinations())
  def testTwoLevelDistribute(self):
    cluster_1_size = 3
    cluster_1 = data_service_test_base.TestCluster(num_workers=cluster_1_size)
    cluster_2 = data_service_test_base.TestCluster(num_workers=1)
    num_sizes = 10
    size_repeats = 5
    strings = ["a" * i for i in range(num_sizes)] * size_repeats
    ds = dataset_ops.Dataset.from_tensor_slices(strings)
    ds = ds.shuffle(len(strings))
    ds = self.make_distributed_dataset(ds, cluster_1)
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
    ds = self.make_distributed_dataset(ds, cluster_2)

    get_next = self.getNext(ds)
    for _ in range(num_sizes):
      element = self.evaluate(get_next())
      for _ in range(1, cluster_1_size):
        self.assertAllEqual(self.evaluate(get_next()), element)
    self.assertEmpty(self.getIteratorOutput(get_next))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testDistributeLargeGraph(self):
    cluster = data_service_test_base.TestCluster(
        num_workers=1, work_dir=NO_WORK_DIR, fault_tolerant_mode=False)
    # Larger than default OSS grpc message size limit of 4MB.
    tensor = array_ops.ones((2, 1000, 1000), dtype=dtypes.float32)
    ds = dataset_ops.Dataset.from_tensors(tensor)
    ds = self.make_distributed_dataset(ds, cluster)
    self.assertDatasetProduces(ds, [tensor])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testBatchDropsAllElements(self):
    cluster = data_service_test_base.TestCluster(
        num_workers=2, fault_tolerant_mode=False)
    dataset = dataset_ops.Dataset.range(10).batch(1000, drop_remainder=True)
    dataset = self.make_distributed_dataset(
        dataset, cluster, processing_mode=data_service_ops.ShardingPolicy.OFF)
    self.assertDatasetProduces(dataset, [])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testBatchDoesNotDropRemainder(self):
    num_workers = 2
    cluster = data_service_test_base.TestCluster(
        num_workers=num_workers, fault_tolerant_mode=False)
    dataset = dataset_ops.Dataset.range(10).batch(1000, drop_remainder=False)
    dataset = self.make_distributed_dataset(
        dataset, cluster, processing_mode=data_service_ops.ShardingPolicy.OFF)
    self.assertDatasetProduces(dataset, [list(range(10))] * num_workers)

  @combinations.generate(
      combinations.times(test_base.graph_only_combinations(),
                         combinations.combine(use_resource=False)) +
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(use_resource=True)))
  def testVariables(self, use_resource):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    if not use_resource:
      with variable_scope.variable_scope("foo", use_resource=False):
        v = variables.VariableV1(10, dtype=dtypes.int64)
    else:
      v = variables.Variable(10, dtype=dtypes.int64)

    ds = dataset_ops.Dataset.range(3)
    ds = ds.map(lambda x: x + v)
    ds = self.make_distributed_dataset(ds, cluster)
    self.evaluate(v.initializer)
    self.assertDatasetProduces(
        ds, list(range(10, 13)), requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testNoShardingPolicy(self):
    cluster = data_service_test_base.TestCluster(num_workers=1)
    dataset = dataset_ops.Dataset.range(20)
    dataset = self.make_distributed_dataset(
        dataset,
        cluster=cluster,
        processing_mode=data_service_ops.ShardingPolicy.OFF)
    self.assertDatasetProduces(dataset, list(range(20)))


if __name__ == "__main__":
  test.main()
