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

from tensorflow.python.data.experimental.kernel_tests import data_service_test_base
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.ops import distribute_options
from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.data.experimental.ops import testing
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

TMP_WORK_DIR = data_service_test_base.TMP_WORK_DIR
NO_WORK_DIR = data_service_test_base.NO_WORK_DIR


class DataServiceOpsTest(data_service_test_base.TestBase,
                         parameterized.TestCase):

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations(),
                         data_service_test_base.all_cluster_configurations()))
  def testDistributeBasic(self, work_dir, fault_tolerant_mode):
    cluster = self.create_cluster(
        num_workers=1,
        work_dir=work_dir,
        fault_tolerant_mode=fault_tolerant_mode)
    num_elements = 10
    ds = self.make_distributed_range_dataset(10, cluster)
    results = [elem.numpy() for elem in ds]
    self.assertEqual(list(range(num_elements)), results)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeSparse(self):
    cluster = self.create_cluster(num_workers=1)
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
    cluster = self.create_cluster(num_workers=1)
    ds = dataset_ops.Dataset.from_tensor_slices([1, 5, 3, 2, 8])
    ds = ds.map(math_ops.range)
    ds = ds.apply(batching.dense_to_ragged_batch(2))
    ds = self.make_distributed_dataset(ds, cluster)
    results = [elem.to_tensor() for elem in ds]
    self.assertAllEqual(results[0], [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4]])
    self.assertAllEqual(results[1], [[0, 1, 2], [0, 1, 0]])
    self.assertAllEqual(results[2], [[0, 1, 2, 3, 4, 5, 6, 7]])

  @combinations.generate(test_base.eager_only_combinations())
  def testDifferentShuffleOrders(self):
    random_seed.set_random_seed(None)
    num_elements = 100
    cluster = self.create_cluster(num_workers=2)
    ds = dataset_ops.Dataset.range(num_elements)
    ds = ds.shuffle(num_elements)
    ds = self.make_distributed_dataset(ds, cluster)
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
    cluster = self.create_cluster(num_workers=1)
    num_elements = 3
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    for _ in range(10):
      self.assertEqual(list(range(num_elements)), [elem.numpy() for elem in ds])

  @combinations.generate(test_base.eager_only_combinations())
  def testRepeatedDataset(self):
    cluster = self.create_cluster(num_workers=1)
    num_elements = 10
    num_repetitions = 5
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    ds = ds.repeat(num_repetitions)
    self.assertDatasetProduces(
        ds, expected_output=num_repetitions * list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testConcurrentEpoch(self):
    cluster = self.create_cluster(num_workers=1)
    num_elements = 10
    num_datasets = 3
    iterators = []
    results = []
    for _ in range(num_datasets):
      ds = self.make_distributed_range_dataset(num_elements, cluster)
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
    cluster = self.create_cluster(num_workers=1)
    num_elements = 10
    num_iterators = 3
    ds = self.make_distributed_range_dataset(num_elements, cluster)
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
    cluster = self.create_cluster(num_workers=num_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(num_elements, cluster)
    results = [elem.numpy() for elem in ds]
    self.assertCountEqual(num_workers * list(range(num_elements)), results)

  @combinations.generate(test_base.eager_only_combinations())
  def testMaxOutstandingRequests(self):
    num_workers = 3
    cluster = self.create_cluster(num_workers=num_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, max_outstanding_requests=1)
    self.assertCountEqual(num_workers * list(range(num_elements)),
                          self.getDatasetOutput(ds))

  @combinations.generate(test_base.eager_only_combinations())
  def testInsideFunction(self):
    num_workers = 3
    cluster = self.create_cluster(num_workers=num_workers)
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

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedJobName(self):
    cluster = self.create_cluster(num_workers=1)
    num_elements = 1000

    def make_ds():
      return dataset_ops.Dataset.range(num_elements).shuffle(num_elements)

    ds1 = self.make_distributed_dataset(make_ds(), cluster, job_name="job_name")
    ds2 = self.make_distributed_dataset(make_ds(), cluster, job_name="job_name")
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
    cluster = self.create_cluster(num_workers=1)
    num_elements = 10
    ds1 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name1")
    ds2 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name2")
    self.assertDatasetProduces(ds1, list(range(num_elements)))
    self.assertDatasetProduces(ds2, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedJobNameMultiIteration(self):
    cluster = self.create_cluster(num_workers=1)
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

  @combinations.generate(test_base.eager_only_combinations())
  def testSharedJobNameRepeat(self):
    cluster = self.create_cluster(num_workers=1)
    num_elements = 100
    num_repetitions = 3
    ds1 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name")
    ds1 = ds1.repeat(num_repetitions)
    ds2 = self.make_distributed_range_dataset(
        num_elements, cluster, job_name="job_name")
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
    cluster = self.create_cluster(
        num_workers=1, job_gc_check_interval_ms=50, job_gc_timeout_ms=20)
    num_elements = 100
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, job_name=job_name)
    it = iter(ds)
    self.assertEqual(next(it).numpy(), 0)
    self.assertEqual(cluster.num_tasks_on_worker(), 1)
    del it
    while cluster.num_tasks_on_worker() > 0:
      time.sleep(0.1)

  @combinations.generate(test_base.eager_only_combinations())
  def testDontGcUsedJob(self):
    cluster = self.create_cluster(
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
    self.assertEqual(2, cluster.num_tasks_on_worker())
    del it1
    del it2
    # Check that only the first job is gced. The second job will not be gced
    # because there is still an outstanding iterator for it.
    while cluster.num_tasks_on_worker() > 1:
      time.sleep(0.1)
    self.assertEqual(1, cluster.num_tasks_on_worker())

  @combinations.generate(test_base.eager_only_combinations())
  def testApplyDeterminismOption(self):
    elements = list(range(10))
    cluster = self.create_cluster(num_workers=1)

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

    options = dataset_ops.Options()
    options.experimental_external_state_policy = external_state_policy
    ds = ds.with_options(options)

    cluster = self.create_cluster(num_workers=3)
    ds = self.make_distributed_dataset(ds, cluster)
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
    cluster = self.create_cluster(num_workers=2)
    vals = [5, 1, 2, 4]
    ds = dataset_ops.Dataset.from_tensor_slices(vals)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(ds, vals, assert_items_equal=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpochInterleave(self):
    cluster = self.create_cluster(num_workers=2)
    elements = [1, 5, 0]
    ds = dataset_ops.Dataset.from_tensor_slices(elements)
    ds = ds.interleave(lambda x: dataset_ops.Dataset.from_tensor_slices([x]))
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(ds, elements, assert_items_equal=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpochParallelInterleave(self):
    cluster = self.create_cluster(num_workers=2)
    elements = [1, 5, 0]
    ds = dataset_ops.Dataset.from_tensor_slices(elements)
    ds = ds.interleave(
        lambda x: dataset_ops.Dataset.from_tensor_slices([x]),
        num_parallel_calls=dataset_ops.AUTOTUNE)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(ds, elements, assert_items_equal=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpochFlatMap(self):
    cluster = self.create_cluster(num_workers=2)
    elements = [1, 5, 0]
    ds = dataset_ops.Dataset.from_tensor_slices(elements)
    ds = ds.flat_map(lambda x: dataset_ops.Dataset.from_tensor_slices([x]))
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(ds, elements, assert_items_equal=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpochRepeat(self):
    cluster = self.create_cluster(num_workers=2)
    num_repeats = 5
    num_elements = 20
    ds = dataset_ops.Dataset.range(num_elements).repeat(num_repeats)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(
        ds, num_repeats * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpochForeverRepeat(self):
    cluster = self.create_cluster(num_workers=2)
    num_elements = 20
    elements_to_read = 1000
    ds = dataset_ops.Dataset.range(num_elements).repeat()
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    it = iter(ds)
    results = {}
    for _ in range(elements_to_read):
      val = next(it).numpy()
      if val not in results:
        results[val] = 0
      results[val] += 1
    for i in range(num_elements):
      self.assertGreater(results[i], elements_to_read / num_elements / 2)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpochForeverRepeatFewElements(self):
    num_workers = 5
    cluster = self.create_cluster(num_workers=num_workers)
    # Less than the number of workers, so that some workers get zero elements on
    # the first repetition.
    num_elements = 1
    ds = dataset_ops.Dataset.range(num_elements).repeat()
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    it = iter(ds)
    for _ in range(100):
      self.assertEqual(next(it).numpy(), 0)

    # Stop all but one worker and check that we can still read.
    for i in range(num_workers - 1):
      cluster.workers[i]._stop()
    for _ in range(100):
      self.assertEqual(next(it).numpy(), 0)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpochShuffleAndRepeat(self):
    cluster = self.create_cluster(num_workers=2)
    num_repeats = 5
    num_elements = 20
    ds = dataset_ops.Dataset.range(num_elements).shuffle(num_elements).repeat(
        num_repeats)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(
        ds, num_repeats * list(range(num_elements)), assert_items_equal=True)

  def testDistributeFromInterleave(self):
    cluster = self.create_cluster(num_workers=1)
    ds = dataset_ops.Dataset.range(2)

    def interleave_fn(_):
      dataset = dataset_ops.Dataset.range(2)
      self.make_distributed_dataset(dataset, cluster)
      return dataset

    ds = ds.interleave(interleave_fn, cycle_length=2)
    self.assertDatasetProduces(ds, [0, 0, 1, 1])

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedEpoch(self):
    cluster = self.create_cluster(num_workers=2)
    num_elements = 100
    ds = dataset_ops.Dataset.range(num_elements)
    ds = self.make_distributed_dataset(
        ds, cluster, processing_mode="distributed_epoch")
    self.assertDatasetProduces(
        ds, list(range(num_elements)), assert_items_equal=True)

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
  def testZipDifferentProcessingModesDatasets(self):
    cluster = self.create_cluster(num_workers=1)
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

  @combinations.generate(test_base.eager_only_combinations())
  def testZipDifferentProcessingModesDatasetsSharedJobName(self):
    cluster = self.create_cluster(num_workers=1)
    num_elements = 100
    ds1 = dataset_ops.Dataset.range(num_elements)
    ds1 = self.make_distributed_dataset(
        ds1, cluster, processing_mode="distributed_epoch", job_name="job_name")
    ds2 = dataset_ops.Dataset.range(num_elements)
    ds2 = self.make_distributed_dataset(
        ds2, cluster, processing_mode="parallel_epochs", job_name="job_name")
    ds = dataset_ops.Dataset.zip((ds1, ds2))
    with self.assertRaisesRegex(errors.FailedPreconditionError,
                                "but there is already an existing job"):
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.eager_only_combinations())
  def testFromDatasetId(self):
    cluster = self.create_cluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    dataset_id = data_service_ops.register_dataset(cluster.target, ds)
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs", cluster.target, dataset_id, ds.element_spec)
    self.assertDatasetProduces(from_dataset_id_ds, list(range(num_elements)))

  @combinations.generate(test_base.eager_only_combinations())
  def testFromDatasetIdMultipleComponents(self):
    cluster = self.create_cluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    ds = dataset_ops.Dataset.zip({"a": (ds, ds), "b": ds})
    dataset_id = data_service_ops.register_dataset(cluster.target, ds)
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs", cluster.target, dataset_id, ds.element_spec)
    output = self.getDatasetOutput(from_dataset_id_ds)
    for i in range(num_elements):
      self.assertEqual(i, output[i]["a"][0])
      self.assertEqual(i, output[i]["a"][1])
      self.assertEqual(i, output[i]["b"])

  @combinations.generate(test_base.eager_only_combinations())
  def testFromDatasetIdWrongElementSpec(self):
    cluster = self.create_cluster(num_workers=1)

    num_elements = 10
    ds = dataset_ops.Dataset.range(num_elements)
    dataset_id = data_service_ops.register_dataset(cluster.target, ds)
    wrong_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.variant)
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs", cluster.target, dataset_id, wrong_spec)
    with self.assertRaisesRegex(errors.FailedPreconditionError,
                                "Expected a tensor of type variant"):
      self.evaluate(self.getNext(from_dataset_id_ds)())

  @combinations.generate(test_base.eager_only_combinations())
  def testFromDatasetIdNotRegistered(self):
    cluster = self.create_cluster(num_workers=1)

    dataset_id = 0
    element_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.variant)
    from_dataset_id_ds = data_service_ops.from_dataset_id(
        "parallel_epochs", cluster.target, dataset_id, element_spec)
    with self.assertRaisesRegex(errors.NotFoundError, "Dataset id"):
      self.evaluate(self.getNext(from_dataset_id_ds)())

  @combinations.generate(test_base.default_test_combinations())
  def testCancellation(self):
    self.skipTest("b/162521601")
    sleep_microseconds = int(1e6) * 1000

    cluster = self.create_cluster(num_workers=1)
    # Create a dataset which produces the first element quickly, and the second
    # element slowly. Fetching the first element triggers prefetching of the
    # second element, which we should be able to cancel.
    slow = dataset_ops.Dataset.range(1)
    slow = slow.apply(testing.sleep(sleep_microseconds))
    ds = dataset_ops.Dataset.range(1).concatenate(slow)
    ds = self.make_distributed_dataset(ds, cluster)
    ds = ds.prefetch(1)
    get_next = self.getNext(ds, requires_initialization=True)
    self.assertEqual(0, self.evaluate(get_next()))
    # Without properly implemented cancellation, we will hang here while trying
    # to garbage collect the dataset iterator.

  @combinations.generate(test_base.eager_only_combinations())
  def testRegisterEquivalentDatasets(self):
    ds_1 = dataset_ops.Dataset.range(10)
    ds_2 = dataset_ops.Dataset.range(10)
    cluster = self.create_cluster(num_workers=1)
    id_1 = data_service_ops.register_dataset(cluster.target, ds_1)
    id_2 = data_service_ops.register_dataset(cluster.target, ds_2)
    self.assertEqual(id_1.numpy(), id_2.numpy())

  @combinations.generate(test_base.eager_only_combinations())
  def testRegisterDifferentDatasets(self):
    ds_1 = dataset_ops.Dataset.range(10)
    ds_2 = dataset_ops.Dataset.range(20)
    cluster = self.create_cluster(num_workers=1)
    id_1 = data_service_ops.register_dataset(cluster.target, ds_1)
    id_2 = data_service_ops.register_dataset(cluster.target, ds_2)
    self.assertNotEqual(id_1.numpy(), id_2.numpy())

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeParallelToDistributed(self):

    cluster_1 = self.create_cluster(num_workers=1)
    cluster_2 = self.create_cluster(num_workers=1)
    num_sizes = 10
    size_repeats = 5
    numbers = [1 * i for i in range(num_sizes)] * size_repeats
    ds = dataset_ops.Dataset.from_tensor_slices(numbers)
    ds = self.make_distributed_dataset(
        ds, cluster_1, processing_mode="parallel_epochs")
    ds = ds.map(lambda x: x + 1)
    ds = self.make_distributed_dataset(
        ds, cluster_2, processing_mode="distributed_epoch")

    error_regex = "Cannot create a split provider for dataset of type DataServiceDataset"  # pylint: disable=line-too-long
    with self.assertRaisesRegex(errors.UnimplementedError, error_regex):
      self.getDatasetOutput(ds)

  @combinations.generate(test_base.eager_only_combinations())
  def testDistributeDistributedToParallel(self):

    cluster_1 = self.create_cluster(num_workers=1)
    cluster_2 = self.create_cluster(num_workers=1)
    num_sizes = 10
    size_repeats = 5
    numbers = [1 * i for i in range(num_sizes)] * size_repeats
    ds = dataset_ops.Dataset.from_tensor_slices(numbers)
    ds = self.make_distributed_dataset(
        ds, cluster_1, processing_mode="distributed_epoch")
    ds = ds.map(lambda x: x + 1)
    ds = self.make_distributed_dataset(
        ds, cluster_2, processing_mode="parallel_epochs")

    self.assertDatasetProduces(
        ds, [i + 1 for i in numbers], assert_items_equal=True)

  def testParallelZippedDistributedDatasets(self):

    cluster_1 = self.create_cluster(num_workers=1)
    cluster_2 = self.create_cluster(num_workers=1)
    cluster_3 = self.create_cluster(num_workers=1)
    num_sizes = 10
    size_repeats = 5
    numbers = [1 * i for i in range(num_sizes)] * size_repeats
    ds1 = dataset_ops.Dataset.from_tensor_slices(numbers)
    ds1 = self.make_distributed_dataset(ds1, cluster_1)

    ds2 = dataset_ops.Dataset.from_tensor_slices(numbers)
    ds2 = self.make_distributed_dataset(ds2, cluster_2)

    ds3 = dataset_ops.Dataset.zip((ds1, ds2))
    ds3 = self.make_distributed_dataset(
        ds3, cluster_3, processing_mode="parallel_epochs")

    self.assertDatasetProduces(
        ds3, list(zip(numbers, numbers)), assert_items_equal=True)

  def testDistributedZippedDistributedDatasets(self):

    cluster_1 = self.create_cluster(num_workers=1)
    cluster_2 = self.create_cluster(num_workers=1)
    cluster_3 = self.create_cluster(num_workers=1)
    num_sizes = 10
    size_repeats = 5
    numbers = [1 * i for i in range(num_sizes)] * size_repeats
    ds1 = dataset_ops.Dataset.from_tensor_slices(numbers)
    ds1 = self.make_distributed_dataset(ds1, cluster_1)

    ds2 = dataset_ops.Dataset.from_tensor_slices(numbers)
    ds2 = self.make_distributed_dataset(ds2, cluster_2)

    ds3 = dataset_ops.Dataset.zip((ds1, ds2))
    ds3 = self.make_distributed_dataset(
        ds3, cluster_3, processing_mode="distributed_epoch")

    error_regex = "Cannot create a split provider for dataset of type ZipDataset"  # pylint: disable=line-too-long
    with self.assertRaisesRegex(errors.UnimplementedError, error_regex):
      self.getDatasetOutput(ds3)

  @combinations.generate(test_base.eager_only_combinations())
  def testTwoLevelDistribute(self):
    cluster_1_size = 3
    cluster_1 = self.create_cluster(num_workers=cluster_1_size)
    cluster_2 = self.create_cluster(num_workers=1)
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

    it = iter(ds)
    for _ in range(num_sizes):
      element = next(it).numpy()
      for _ in range(1, cluster_1_size):
        self.assertAllEqual(next(it).numpy(), element)
    self.assertEmpty(list(it))

  # @combinations.generate(test_base.eager_only_combinations())
  # def testCyclicDistribute(self):

  #   cluster_1 = self.create_cluster(num_workers=1)
  #   cluster_2 = self.create_cluster(num_workers=1)
  #   num_sizes = 10
  #   size_repeats = 5
  #   numbers = [1 * i for i in range(num_sizes)] * size_repeats
  #   ds = dataset_ops.Dataset.from_tensors(numbers)
  #   ds = self.make_distributed_dataset(ds, cluster_1, processing_mode="parallel_epochs")
  #   ds = ds.map(lambda x: x + 1)
  #   ds = self.make_distributed_dataset(ds, cluster_2, processing_mode="parallel_epochs")
  #   ds = ds.map(lambda x: x - 1)
  #   ds = self.make_distributed_dataset(
  #       ds, cluster_1, processing_mode="parallel_epochs", job_name="temp_jobname")

  #   self.assertDatasetProduces(ds, numbers, assert_items_equal=True)

  @combinations.generate(
      combinations.times(test_base.eager_only_combinations()))
  def testDistributeLargeGraph(self):
    cluster = self.create_cluster(
        num_workers=1, work_dir=NO_WORK_DIR, fault_tolerant_mode=False)
    # Larger than default OSS grpc message size limit of 4MB.
    tensor = array_ops.ones((2, 1000, 1000), dtype=dtypes.float32)
    ds = dataset_ops.Dataset.from_tensors(tensor)
    ds = self.make_distributed_dataset(ds, cluster)
    self.assertDatasetProduces(ds, [tensor])


if __name__ == "__main__":
  test.main()
