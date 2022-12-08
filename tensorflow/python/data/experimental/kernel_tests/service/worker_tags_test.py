# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests tf.data service reading from workers with specific tags."""

import time
import multiprocessing

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import multi_process_cluster
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.framework import combinations

_COLOCATED_WORKER_TAG = "COLOCATED"


class WorkerTagsTest(data_service_test_base.TestBase, parameterized.TestCase):
  """Tests tf.data service reading from local or non-TPU workers.

  When `target_workers` is "AUTO", tf.data service avoids cross-TPU reads, to
  avoid RPCS and data serialization / deserialization, thus improving resource
  utilization of TPU hosts.
  """

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_remote_workers=[0, 3])))
  def testReadFromLocalWorker(self, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=1,
        num_remote_workers=num_remote_workers,
        worker_tags=[_COLOCATED_WORKER_TAG])
    num_elements = 100
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    # Only reads from the local worker.
    self.assertDatasetProduces(dataset, list(range(num_elements)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testReadFromLocalAndNonTpuWorkers(self, num_local_workers,
                                        num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers,
        worker_tags=[_COLOCATED_WORKER_TAG])
    cluster.start_remote_worker(worker_tags=None)

    num_elements = 100
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    # Reads from the local worker or non-colocated worker.
    self.assertDatasetProduces(
        dataset, (num_local_workers + 1) * list(range(num_elements)),
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(num_remote_workers=[0, 3])))
  def testLocalWorkerHasNoTag(self, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=0,
        num_remote_workers=num_remote_workers,
        worker_tags=[_COLOCATED_WORKER_TAG])
    cluster.start_local_worker(worker_tags=None)

    num_elements = 100
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    # Only reads from the local worker.
    self.assertDatasetProduces(dataset, list(range(num_elements)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testReadFromLocalAndNonTpuWorkers_DynamicSharding(self, num_local_workers,
                                                        num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=3,
        worker_tags=[_COLOCATED_WORKER_TAG])
    cluster.start_remote_worker(worker_tags=None)

    num_elements = 100
    dataset = self.make_distributed_range_dataset(
        num_elements,
        cluster,
        processing_mode=data_service_ops.ShardingPolicy.DYNAMIC)
    self.assertDatasetProduces(
        dataset, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testReadFromLocalWorker_StaticSharding(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=1,
        num_remote_workers=3,
        worker_addresses=["localhost:%port%"] * 5,
        worker_tags=[_COLOCATED_WORKER_TAG])
    cluster.start_remote_worker(worker_tags=None)

    num_elements = 100
    dataset = self.make_distributed_range_dataset(
        num_elements,
        cluster,
        processing_mode=data_service_ops.ShardingPolicy.FILE_OR_DATA)

    # Static sharding will only read from the local worker.
    self.assertDatasetProduces(dataset, list(range(0, num_elements, 5)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[1, 3])))
  def testCoordinatedRead(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers,
        worker_tags=[_COLOCATED_WORKER_TAG])
    num_consumers = 4
    dataset = self.make_coordinated_read_dataset(cluster, num_consumers)
    get_next = self.getNext(dataset, requires_initialization=True)
    results = [self.evaluate(get_next()) for _ in range(200)]
    self.checkCoordinatedReadGroups(results, num_consumers)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[1, 3])))
  def testAddRemoteWorkersMidJob(self, num_local_workers, num_remote_workers):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers,
        worker_tags=[_COLOCATED_WORKER_TAG])

    # num_elements needs to be bigger than (100 + <cpu core count>), the extra
    # 100 is just a bit of margin. The CPU core count is involved as
    # elements are prefetched, one element per CPU core.
    num_elements = 200 + multiprocessing.cpu_count()
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    get_next = self.getNext(dataset)
    results = [self.evaluate(get_next()) for _ in range(100)]

    # Will only read from the two non-TPU workers.
    cluster.start_remote_worker(worker_tags=None)
    cluster.start_remote_worker(worker_tags=[_COLOCATED_WORKER_TAG])
    cluster.start_remote_worker(worker_tags=None)
    cluster.start_remote_worker(worker_tags=[_COLOCATED_WORKER_TAG])
    expect_num_workers_to_read = num_local_workers + 2

    # Wait for the new worker to register with the dispatcher.
    while cluster._dispatcher._num_workers() < (num_local_workers +
                                                num_remote_workers + 4):
      time.sleep(10 / 1000)  # 10ms

    results += self.getIteratorOutput(get_next)
    self.assertCountEqual(
        results, expect_num_workers_to_read * list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testMultipleTags(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=1,
        num_remote_workers=3,
        worker_tags=[_COLOCATED_WORKER_TAG, "COLOCATED_2", "COLOCATED_3"])
    num_elements = 100
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    # Only reads from the local worker.
    self.assertDatasetProduces(dataset, list(range(num_elements)))

  @combinations.generate(test_base.default_test_combinations())
  def testUnusedTags(self):
    cluster = multi_process_cluster.MultiProcessCluster(
        num_local_workers=1,
        num_remote_workers=3,
        worker_tags=["Unused tag 1", "Unused tag 2", "Unused tag 3"])
    num_elements = 100
    dataset = self.make_distributed_range_dataset(num_elements, cluster)
    # The tags don't have an effect. tf.data service will read from all workers.
    self.assertDatasetProduces(
        dataset, 4 * list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(test_base.default_test_combinations())
  def testInvalidTag(self):
    with self.assertRaisesRegex(RuntimeError, "Worker tags cannot be empty."):
      _ = multi_process_cluster.MultiProcessCluster(
          num_local_workers=1,
          num_remote_workers=3,
          worker_tags=["", _COLOCATED_WORKER_TAG])


if __name__ == "__main__":
  multi_process_cluster.test_main()
