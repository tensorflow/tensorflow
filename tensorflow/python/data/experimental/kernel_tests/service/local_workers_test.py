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
"""Tests tf.data service with local and remote workers."""

import tempfile

from absl.testing import parameterized

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.platform import googletest


class RemoteWorkerProcess(multi_process_lib.Process):
  """Runs worker servers in a new process to simulate remote workers."""

  def __init__(self, dispatcher_address, num_workers, pipe_writer):
    super(RemoteWorkerProcess, self).__init__()
    self._dispatcher_address = dispatcher_address
    self._num_workers = num_workers
    self._pipe_writer = pipe_writer

  # `run` is hidden in multi_process_lib.py. It is assigned with a decorated
  # `run` that runs the process in `absl.app.run`.
  def run(self):  # pylint: disable=method-hidden
    self.start_workers()

  def start_workers(self):
    self._workers = []
    for _ in range(self._num_workers):
      worker = server_lib.WorkerServer(
          server_lib.WorkerConfig(dispatcher_address=self._dispatcher_address),
          start=True)
      self._workers.append(worker)

    self._pipe_writer.send("Remote workers are ready.")
    for worker in self._workers:
      worker.join()


class MultiProcessCluster(object):
  """Creates a tf.data service cluster with local and remote workers."""

  def __init__(self,
               num_local_workers,
               num_remote_workers,
               dispatcher_port=0,
               worker_shutdown_quiet_period_ms=0):
    work_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    self._worker_shutdown_quiet_period_ms = worker_shutdown_quiet_period_ms
    self._dispatcher = server_lib.DispatchServer(
        server_lib.DispatcherConfig(
            port=dispatcher_port, work_dir=work_dir, protocol="grpc"),
        start=True)
    self._local_workers = self.start_local_workers(num_local_workers)
    self.start_remote_workers(num_remote_workers)

  def dispatcher_address(self):
    return self._dispatcher.target.split("://")[1]

  def start_local_workers(self, num_workers):
    workers = []
    for _ in range(num_workers):
      worker = data_service_test_base.TestWorker(
          self.dispatcher_address(), self._worker_shutdown_quiet_period_ms)
      worker.start()
      workers.append(worker)
    return workers

  def start_remote_workers(self, num_workers):
    pipe_reader, pipe_writer = multi_process_lib.multiprocessing.Pipe(
        duplex=False)
    self._remote_workers_process = RemoteWorkerProcess(
        self.dispatcher_address(), num_workers, pipe_writer)
    self._remote_workers_process.start()
    pipe_reader.recv()

  def __del__(self):
    for worker in self._local_workers:
      worker.stop()
    self._remote_workers_process.terminate()
    self._remote_workers_process.close()
    self._dispatcher._stop()


class LocalWorkersTest(data_service_test_base.TestBase, parameterized.TestCase):
  """Tests reading from local workers if `target_workers` is `local`."""

  def testOneLocalWorker(self):
    cluster = MultiProcessCluster(num_local_workers=1, num_remote_workers=5)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="local")
    self.assertDatasetProduces(ds, list(range(num_elements)))

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testLocalWorkers(self, num_local_workers, num_remote_workers):
    cluster = MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    self.assertDatasetProduces(
        ds,
        num_local_workers * list(range(num_elements)),
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testRepeatedDataset(self, num_local_workers, num_remote_workers):
    cluster = MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    num_repetitions = 5
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    ds = ds.repeat(num_repetitions)
    self.assertDatasetProduces(
        ds,
        expected_output=num_local_workers * num_repetitions *
        list(range(num_elements)),
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testPrefetchingDataset(self, num_local_workers, num_remote_workers):
    cluster = MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    ds = ds.prefetch(10)
    self.assertDatasetProduces(
        ds,
        expected_output=num_local_workers * list(range(num_elements)),
        assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testMultipleEpochs(self, num_local_workers, num_remote_workers):
    cluster = MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    for _ in range(10):
      self.assertDatasetProduces(
          ds,
          num_local_workers * list(range(num_elements)),
          assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testDistributedEpoch(self, num_local_workers, num_remote_workers):
    cluster = MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 100
    ds = self.make_distributed_range_dataset(
        num_elements,
        cluster,
        processing_mode="distributed_epoch",
        target_workers="LOCAL")
    self.assertDatasetProduces(
        ds, list(range(num_elements)), assert_items_equal=True)

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testEmptyDataset(self, num_local_workers, num_remote_workers):
    cluster = MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 0
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    self.assertDatasetProduces(ds, [])

  @combinations.generate(
      combinations.times(
          test_base.default_test_combinations(),
          combinations.combine(
              num_local_workers=[1, 3], num_remote_workers=[0, 3])))
  def testNonLocalRead(self, num_local_workers, num_remote_workers):
    """This test ensures the remote workers are running and producing data."""

    cluster = MultiProcessCluster(
        num_local_workers=num_local_workers,
        num_remote_workers=num_remote_workers)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="any")
    num_workers = num_local_workers + num_remote_workers
    self.assertDatasetProduces(
        ds, num_workers * list(range(num_elements)), assert_items_equal=True)

  def testNoLocalWorker(self):
    cluster = MultiProcessCluster(num_local_workers=0, num_remote_workers=3)
    num_elements = 10
    ds = self.make_distributed_range_dataset(
        num_elements, cluster, target_workers="LOCAL")
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "no local worker is found"):
      get_next = self.getNext(ds)
      self.evaluate(get_next())

  def testCoordinatedRead(self):
    cluster = MultiProcessCluster(num_local_workers=3, num_remote_workers=3)
    ds = dataset_ops.Dataset.range(10).repeat()
    ds = self.make_distributed_dataset(
        ds,
        cluster,
        job_name="test",
        consumer_index=0,
        num_consumers=3,
        target_workers="LOCAL")
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Coordinated reads require non-local workers"):
      get_next = self.getNext(ds)
      self.evaluate(get_next())


if __name__ == "__main__":
  multi_process_lib.test_main()
