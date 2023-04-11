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
"""Test base for tf.data service tests."""
import tempfile

from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

# This will be resolved to a tmp directory by `start_dispatch_server`.
TMP_WORK_DIR = "tmp_work_dir_placeholder"
# `""` indicates not to use a work directory.
NO_WORK_DIR = ""
# We use a faster than normal heartbeat interval so that tests run faster.
TEST_HEARTBEAT_INTERVAL_MS = 100
TEST_DISPATCHER_TIMEOUT_MS = 5000
TEST_WORKER_TIMEOUT_MS = 200
TEST_JOB_GC_CHECK_INTERNAL_MS = 1000
PROTOCOL = "grpc"


def all_cluster_configurations():
  with_work_dir = combinations.combine(
      work_dir=TMP_WORK_DIR, fault_tolerant_mode=[True, False])
  without_work_dir = combinations.combine(
      work_dir=NO_WORK_DIR, fault_tolerant_mode=False)
  return with_work_dir + without_work_dir


def _make_worker(dispatcher_address,
                 data_transfer_protocol,
                 shutdown_quiet_period_ms=0,
                 port=0,
                 worker_tags=None,
                 cross_trainer_cache_size_bytes=None):
  """Creates a worker server."""
  defaults = server_lib.WorkerConfig(dispatcher_address=dispatcher_address)
  config_proto = service_config_pb2.WorkerConfig(
      dispatcher_address=dispatcher_address,
      worker_address=defaults.worker_address,
      port=port,
      protocol=PROTOCOL,
      worker_tags=worker_tags,
      heartbeat_interval_ms=TEST_HEARTBEAT_INTERVAL_MS,
      dispatcher_timeout_ms=TEST_DISPATCHER_TIMEOUT_MS,
      data_transfer_protocol=data_transfer_protocol,
      data_transfer_address=defaults.worker_address,
      shutdown_quiet_period_ms=shutdown_quiet_period_ms,
      cross_trainer_cache_size_bytes=cross_trainer_cache_size_bytes)
  return server_lib.WorkerServer(config_proto, start=False)


# pylint: disable=protected-access
class TestWorker:
  """A tf.data service worker."""

  def __init__(self,
               dispatcher_address,
               shutdown_quiet_period_ms,
               data_transfer_protocol=None,
               port=0,
               worker_tags=None,
               cross_trainer_cache_size_bytes=None):
    self._dispatcher_address = dispatcher_address
    self._shutdown_quiet_period_ms = shutdown_quiet_period_ms
    self._server = _make_worker(
        dispatcher_address,
        data_transfer_protocol,
        shutdown_quiet_period_ms,
        port=port,
        worker_tags=worker_tags,
        cross_trainer_cache_size_bytes=cross_trainer_cache_size_bytes)
    self._running = False
    self._data_transfer_protocol = data_transfer_protocol

  def stop(self):
    self._server._stop()
    self._running = False

  def start(self):
    self._server.start()
    self._port = int(self._server._address.split(":")[1])
    self._running = True

  def restart(self, use_same_port=True):
    """Restarts the worker, stopping it first if it is already running."""
    if self._running:
      self.stop()
    port = 0
    if use_same_port:
      port = self._port
    self._server = _make_worker(self._dispatcher_address,
                                self._data_transfer_protocol,
                                self._shutdown_quiet_period_ms, port)
    self._server.start()
    self._port = int(self._server._address.split(":")[1])
    self._running = True

  def join(self):
    self._server.join()

  def num_tasks(self):
    return self._server._num_tasks()

  def snapshot_task_progresses(self):
    return self._server._snapshot_task_progresses()

  def worker_address(self):
    return self._server._address


class TestCluster:
  """Test tf.data service cluster."""

  def __init__(
      self,
      num_workers,
      dispatcher_port=0,
      work_dir=TMP_WORK_DIR,
      fault_tolerant_mode=True,
      job_gc_check_interval_ms=TEST_JOB_GC_CHECK_INTERNAL_MS,
      job_gc_timeout_ms=None,
      worker_timeout_ms=TEST_WORKER_TIMEOUT_MS,
      worker_shutdown_quiet_period_ms=0,
      start=True,
      data_transfer_protocol=None,
  ):
    """Creates a tf.data service test cluster.

    Args:
      num_workers: The number of workers to initially add to the cluster.
      dispatcher_port: The port to use for the dispatcher.
      work_dir: The work directory to use for the dispatcher. If set to
        `TMP_WORK_DIR`, the cluster will create a new temporary directory to use
        as the work directory. If set to `NO_WORK_DIR`, no work directory will
        be used.
      fault_tolerant_mode: Whether the dispatcher should write its state to a
        journal so that it can recover from restarts.
      job_gc_check_interval_ms: How often the dispatcher should scan through to
        delete old and unused jobs, in milliseconds.
      job_gc_timeout_ms: How long a job needs to be unused before it becomes a
        candidate for garbage collection, in milliseconds.
      worker_timeout_ms: How long to wait for a worker to heartbeat before
        considering it missing, in milliseconds.
      worker_shutdown_quiet_period_ms: When shutting down a worker, how long to
        wait for the gRPC server to process the final requests.
      start: Whether to immediately start the servers in the cluster. If
        `False`, the servers can be started later by calling
        `start_dispatcher()` and `start_workers()`.
      data_transfer_protocol: (Optional.) The protocol to use for transferring
        data with the tf.data service.
    """
    if work_dir == TMP_WORK_DIR:
      work_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    self._worker_shutdown_quiet_period_ms = worker_shutdown_quiet_period_ms
    self._data_transfer_protocol = data_transfer_protocol
    self.dispatcher = server_lib.DispatchServer(
        server_lib.DispatcherConfig(
            port=dispatcher_port,
            work_dir=work_dir,
            protocol=PROTOCOL,
            fault_tolerant_mode=fault_tolerant_mode,
            job_gc_check_interval_ms=job_gc_check_interval_ms,
            job_gc_timeout_ms=job_gc_timeout_ms,
            worker_timeout_ms=worker_timeout_ms,
        ),
        start=start,
    )

    self.workers = []
    for _ in range(num_workers):
      self.add_worker(start=start)

  def dispatcher_address(self):
    return self.dispatcher.target.split("://")[1]

  def add_worker(self, start=True):
    worker = TestWorker(self.dispatcher_address(),
                        self._worker_shutdown_quiet_period_ms,
                        self._data_transfer_protocol)
    if start:
      worker.start()
    self.workers.append(worker)

  def start_dispatcher(self):
    self.dispatcher.start()

  def start_workers(self):
    for worker in self.workers:
      worker.start()

  def stop_dispatcher(self):
    # pylint: disable=protected-access
    self.dispatcher._stop()

  def stop_worker(self, index):
    self.workers[index].stop()

  def stop_workers(self):
    for worker in self.workers:
      worker.stop()

  # pylint: disable=protected-access
  def restart_dispatcher(self):
    """Stops `dispatcher` and creates a new dispatcher with the same port.

    Restarting is supported only when the dispatcher is configured with
    `fault_tolerant_mode=True`.
    """
    if not self.dispatcher._config.fault_tolerant_mode:
      raise ValueError(
          "Trying to restart the dispatcher without fault-tolerance.")
    port = int(self.dispatcher_address().split(":")[1])
    self.dispatcher._stop()
    self.dispatcher = server_lib.DispatchServer(
        server_lib.DispatcherConfig(
            port=port,
            work_dir=self.dispatcher._config.work_dir,
            protocol=PROTOCOL,
            fault_tolerant_mode=self.dispatcher._config.fault_tolerant_mode))

  def num_registered_workers(self):
    return self.dispatcher._num_workers()

  def num_tasks_on_workers(self):
    return sum(worker.num_tasks() for worker in self.workers)

  def snapshot_streams(self, path):
    return self.dispatcher._snapshot_streams(path)

  def __del__(self):
    # Destroy workers before the dispatcher for clean shutdown.
    self.workers.clear()
    del self.dispatcher


class TestBase(test_base.DatasetTestBase):
  """Base class for tf.data service tests."""

  def setUp(self):
    self.default_data_transfer_protocol = None
    self.default_compression = "AUTO"

  def set_default_data_transfer_protocol(self, protocol):
    self.default_data_transfer_protocol = protocol

  def set_default_compression(self, compression):
    self.default_compression = compression

  def make_test_cluster(self, *args, **kwargs):
    if "data_transfer_protocol" not in kwargs:
      kwargs["data_transfer_protocol"] = self.default_data_transfer_protocol
    return TestCluster(*args, **kwargs)

  def make_distributed_dataset(self,
                               dataset,
                               cluster,
                               processing_mode="parallel_epochs",
                               **kwargs):
    kwargs["task_refresh_interval_hint_ms"] = 20
    if "data_transfer_protocol" not in kwargs:
      kwargs["data_transfer_protocol"] = self.default_data_transfer_protocol
    if "compression" not in kwargs:
      kwargs["compression"] = self.default_compression

    # pylint: disable=protected-access
    return dataset.apply(
        data_service_ops._distribute(
            processing_mode,
            cluster.dispatcher_address(),
            **kwargs))

  def make_distributed_range_dataset(self,
                                     num_elements,
                                     cluster,
                                     **kwargs):
    dataset = dataset_ops.Dataset.range(num_elements)
    return self.make_distributed_dataset(dataset, cluster, **kwargs)

  def make_coordinated_read_dataset(
      self,
      cluster,
      num_consumers,
      sharding_policy=data_service_ops.ShardingPolicy.OFF):
    """Creates a dataset that performs coordinated reads.

    The dataset simulates `num_consumers` consumers by using parallel
    interleave to read with `num_consumers` threads, one for each consumer. The
    nth element of the dataset is produced by consumer `n % num_consumers`.

    The dataset executed on each worker will produce groups of `num_consumers`
    sequentially increasing numbers. For example, if `num_consumers=3` a worker
    dataset could produce [0, 1, 2, 9, 10, 11, 21, 22, 23]. This enables
    `checkCoordinatedReadGroups` below to assess whether the values received in
    each step came from the same group.

    Args:
      cluster: A tf.data service `TestCluster`.
      num_consumers: The number of consumers to simulate.
      sharding_policy: The sharding policy to use. Currently only OFF and
        DYNAMIC are supported.

    Returns:
      A dataset that simulates reading with `num_consumers` consumers.
    """
    if sharding_policy not in [
        data_service_ops.ShardingPolicy.OFF,
        data_service_ops.ShardingPolicy.DYNAMIC
    ]:
      raise ValueError(f"Unsupported sharding policy: {sharding_policy}")
    # Start from 0 so that we can detect when a new worker is added with
    # ShardingPolicy.OFF.
    ds = dataset_ops.Dataset.from_tensors(math_ops.cast(0, dtypes.int64))
    ds = ds.concatenate(dataset_ops.Dataset.random())
    # Ensure that all elements in the same group are consecutive.
    def make_group(x):
      # Avoid overflowing an int64 in (x+1)*num_consumers below.
      x = x % (2**32)
      return dataset_ops.Dataset.range(x*num_consumers, (x+1)*num_consumers)
    ds = ds.flat_map(make_group)
    consumers = []
    for consumer_index in range(num_consumers):
      consumers.append(
          self.make_distributed_dataset(
              ds,
              cluster,
              job_name="test",
              processing_mode=sharding_policy,
              consumer_index=consumer_index,
              num_consumers=num_consumers))
    # Use parallel interleave to read from consumers in parallel.
    ds = dataset_ops.Dataset.from_tensor_slices(consumers)
    ds = ds.interleave(
        lambda x: x,
        cycle_length=num_consumers,
        num_parallel_calls=num_consumers)
    return ds

  def checkCoordinatedReadGroups(self, results, num_consumers):
    """Validates results from a `make_coordinted_read_dataset` dataset.

    Each group of `num_consumers` results should be consecutive, indicating that
    they were produced by the same worker.

    Args:
      results: The elements produced by the dataset.
      num_consumers: The number of consumers.
    """
    groups = [
        results[start:start + num_consumers]
        for start in range(0, len(results), num_consumers)
    ]
    incorrect_groups = []
    for group in groups:
      # Check that each group of `num_consumers` results are consecutive.
      for offset in range(1, len(group)):
        if group[0] + offset != group[offset]:
          incorrect_groups.append(group)
          break
    self.assertEmpty(
        incorrect_groups,
        "Incorrect groups: {}.\nAll groups: {}".format(incorrect_groups,
                                                       groups))

  def read(self, get_next, results, count):
    for _ in range(count):
      results.append(self.evaluate(get_next()))
