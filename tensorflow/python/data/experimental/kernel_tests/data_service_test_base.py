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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile

from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.platform import googletest

# This will be resolved to a tmp directory by `start_dispatch_server`.
TMP_WORK_DIR = "tmp_work_dir_placeholder"
# `""` indicates not to use a work directory.
NO_WORK_DIR = ""
# We use a faster than normal heartbeat interval so that tests run faster.
TEST_HEARTBEAT_INTERVAL_MS = 100


def all_cluster_configurations():
  with_work_dir = combinations.combine(
      work_dir=TMP_WORK_DIR, fault_tolerant_mode=[True, False])
  without_work_dir = combinations.combine(
      work_dir=NO_WORK_DIR, fault_tolerant_mode=False)
  return with_work_dir + without_work_dir


class TestCluster(object):
  """Test tf.data service cluster."""

  def __init__(self,
               num_workers,
               dispatcher_port=0,
               work_dir=TMP_WORK_DIR,
               fault_tolerant_mode=True,
               job_gc_check_interval_ms=None,
               job_gc_timeout_ms=None,
               start=True):
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
      start: Whether to immediately start the servers in the cluster. If
        `False`, the servers can be started later by calling
        `start_dispatcher()` and `start_workers()`.
    """
    if work_dir == TMP_WORK_DIR:
      work_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    self.dispatcher = server_lib.DispatchServer(
        server_lib.DispatcherConfig(
            port=dispatcher_port,
            work_dir=work_dir,
            fault_tolerant_mode=fault_tolerant_mode,
            job_gc_check_interval_ms=job_gc_check_interval_ms,
            job_gc_timeout_ms=job_gc_timeout_ms),
        start=start)

    self.workers = []
    for _ in range(num_workers):
      self.add_worker(start=start)

  @property
  def target(self):
    return self.dispatcher.target

  def dispatcher_address(self):
    return self.dispatcher.target.split("://")[1]

  def add_worker(self, start=True):
    self.workers.append(
        server_lib.WorkerServer(
            server_lib.WorkerConfig(
                dispatcher_address=self.dispatcher_address(),
                heartbeat_interval_ms=TEST_HEARTBEAT_INTERVAL_MS,
                dispatcher_timeout_ms=1000),
            start=start))

  def start_dispatcher(self):
    self.dispatcher.start()

  def start_workers(self):
    for worker in self.workers:
      worker.start()

  def stop_dispatcher(self):
    # pylint: disable=protected-access
    self.dispatcher._stop()

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
            fault_tolerant_mode=self.dispatcher._config.fault_tolerant_mode))

  # pylint: disable=protected-access
  def restart_worker(self, worker_index=0, use_same_port=True):
    """Replaces the worker at index `worker_index` with a new worker."""
    worker = self.workers[worker_index]
    port = 0
    if use_same_port:
      port = int(worker._address.split(":")[1])
    worker._stop()
    self.workers[worker_index] = server_lib.WorkerServer(
        server_lib.WorkerConfig(
            dispatcher_address=self.dispatcher_address(),
            port=port,
            heartbeat_interval_ms=worker._config.heartbeat_interval_ms))

  def num_registered_workers(self):
    return self.dispatcher._num_workers()

  def num_tasks_on_worker(self, worker_index=0):
    return self.workers[worker_index]._num_tasks()


class TestBase(test_base.DatasetTestBase):
  """Base class for tf.data service tests."""

  def create_cluster(self,
                     num_workers,
                     dispatcher_port=0,
                     work_dir=TMP_WORK_DIR,
                     fault_tolerant_mode=True,
                     job_gc_check_interval_ms=None,
                     job_gc_timeout_ms=None,
                     start=True):
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
      start: Whether to immediately start the servers in the cluster. If
        `False`, the servers can be started later by calling
        `start_dispatcher()` and `start_workers()`.

    Returns:
      The created cluster.
    """
    return TestCluster(
        num_workers=num_workers,
        dispatcher_port=dispatcher_port,
        work_dir=work_dir,
        fault_tolerant_mode=fault_tolerant_mode,
        job_gc_check_interval_ms=job_gc_check_interval_ms,
        job_gc_timeout_ms=job_gc_timeout_ms,
        start=start)

  def make_distributed_dataset(self,
                               dataset,
                               cluster,
                               processing_mode="parallel_epochs",
                               job_name=None,
                               max_outstanding_requests=None):
    # pylint: disable=protected-access
    return dataset.apply(
        data_service_ops._distribute(
            processing_mode,
            cluster.target,
            job_name=job_name,
            max_outstanding_requests=max_outstanding_requests,
            task_refresh_interval_hint_ms=20))

  def make_distributed_range_dataset(self,
                                     num_elements,
                                     cluster,
                                     job_name=None,
                                     max_outstanding_requests=None):
    dataset = dataset_ops.Dataset.range(num_elements)
    return self.make_distributed_dataset(
        dataset,
        cluster,
        job_name=job_name,
        max_outstanding_requests=max_outstanding_requests)
