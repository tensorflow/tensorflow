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

import os

from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations

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
      data_service_ops._distribute(  # pylint: disable=protected-access
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


class TestBase(test_base.DatasetTestBase):
  """Base class for tf.data service tests."""

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
            dispatcher_address=self.dispatcher_address(dispatcher),
            port=port,
            heartbeat_interval_ms=200))

  # pylint: disable=protected-access
  def restart_dispatcher(self, dispatcher):
    """Stops `dispatcher` and returns a new dispatcher with the same port."""
    port = int(self.dispatcher_address(dispatcher).split(":")[1])
    dispatcher._stop()
    return self.start_dispatch_server(
        port=port,
        work_dir=dispatcher._config.work_dir,
        fault_tolerant_mode=dispatcher._config.fault_tolerant_mode)

  # pylint: disable=protected-access
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

  def dispatcher_address(self, dispatcher):
    # Targets are in the format <protocol>://<address>
    return dispatcher.target.split("://")[1]

  def make_distributed_dataset(self,
                               dataset,
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

  def make_distributed_range_dataset(self,
                                     num_elements,
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
    return self.make_distributed_dataset(dataset, dispatcher, job_name,
                                         max_outstanding_requests)
