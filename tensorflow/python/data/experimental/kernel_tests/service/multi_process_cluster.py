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
"""tf.data service test-cluster with local and remote workers."""

import tempfile

from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.platform import googletest

_WORKER_SHUTDOWN_QUIET_PERIOD_MS = 100


# pylint: disable=protected-access
class _RemoteWorkerProcess(multi_process_lib.Process):
  """Runs a worker server in a new process to simulate a remote worker."""

  def __init__(self, dispatcher_address, pipe_writer):
    super(_RemoteWorkerProcess, self).__init__()
    self._dispatcher_address = dispatcher_address
    self._pipe_writer = pipe_writer

  # `run` is hidden in multi_process_lib.py. It is assigned with a decorated
  # `run` that runs the process in `absl.app.run`.
  def run(self):  # pylint: disable=method-hidden
    self.start_worker()

  def start_worker(self):
    self._worker = data_service_test_base.TestWorker(
        self._dispatcher_address, _WORKER_SHUTDOWN_QUIET_PERIOD_MS)
    self._worker.start()
    self._pipe_writer.send(self._worker.worker_address())
    self._worker.join()


class MultiProcessCluster(object):
  """tf.data service cluster with local and remote workers.

  Represents a cluster with a dispatcher, `num_local_workers` local workers, and
  `num_remote_workers` remote workers. Remote workers run in separate processes.
  This is useful to test reading from local in-process workers. For example:

  ```
  cluster = multi_process_cluster.MultiProcessCluster(
      num_local_workers=1, num_remote_workers=3)
  num_elements = 10
  dataset = self.make_distributed_range_dataset(
      num_elements, cluster, target_workers="LOCAL")
  self.assertDatasetProduces(dataset, list(range(num_elements)))
  ```
  """

  def __init__(self, num_local_workers, num_remote_workers, dispatcher_port=0):
    self._start_dispatcher(dispatcher_port)
    self._start_local_workers(num_local_workers)
    self._start_remote_workers(num_remote_workers)

  def _start_dispatcher(self, dispatcher_port):
    work_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    self._dispatcher = server_lib.DispatchServer(
        server_lib.DispatcherConfig(
            port=dispatcher_port, work_dir=work_dir, protocol="grpc"),
        start=True)

  def _start_local_workers(self, num_workers):
    self._local_workers = []
    for _ in range(num_workers):
      worker = data_service_test_base.TestWorker(
          self.dispatcher_address(), _WORKER_SHUTDOWN_QUIET_PERIOD_MS)
      worker.start()
      self._local_workers.append(worker)

  def _start_remote_workers(self, num_workers):
    # List of (worker address, remote worker process) tuples.
    self._remote_workers = []
    for _ in range(num_workers):
      self._start_remote_worker()

  def _start_remote_worker(self):
    pipe_reader, pipe_writer = multi_process_lib.multiprocessing.Pipe(
        duplex=False)
    worker_process = _RemoteWorkerProcess(
        self.dispatcher_address(), pipe_writer)
    worker_process.start()
    worker_address = pipe_reader.recv()
    self._remote_workers.append((worker_address, worker_process))

  def dispatcher_address(self):
    return self._dispatcher._address

  def local_worker_addresses(self):
    return [worker.worker_address() for worker in self._local_workers]

  def remote_worker_addresses(self):
    return [worker[0] for worker in self._remote_workers]

  def _stop(self):
    for worker in self._local_workers:
      worker.stop()
    for worker in self._remote_workers:
      worker[1].terminate()
    self._dispatcher._stop()

  def __del__(self):
    self._stop()


def test_main():
  """Main function to be called within `__main__` of a test file."""
  multi_process_lib.test_main()
