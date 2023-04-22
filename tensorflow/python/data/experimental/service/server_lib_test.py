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
"""Tests for tf.data service server lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import tempfile
import threading
import unittest
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.framework import errors
from tensorflow.python.platform import test
from tensorflow.python.profiler import profiler_client

_portpicker_import_error = None
try:
  import portpicker  # pylint: disable=g-import-not-at-top
except ImportError as _error:  # pylint: disable=invalid-name
  _portpicker_import_error = _error
  portpicker = None

ASSIGNED_PORTS = set()
lock = threading.Lock()


def pick_unused_port():
  """Returns an unused and unassigned local port."""

  if _portpicker_import_error:
    raise _portpicker_import_error  # pylint: disable=raising-bad-type

  global ASSIGNED_PORTS
  with lock:
    while True:
      try:
        port = portpicker.pick_unused_port()
      except portpicker.NoFreePortFoundError:
        raise unittest.SkipTest("Flakes in portpicker library do not represent "
                                "TensorFlow errors.")
      if port > 10000 and port not in ASSIGNED_PORTS:
        ASSIGNED_PORTS.add(port)
        logging.info("Using local port %r", port)
        return port


class ServerLibTest(test.TestCase):

  def testStartDispatcher(self):
    dispatcher = server_lib.DispatchServer(start=False)
    dispatcher.start()

  def testStartDispatcherWithPortConfig(self):
    port = pick_unused_port()
    config = server_lib.DispatcherConfig(port=port)
    dispatcher = server_lib.DispatchServer(config=config, start=True)
    self.assertEqual(dispatcher.target, "grpc://localhost:{}".format(port))

  def testStartDispatcherWithWorkDirConfig(self):
    temp_dir = tempfile.mkdtemp()
    config = server_lib.DispatcherConfig(work_dir=temp_dir)
    dispatcher = server_lib.DispatchServer(  # pylint: disable=unused-variable
        config=config, start=True)

  def testStartDispatcherWithFaultTolerantConfig(self):
    temp_dir = tempfile.mkdtemp()
    config = server_lib.DispatcherConfig(
        work_dir=temp_dir, fault_tolerant_mode=True)
    dispatcher = server_lib.DispatchServer(  # pylint: disable=unused-variable
        config=config, start=True)

  def testStartDispatcherWithWrongFaultTolerantConfig(self):
    config = server_lib.DispatcherConfig(fault_tolerant_mode=True)
    error = "Cannot enable fault tolerant mode without configuring a work_dir"
    with self.assertRaisesRegex(ValueError, error):
      dispatcher = server_lib.DispatchServer(  # pylint: disable=unused-variable
          config=config, start=True)

  def testMultipleStartDispatcher(self):
    dispatcher = server_lib.DispatchServer(start=True)
    dispatcher.start()

  def testStartWorker(self):
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address), start=False)
    worker.start()

  def testStartWorkerWithPortConfig(self):
    dispatcher = server_lib.DispatchServer()
    port = pick_unused_port()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address, port=port), start=True)
    self.assertEqual(worker._address, "localhost:{}".format(port))

  def testMultipleStartWorker(self):
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address), start=True)
    worker.start()

  def testStopDispatcher(self):
    dispatcher = server_lib.DispatchServer()
    dispatcher._stop()
    dispatcher._stop()

  def testStopWorker(self):
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address))
    worker._stop()
    worker._stop()

  def testStopStartDispatcher(self):
    dispatcher = server_lib.DispatchServer()
    dispatcher._stop()
    with self.assertRaisesRegex(
        RuntimeError, "Server cannot be started after it has been stopped"):
      dispatcher.start()

  def testStopStartWorker(self):
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address))
    worker._stop()
    with self.assertRaisesRegex(
        RuntimeError, "Server cannot be started after it has been stopped"):
      worker.start()

  def testJoinDispatcher(self):
    dispatcher = server_lib.DispatchServer()
    dispatcher._stop()
    dispatcher.join()

  def testJoinWorker(self):
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address))
    worker._stop()
    worker.join()

  def testDispatcherNumWorkers(self):
    dispatcher = server_lib.DispatchServer()
    self.assertEqual(0, dispatcher._num_workers())
    worker1 = server_lib.WorkerServer(  # pylint: disable=unused-variable
        server_lib.WorkerConfig(dispatcher._address))
    self.assertEqual(1, dispatcher._num_workers())
    worker2 = server_lib.WorkerServer(  # pylint: disable=unused-variable
        server_lib.WorkerConfig(dispatcher._address))
    self.assertEqual(2, dispatcher._num_workers())

  def testProfileWorker(self):
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address))
    # Test the profilers are successfully started and connected to profiler
    # service on the worker. Since there is no op running, it is expected to
    # return UnavailableError with no trace events collected string.
    with self.assertRaises(errors.UnavailableError) as error:
      profiler_client.trace(worker._address, tempfile.mkdtemp(), duration_ms=10)
    self.assertStartsWith(str(error.exception), "No trace event was collected")


if __name__ == "__main__":
  test.main()
