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

from tensorflow.python.data.experimental.service import server_lib

from tensorflow.python.platform import test


class ServerLibTest(test.TestCase):

  def testStartDispatcher(self):
    dispatcher = server_lib.DispatchServer(0, start=False)
    dispatcher.start()

  def testMultipleStartDispatcher(self):
    dispatcher = server_lib.DispatchServer(0, start=True)
    dispatcher.start()

  def testStartWorker(self):
    dispatcher = server_lib.DispatchServer(0)
    worker = server_lib.WorkerServer(0, dispatcher._address, start=False)
    worker.start()

  def testMultipleStartWorker(self):
    dispatcher = server_lib.DispatchServer(0)
    worker = server_lib.WorkerServer(0, dispatcher._address, start=True)
    worker.start()

  def testStopDispatcher(self):
    dispatcher = server_lib.DispatchServer(0)
    dispatcher._stop()
    dispatcher._stop()

  def testStopWorker(self):
    dispatcher = server_lib.DispatchServer(0)
    worker = server_lib.WorkerServer(0, dispatcher._address)
    worker._stop()
    worker._stop()

  def testStopStartDispatcher(self):
    dispatcher = server_lib.DispatchServer(0)
    dispatcher._stop()
    with self.assertRaisesRegex(
        RuntimeError, "Server cannot be started after it has been stopped"):
      dispatcher.start()

  def testStopStartWorker(self):
    dispatcher = server_lib.DispatchServer(0)
    worker = server_lib.WorkerServer(0, dispatcher._address)
    worker._stop()
    with self.assertRaisesRegex(
        RuntimeError, "Server cannot be started after it has been stopped"):
      worker.start()

  def testJoinDispatcher(self):
    dispatcher = server_lib.DispatchServer(0)
    dispatcher._stop()
    dispatcher.join()

  def testJoinWorker(self):
    dispatcher = server_lib.DispatchServer(0)
    worker = server_lib.WorkerServer(0, dispatcher._address)
    worker._stop()
    worker.join()

  def testDispatcherNumWorkers(self):
    dispatcher = server_lib.DispatchServer(0)
    self.assertEqual(0, dispatcher._num_workers())
    worker1 = server_lib.WorkerServer(0, dispatcher._address)  # pylint: disable=unused-variable
    self.assertEqual(1, dispatcher._num_workers())
    worker2 = server_lib.WorkerServer(0, dispatcher._address)  # pylint: disable=unused-variable
    self.assertEqual(2, dispatcher._num_workers())


if __name__ == "__main__":
  test.main()
