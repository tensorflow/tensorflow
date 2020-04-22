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
"""A Python interface for creating dataset servers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-import-order,g-bad-import-order, unused-import
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.service import _pywrap_server_lib


class MasterServer(object):
  """An in-process tf.data service master, for use in testing."""

  def __init__(self, protocol):
    """Creates and starts a new tf.data master server.

    The server will choose an available port. Use `target()` to get the string
    for connecting to the server.

    Args:
      protocol: A string representing the type of protocol to use when creating
        channels. For no security, use "grpc". For local credentials, use
        "grpc+local", and make sure your binary links in
        `data/service:local_credentials`.
    """
    self._server = _pywrap_server_lib.TF_DATA_NewMasterServer(0, protocol)
    self._running = True

  @property
  def target(self):
    """Returns the target for connecting to this server.

    The returned string will be in the form protocol://address:port, e.g.
    "grpc://localhost:1000".
    """
    return _pywrap_server_lib.TF_DATA_MasterServerTarget(self._server)

  def num_tasks(self):
    """Returns the number of tasks on the master."""
    return _pywrap_server_lib.TF_DATA_MasterServerNumTasks(self._server)

  def stop(self):
    """Shuts down and deletes the server.

    This method will block until all outstanding rpcs have completed and the
    server has been shut down.
    """
    if self._running:
      self._running = False
      _pywrap_server_lib.TF_DATA_DeleteMasterServer(self._server)

  def __del__(self):
    self.stop()


class WorkerServer(object):
  """An in-process tf.data service worker, for use in testing."""

  def __init__(self, protocol, master_address, port=0):
    """Creates and starts a new tf.data worker server.

    The server will choose an available port. Use `target()` to get the string
    for connecting to the server.

    Args:
      protocol: A string representing the type of protocol to use when creating
        channels. For no security, use "grpc". For local credentials, use
        "grpc+local", and make sure your binary links in
        `data/service:local_credentials`.
      master_address: The address of the tf.data master server to register with.
      port: The port to bind to.
    """
    self._server = _pywrap_server_lib.TF_DATA_NewWorkerServer(
        port, protocol, master_address)
    self._running = True

  @property
  def target(self):
    """Returns the target for connecting to this server.

    The returned string will be in the form protocol://address:port, e.g.
    "grpc://localhost:1000".
    """
    return _pywrap_server_lib.TF_DATA_WorkerServerTarget(self._server)

  def stop(self):
    """Shuts down and deletes the server.

    This method will block until all outstanding rpcs have completed and the
    server has been shut down.
    """
    if self._running:
      self._running = False
      _pywrap_server_lib.TF_DATA_DeleteWorkerServer(self._server)

  def __del__(self):
    self.stop()
