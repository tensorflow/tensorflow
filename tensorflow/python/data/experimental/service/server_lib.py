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
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.service.MasterServer", v1=[])
class MasterServer(object):
  """An in-process tf.data service master server.

  A `tf.data.experimental.service.MasterServer` coordinates a cluster of
  `tf.data.experimental.service.WorkerServer`s. When the workers start, they
  register themselves with the master.

  >>> master = tf.data.experimental.service.MasterServer(port=0)
  >>> master_address = master.target.split("://")[1]
  >>> worker = tf.data.experimental.service.WorkerServer(
  ...     port=0, master_address=master_address)
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
  ...     processing_mode="parallel_epochs", service=master.target))
  >>> print(list(dataset.as_numpy_iterator()))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  When starting a dedicated tf.data master process, use join() to block
  indefinitely after starting up the server.

  ```
  master = tf.data.experimental.service.MasterServer(port=5050)
  master.join()
  ```
  """

  def __init__(self, port, protocol=None, start=True):
    """Creates a new master server.

    Args:
      port: Specifies the port to bind to.
      protocol: (Optional.) Specifies the protocol to be used by the server.
        Acceptable values include `"grpc", "grpc+local"`. Defaults to `"grpc"`.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        creating the TensorFlow server.
    """
    if protocol is None:
      protocol = "grpc"
    self._protocol = protocol
    self._server = _pywrap_server_lib.TF_DATA_NewMasterServer(port, protocol)
    if start:
      self._server.start()

  def start(self):
    """Starts this server.

    >>> master = tf.data.experimental.service.MasterServer(port=0, start=False)
    >>> master.start()

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        starting the server.
    """
    self._server.start()

  def join(self):
    """Blocks until the server has shut down.

    This is useful when starting a dedicated master process.

    ```
    master = tf.data.experimental.service.MasterServer(port=5050)
    master.join()
    ```

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        joining the server.
    """
    self._server.join()

  @property
  def target(self):
    """Returns a target that can be used to connect to the server.

    >>> master = tf.data.experimental.service.MasterServer(port=0)
    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
    ...     processing_mode="parallel_epochs", service=master.target))

    The returned string will be in the form protocol://address, e.g.
    "grpc://localhost:5050".
    """
    return "{0}://localhost:{1}".format(self._protocol,
                                        self._server.bound_port())

  def _stop(self):
    """Stops the server.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        stopping the server.
    """
    self._server.stop()

  def __del__(self):
    self._stop()

  @property
  def _address(self):
    """Returns the address of the server.

    The returned string will be in the form address:port, e.g. "localhost:1000".
    """
    return "localhost:{0}".format(self._server.bound_port())

  def _num_workers(self):
    """Returns the number of workers registered with the master."""
    return self._server.num_workers()


@tf_export("data.experimental.service.WorkerServer", v1=[])
class WorkerServer(object):
  """An in-process tf.data service worker server.

  A `tf.data.experimental.service.WorkerServer` performs `tf.data.Dataset`
  processing for user-defined datasets, and provides the resulting elements over
  RPC. A worker is associated with a single
  `tf.data.experimental.service.MasterServer`.

  >>> master = tf.data.experimental.service.MasterServer(port=0)
  >>> master_address = master.target.split("://")[1]
  >>> worker = tf.data.experimental.service.WorkerServer(
  ...     port=0, master_address=master_address)
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
  ...     processing_mode="parallel_epochs", service=master.target))
  >>> print(list(dataset.as_numpy_iterator()))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  When starting a dedicated tf.data worker process, use join() to block
  indefinitely after starting up the server.

  ```
  worker = tf.data.experimental.service.WorkerServer(
      port=5051, master_address="grpc://localhost:5050")
  worker.join()
  ```
  """

  def __init__(self,
               port,
               master_address,
               worker_address=None,
               protocol=None,
               start=True):
    """Creates a new worker server.

    Args:
      port: Specifies the port to bind to. A value of 0 indicates that the
        worker can bind to any available port.
      master_address: Specifies the address of the master server.
      worker_address: (Optional.) Specifies the address of the worker server.
        This address is passed to the master server so that the master can tell
        clients how to connect to this worker. Defaults to `"localhost:%port%"`,
          where `%port%` will be replaced with the port used by the worker.
      protocol: (Optional.) Specifies the protocol to be used by the server.
        Acceptable values include `"grpc", "grpc+local"`. Defaults to `"grpc"`.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        creating the TensorFlow server.
    """
    if worker_address is None:
      worker_address = "localhost:%port%"
    if protocol is None:
      protocol = "grpc"

    self._protocol = protocol
    self._server = _pywrap_server_lib.TF_DATA_NewWorkerServer(
        port, protocol, master_address, worker_address)
    if start:
      self._server.start()

  def start(self):
    """Starts this server.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        starting the server.
    """
    self._server.start()

  def join(self):
    """Blocks until the server has shut down.

    This is useful when starting a dedicated worker process.

    ```
    worker_server = tf.data.experimental.service.WorkerServer(
        port=5051, master_address="grpc://localhost:5050")
    worker_server.join()
    ```

    This method currently blocks forever.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        joining the server.
    """
    self._server.join()

  def _stop(self):
    """Stops the server.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        stopping the server.
    """
    self._server.stop()

  def __del__(self):
    self._stop()

  @property
  def _address(self):
    """Returns the address of the server.

    The returned string will be in the form address:port, e.g. "localhost:1000".
    """
    return "localhost:{0}".format(self._server.bound_port())
