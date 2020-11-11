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

import collections

# pylint: disable=invalid-import-order,g-bad-import-order, unused-import
from tensorflow.core.protobuf.data.experimental import service_config_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.data.experimental.service import _pywrap_server_lib
from tensorflow.python.util.tf_export import tf_export


@tf_export("data.experimental.service.DispatcherConfig")
class DispatcherConfig(
    collections.namedtuple("DispatcherConfig", [
        "port", "protocol", "work_dir", "fault_tolerant_mode",
        "job_gc_check_interval_ms", "job_gc_timeout_ms"
    ])):
  """Configuration class for tf.data service dispatchers.

  Fields:
    port: Specifies the port to bind to. A value of 0 indicates that the server
      may bind to any available port.
    protocol: The protocol to use for communicating with the tf.data service.
      Defaults to `"grpc"`.
    work_dir: A directory to store dispatcher state in. This
      argument is required for the dispatcher to be able to recover from
      restarts.
    fault_tolerant_mode: Whether the dispatcher should write its state to a
      journal so that it can recover from restarts. Dispatcher state, including
      registered datasets and created jobs, is synchronously written to the
      journal before responding to RPCs. If `True`, `work_dir` must also be
      specified.
    job_gc_check_interval_ms: How often the dispatcher should scan through to
      delete old and unused jobs, in milliseconds. If not set, the runtime will
      select a reasonable default. A higher value will reduce load on the
      dispatcher, while a lower value will reduce the time it takes for the
      dispatcher to garbage collect expired jobs.
    job_gc_timeout_ms: How long a job needs to be unused before it becomes a
      candidate for garbage collection, in milliseconds. If not set, the runtime
      will select a reasonable default. A higher value will cause jobs to stay
      around longer with no consumers. This is useful if there is a large gap in
      time between when consumers read from the job. A lower value will reduce
      the time it takes to reclaim the resources from expired jobs.
  """

  def __new__(cls,
              port=0,
              protocol="grpc",
              work_dir=None,
              fault_tolerant_mode=False,
              job_gc_check_interval_ms=None,
              job_gc_timeout_ms=None):
    if job_gc_check_interval_ms is None:
      job_gc_check_interval_ms = 10 * 60 * 1000  # 10 minutes.
    if job_gc_timeout_ms is None:
      job_gc_timeout_ms = 5 * 60 * 1000  # 5 minutes.
    return super(DispatcherConfig,
                 cls).__new__(cls, port, protocol, work_dir,
                              fault_tolerant_mode, job_gc_check_interval_ms,
                              job_gc_timeout_ms)


@tf_export("data.experimental.service.DispatchServer", v1=[])
class DispatchServer(object):
  """An in-process tf.data service dispatch server.

  A `tf.data.experimental.service.DispatchServer` coordinates a cluster of
  `tf.data.experimental.service.WorkerServer`s. When the workers start, they
  register themselves with the dispatcher.

  >>> dispatcher = tf.data.experimental.service.DispatchServer()
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> worker = tf.data.experimental.service.WorkerServer(WorkerConfig(
  ...     dispatcher_address=dispatcher_address))
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
  ...     processing_mode="parallel_epochs", service=dispatcher.target))
  >>> print(list(dataset.as_numpy_iterator()))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  When starting a dedicated tf.data dispatch process, use join() to block
  indefinitely after starting up the server.

  ```
  dispatcher = tf.data.experimental.service.DispatchServer(
      tf.data.experimental.service.DispatcherConfig(port=5050))
  dispatcher.join()
  ```

  To start a `DispatchServer` in fault-tolerant mode, set `work_dir` and
  `fault_tolerant_mode` like below:

  ```
  dispatcher = tf.data.experimental.service.DispatchServer(
      tf.data.experimental.service.DispatcherConfig(
          port=5050,
          work_dir="gs://my-bucket/dispatcher/work_dir",
          fault_tolerant_mode=True))
  ```
  """

  def __init__(self, config=None, start=True):
    """Creates a new dispatch server.

    Args:
      config: (Optional.) A `tf.data.experimental.service.DispatcherConfig`
        configration. If `None`, the dispatcher will use default
        configuration values.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to True.
    """
    config = config or DispatcherConfig()
    if config.fault_tolerant_mode and not config.work_dir:
      raise ValueError(
          "Cannot enable fault tolerant mode without configuring a work_dir")
    self._config = config
    config_proto = service_config_pb2.DispatcherConfig(
        port=config.port,
        protocol=config.protocol,
        work_dir=config.work_dir,
        fault_tolerant_mode=config.fault_tolerant_mode,
        job_gc_check_interval_ms=config.job_gc_check_interval_ms,
        job_gc_timeout_ms=config.job_gc_timeout_ms)
    self._server = _pywrap_server_lib.TF_DATA_NewDispatchServer(
        config_proto.SerializeToString())
    if start:
      self._server.start()

  def start(self):
    """Starts this server.

    >>> dispatcher = tf.data.experimental.service.DispatchServer(start=False)
    >>> dispatcher.start()

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        starting the server.
    """
    self._server.start()

  def join(self):
    """Blocks until the server has shut down.

    This is useful when starting a dedicated dispatch process.

    ```
    dispatcher = tf.data.experimental.service.DispatchServer(
        tf.data.experimental.service.DispatcherConfig(port=5050))
    dispatcher.join()
    ```

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        joining the server.
    """
    self._server.join()

  @property
  def target(self):
    """Returns a target that can be used to connect to the server.

    >>> dispatcher = tf.data.experimental.service.DispatchServer()
    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
    ...     processing_mode="parallel_epochs", service=dispatcher.target))

    The returned string will be in the form protocol://address, e.g.
    "grpc://localhost:5050".
    """
    return "{0}://localhost:{1}".format(self._config.protocol,
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
    """Returns the number of workers registered with the dispatcher."""
    return self._server.num_workers()


@tf_export("data.experimental.service.WorkerConfig")
class WorkerConfig(
    collections.namedtuple("WorkerConfig", [
        "dispatcher_address", "worker_address", "port", "protocol",
        "heartbeat_interval_ms", "dispatcher_timeout_ms"
    ])):
  """Configuration class for tf.data service dispatchers.

  Fields:
    dispatcher_address: Specifies the address of the dispatcher.
    worker_address: Specifies the address of the worker server. This address is
      passed to the dispatcher so that the dispatcher can tell clients how to
      connect to this worker.
    port: Specifies the port to bind to. A value of 0 indicates that the worker
      can bind to any available port.
    protocol: (Optional.) Specifies the protocol to be used by the server.
      Defaults to `"grpc"`.
    heartbeat_interval_ms: How often the worker should heartbeat to the
      dispatcher, in milliseconds. If not set, the runtime will select a
      reasonable default. A higher value will reduce the load on the dispatcher,
      while a lower value will reduce the time it takes to reclaim resources
      from finished jobs.
    dispatcher_timeout_ms: How long, in milliseconds, to retry requests to the
      dispatcher before giving up and reporting an error. Defaults to 1 hour.
  """

  def __new__(cls,
              dispatcher_address,
              worker_address=None,
              port=0,
              protocol="grpc",
              heartbeat_interval_ms=None,
              dispatcher_timeout_ms=None):
    if worker_address is None:
      worker_address = "localhost:%port%"
    if heartbeat_interval_ms is None:
      heartbeat_interval_ms = 30 * 1000  # 30 seconds
    if dispatcher_timeout_ms is None:
      dispatcher_timeout_ms = 60 * 60 * 1000  # 1 hour

    return super(WorkerConfig,
                 cls).__new__(cls, dispatcher_address, worker_address, port,
                              protocol, heartbeat_interval_ms,
                              dispatcher_timeout_ms)


@tf_export("data.experimental.service.WorkerServer", v1=[])
class WorkerServer(object):
  """An in-process tf.data service worker server.

  A `tf.data.experimental.service.WorkerServer` performs `tf.data.Dataset`
  processing for user-defined datasets, and provides the resulting elements over
  RPC. A worker is associated with a single
  `tf.data.experimental.service.DispatchServer`.

  >>> dispatcher = tf.data.experimental.service.DispatchServer()
  >>> dispatcher_address = dispatcher.target.split("://")[1]
  >>> worker = tf.data.experimental.service.WorkerServer(
  ...     tf.data.experimental.service.WorkerConfig(
  ...         dispatcher_address=dispatcher_address))
  >>> dataset = tf.data.Dataset.range(10)
  >>> dataset = dataset.apply(tf.data.experimental.service.distribute(
  ...     processing_mode="parallel_epochs", service=dispatcher.target))
  >>> print(list(dataset.as_numpy_iterator()))
  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

  When starting a dedicated tf.data worker process, use join() to block
  indefinitely after starting up the server.

  ```
  worker = tf.data.experimental.service.WorkerServer(
      port=5051, dispatcher_address="grpc://localhost:5050")
  worker.join()
  ```
  """

  def __init__(self, config, start=True):
    """Creates a new worker server.

    Args:
      config: A `tf.data.experimental.service.WorkerConfig` configration.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to True.
    """
    if config.dispatcher_address is None:
      raise ValueError("must specify a dispatcher_address")
    self._config = config
    config_proto = service_config_pb2.WorkerConfig(
        dispatcher_address=config.dispatcher_address,
        worker_address=config.worker_address,
        port=config.port,
        protocol=config.protocol,
        heartbeat_interval_ms=config.heartbeat_interval_ms,
        dispatcher_timeout_ms=config.dispatcher_timeout_ms)
    self._server = _pywrap_server_lib.TF_DATA_NewWorkerServer(
        config_proto.SerializeToString())
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
        port=5051, dispatcher_address="grpc://localhost:5050")
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

  def _num_tasks(self):
    """Returns the number of tasks currently being executed on the worker."""
    return self._server.num_tasks()
