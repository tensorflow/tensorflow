# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""A Python interface for creating TensorFlow servers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import errors
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


def _make_server_def(server_or_cluster_def, job_name, task_index, protocol,
                     config):
  """Creates a `tf.train.ServerDef` protocol buffer.

  Args:
    server_or_cluster_def: A `tf.train.ServerDef` or
      `tf.train.ClusterDef` protocol buffer, or a
      `tf.train.ClusterSpec` object, describing the server to be
      defined and/or the cluster of which it is a member.
    job_name: (Optional.) Specifies the name of the job of which the server
      is a member. Defaults to the value in `server_or_cluster_def`, if
      specified.
    task_index: (Optional.) Specifies the task index of the server in its job.
      Defaults to the value in `server_or_cluster_def`, if specified. Otherwise
      defaults to 0 if the server's job has only one task.
    protocol: (Optional.) Specifies the protocol to be used by the server.
      Acceptable values include `"grpc", "grpc+verbs"`. Defaults to the value
      in `server_or_cluster_def`, if specified. Otherwise defaults to `"grpc"`.
    config: (Options.) A `tf.ConfigProto` that specifies default configuration
      options for all sessions that run on this server.

  Returns:
    A `tf.train.ServerDef`.

  Raises:
    TypeError: If the arguments do not have the appropriate type.
    ValueError: If an argument is not specified and cannot be inferred.
  """
  server_def = tensorflow_server_pb2.ServerDef()
  if isinstance(server_or_cluster_def, tensorflow_server_pb2.ServerDef):
    server_def.MergeFrom(server_or_cluster_def)
    if job_name is not None:
      server_def.job_name = job_name
    if task_index is not None:
      server_def.task_index = task_index
    if protocol is not None:
      server_def.protocol = protocol
    if config is not None:
      server_def.default_session_config.MergeFrom(config)
  else:
    try:
      cluster_spec = ClusterSpec(server_or_cluster_def)
    except TypeError:
      raise TypeError("Could not convert `server_or_cluster_def` to a "
                      "`tf.train.ServerDef` or `tf.train.ClusterSpec`.")
    if job_name is None:
      if len(cluster_spec.jobs) == 1:
        job_name = cluster_spec.jobs[0]
      else:
        raise ValueError("Must specify an explicit `job_name`.")
    if task_index is None:
      task_indices = cluster_spec.task_indices(job_name)
      if len(task_indices) == 1:
        task_index = task_indices[0]
      else:
        raise ValueError("Must specify an explicit `task_index`.")
    if protocol is None:
      protocol = "grpc"

    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_spec.as_cluster_def(),
        job_name=job_name, task_index=task_index, protocol=protocol)
    if config is not None:
      server_def.default_session_config.MergeFrom(config)
  return server_def


@tf_export("train.Server")
class Server(object):
  """An in-process TensorFlow server, for use in distributed training.

  A `tf.train.Server` instance encapsulates a set of devices and a
  @{tf.Session} target that
  can participate in distributed training. A server belongs to a
  cluster (specified by a @{tf.train.ClusterSpec}), and
  corresponds to a particular task in a named job. The server can
  communicate with any other server in the same cluster.
  """

  def __init__(self,
               server_or_cluster_def,
               job_name=None,
               task_index=None,
               protocol=None,
               config=None,
               start=True):
    """Creates a new server with the given definition.

    The `job_name`, `task_index`, and `protocol` arguments are optional, and
    override any information provided in `server_or_cluster_def`.

    Args:
      server_or_cluster_def: A `tf.train.ServerDef` or
        `tf.train.ClusterDef` protocol buffer, or a
        `tf.train.ClusterSpec` object, describing the server to be
        created and/or the cluster of which it is a member.
      job_name: (Optional.) Specifies the name of the job of which the server
        is a member. Defaults to the value in `server_or_cluster_def`, if
        specified.
      task_index: (Optional.) Specifies the task index of the server in its
        job. Defaults to the value in `server_or_cluster_def`, if specified.
        Otherwise defaults to 0 if the server's job has only one task.
      protocol: (Optional.) Specifies the protocol to be used by the server.
        Acceptable values include `"grpc", "grpc+verbs"`. Defaults to the
        value in `server_or_cluster_def`, if specified. Otherwise defaults to
        `"grpc"`.
      config: (Options.) A `tf.ConfigProto` that specifies default
        configuration options for all sessions that run on this server.
      start: (Optional.) Boolean, indicating whether to start the server
        after creating it. Defaults to `True`.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        creating the TensorFlow server.
    """
    self._server_def = _make_server_def(server_or_cluster_def,
                                        job_name, task_index, protocol, config)
    with errors.raise_exception_on_not_ok_status() as status:
      self._server = pywrap_tensorflow.PyServer_New(
          self._server_def.SerializeToString(), status)
    if start:
      self.start()

  def start(self):
    """Starts this server.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        starting the TensorFlow server.
    """
    with errors.raise_exception_on_not_ok_status() as status:
      pywrap_tensorflow.PyServer_Start(self._server, status)

  def join(self):
    """Blocks until the server has shut down.

    This method currently blocks forever.

    Raises:
      tf.errors.OpError: Or one of its subclasses if an error occurs while
        joining the TensorFlow server.
    """
    with errors.raise_exception_on_not_ok_status() as status:
      pywrap_tensorflow.PyServer_Join(self._server, status)

  @property
  def server_def(self):
    """Returns the `tf.train.ServerDef` for this server.

    Returns:
      A `tf.train.ServerDef` protocol buffer that describes the configuration
      of this server.
    """
    return self._server_def

  @property
  def target(self):
    """Returns the target for a `tf.Session` to connect to this server.

    To create a
    @{tf.Session} that
    connects to this server, use the following snippet:

    ```python
    server = tf.train.Server(...)
    with tf.Session(server.target):
      # ...
    ```

    Returns:
      A string containing a session target for this server.
    """
    return self._server.target()

  @staticmethod
  def create_local_server(config=None, start=True):
    """Creates a new single-process cluster running on the local host.

    This method is a convenience wrapper for creating a
    `tf.train.Server` with a `tf.train.ServerDef` that specifies a
    single-process cluster containing a single task in a job called
    `"local"`.

    Args:
      config: (Options.) A `tf.ConfigProto` that specifies default
        configuration options for all sessions that run on this server.
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.

    Returns:
      A local `tf.train.Server`.
    """
    # Specifying port 0 means that the OS will choose a free port for the
    # server.
    return Server({"local": ["localhost:0"]}, protocol="grpc", config=config,
                  start=start)


@tf_export("train.ClusterSpec")
class ClusterSpec(object):
  """Represents a cluster as a set of "tasks", organized into "jobs".

  A `tf.train.ClusterSpec` represents the set of processes that
  participate in a distributed TensorFlow computation. Every
  @{tf.train.Server} is constructed in a particular cluster.

  To create a cluster with two jobs and five tasks, you specify the
  mapping from job names to lists of network addresses (typically
  hostname-port pairs).

  ```python
  cluster = tf.train.ClusterSpec({"worker": ["worker0.example.com:2222",
                                             "worker1.example.com:2222",
                                             "worker2.example.com:2222"],
                                  "ps": ["ps0.example.com:2222",
                                         "ps1.example.com:2222"]})
  ```

  Each job may also be specified as a sparse mapping from task indices
  to network addresses. This enables a server to be configured without
  needing to know the identity of (for example) all other worker
  tasks:

  ```python
  cluster = tf.train.ClusterSpec({"worker": {1: "worker1.example.com:2222"},
                                  "ps": ["ps0.example.com:2222",
                                         "ps1.example.com:2222"]})
  ```
  """

  def __init__(self, cluster):
    """Creates a `ClusterSpec`.

    Args:
      cluster: A dictionary mapping one or more job names to (i) a
        list of network addresses, or (ii) a dictionary mapping integer
        task indices to network addresses; or a `tf.train.ClusterDef`
        protocol buffer.

    Raises:
      TypeError: If `cluster` is not a dictionary mapping strings to lists
        of strings, and not a `tf.train.ClusterDef` protobuf.
    """
    if isinstance(cluster, dict):
      self._cluster_spec = {}
      for job_name, tasks in cluster.items():
        if isinstance(tasks, (list, tuple)):
          job_tasks = {i: task for i, task in enumerate(tasks)}
        elif isinstance(tasks, dict):
          job_tasks = {i: task for i, task in tasks.items()}
        else:
          raise TypeError("The tasks for job %r must be a list or a dictionary "
                          "from integers to strings." % job_name)
        self._cluster_spec[job_name] = job_tasks
      self._make_cluster_def()
    elif isinstance(cluster, cluster_pb2.ClusterDef):
      self._cluster_def = cluster
      self._cluster_spec = {}
      for job_def in self._cluster_def.job:
        self._cluster_spec[job_def.name] = {
            i: t for i, t in job_def.tasks.items()}
    elif isinstance(cluster, ClusterSpec):
      self._cluster_def = cluster_pb2.ClusterDef()
      self._cluster_def.MergeFrom(cluster.as_cluster_def())
      self._cluster_spec = {}
      for job_def in self._cluster_def.job:
        self._cluster_spec[job_def.name] = {
            i: t for i, t in job_def.tasks.items()}
    else:
      raise TypeError("`cluster` must be a dictionary mapping one or more "
                      "job names to lists of network addresses, or a "
                      "`ClusterDef` protocol buffer")

  def __nonzero__(self):
    return bool(self._cluster_spec)

  # Python 3.x
  __bool__ = __nonzero__

  def __eq__(self, other):
    return self._cluster_spec == other

  def __ne__(self, other):
    return self._cluster_spec != other

  def __str__(self):
    key_values = self.as_dict()
    string_items = [
        repr(k) + ": " + repr(key_values[k]) for k in sorted(key_values)]
    return "ClusterSpec({" + ", ".join(string_items) + "})"

  def as_dict(self):
    """Returns a dictionary from job names to their tasks.

    For each job, if the task index space is dense, the corresponding
    value will be a list of network addresses; otherwise it will be a
    dictionary mapping (sparse) task indices to the corresponding
    addresses.

    Returns:
      A dictionary mapping job names to lists or dictionaries
      describing the tasks in those jobs.
    """
    ret = {}
    for job in self.jobs:
      task_indices = self.task_indices(job)
      if max(task_indices) + 1 == len(task_indices):
        # Return a list because the task indices are dense. This
        # matches the behavior of `as_dict()` before support for
        # sparse jobs was added.
        ret[job] = self.job_tasks(job)
      else:
        ret[job] = {i: self.task_address(job, i) for i in task_indices}
    return ret

  def as_cluster_def(self):
    """Returns a `tf.train.ClusterDef` protocol buffer based on this cluster."""
    return self._cluster_def

  @property
  def jobs(self):
    """Returns a list of job names in this cluster.

    Returns:
      A list of strings, corresponding to the names of jobs in this cluster.
    """
    return list(self._cluster_spec.keys())

  def num_tasks(self, job_name):
    """Returns the number of tasks defined in the given job.

    Args:
      job_name: The string name of a job in this cluster.

    Returns:
      The number of tasks defined in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    return len(job)

  def task_indices(self, job_name):
    """Returns a list of valid task indices in the given job.

    Args:
      job_name: The string name of a job in this cluster.

    Returns:
      A list of valid task indices in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster,
      or no task with index `task_index` is defined in that job.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    return list(sorted(job.keys()))

  def task_address(self, job_name, task_index):
    """Returns the address of the given task in the given job.

    Args:
      job_name: The string name of a job in this cluster.
      task_index: A non-negative integer.

    Returns:
      The address of the given task in the given job.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster,
      or no task with index `task_index` is defined in that job.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    try:
      return job[task_index]
    except KeyError:
      raise ValueError("No task with index %r in job %r"
                       % (task_index, job_name))

  def job_tasks(self, job_name):
    """Returns a mapping from task ID to address in the given job.

    NOTE: For backwards compatibility, this method returns a list. If
    the given job was defined with a sparse set of task indices, the
    length of this list may not reflect the number of tasks defined in
    this job. Use the @{tf.train.ClusterSpec.num_tasks} method
    to find the number of tasks defined in a particular job.

    Args:
      job_name: The string name of a job in this cluster.

    Returns:
      A list of task addresses, where the index in the list
      corresponds to the task index of each task. The list may contain
      `None` if the job was defined with a sparse set of task indices.

    Raises:
      ValueError: If `job_name` does not name a job in this cluster.
    """
    try:
      job = self._cluster_spec[job_name]
    except KeyError:
      raise ValueError("No such job in cluster: %r" % job_name)
    ret = [None for _ in range(max(job.keys()) + 1)]
    for i, task in job.items():
      ret[i] = task
    return ret

  def _make_cluster_def(self):
    """Creates a `tf.train.ClusterDef` based on the given `cluster_spec`.

    Raises:
      TypeError: If `cluster_spec` is not a dictionary mapping strings to lists
        of strings.
    """
    self._cluster_def = cluster_pb2.ClusterDef()

    # NOTE(mrry): Sort by job_name to produce deterministic protobufs.
    for job_name, tasks in sorted(self._cluster_spec.items()):
      try:
        job_name = compat.as_bytes(job_name)
      except TypeError:
        raise TypeError("Job name %r must be bytes or unicode" % job_name)

      job_def = self._cluster_def.job.add()
      job_def.name = job_name

      for i, task_address in sorted(tasks.items()):
        try:
          task_address = compat.as_bytes(task_address)
        except TypeError:
          raise TypeError(
              "Task address %r must be bytes or unicode" % task_address)
        job_def.tasks[i] = task_address
