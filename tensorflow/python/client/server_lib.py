# Copyright 2015 Google Inc. All Rights Reserved.
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

import six  # pylint: disable=unused-import

from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.util import compat


class GrpcServer(object):
  """An in-process TensorFlow server.

  NOTE(mrry): This class is experimental and not yet suitable for use.
  """

  def __init__(self, server_def, start=True):
    """Creates a new server with the given definition.

    Args:
      server_def: A `tf.ServerDef` protocol buffer, describing the server to
        be created (and the cluster of which it is a member).
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.
    """
    if not isinstance(server_def, tensorflow_server_pb2.ServerDef):
      raise TypeError("server_def must be a tf.ServerDef")

    self._server = pywrap_tensorflow.NewServer(server_def.SerializeToString())
    if start:
      self.start()

  def start(self):
    """Starts this server."""
    self._server.Start()

  def stop(self):
    """Stops this server.

    NOTE(mrry): This method is currently not implemented.
    """
    # TODO(mrry): Implement this.
    raise NotImplementedError("GrpcServer.stop()")

  def join(self):
    """Blocks until the server has shut down.

    NOTE(mrry): Since `GrpcServer.stop()` is not currently implemented, this
    method blocks forever.
    """
    self._server.Join()

  @property
  def target(self):
    """Returns the target for a `tf.Session` to connect to this server.

    To create a
    [`tf.Session`](../../api_docs/python/client.md#Session) that
    connects to this server, use the following snippet:

    ```python
    server = tf.GrpcServer(...)
    with tf.Session(server.target):
      # ...
    ```

    Returns:
      A string containing a session target for this server.
    """
    return self._server.target()

  @staticmethod
  def create_local_server(start=True):
    """Creates a new single-process cluster running on the local host.

    This method is a convenience wrapper for calling
    `GrpcServer.__init__()` with a `tf.ServerDef` that specifies a
    single-process cluster, with a single task in a job called
    `"local"`.

    Args:
      start: (Optional.) Boolean, indicating whether to start the server after
        creating it. Defaults to `True`.

    Returns:
      A local `tf.GrpcServer`.
    """
    server_def = tensorflow_server_pb2.ServerDef(protocol="grpc")
    job_def = server_def.cluster.job.add()
    job_def.name = "local"
    job_def.tasks[0] = "localhost:0"
    server_def.job_name = job_def.name
    server_def.task_index = 0
    return GrpcServer(server_def, start)


def make_cluster_def(cluster_spec):
  """Returns a `tf.ClusterDef` based on the given `cluster_spec`.

  Args:
    cluster_spec: A dictionary mapping one or more job names to lists
      of network addresses.

  Returns:
    A `tf.ClusterDef` protocol buffer.

  Raises:
    TypeError: If `cluster_spec` is not a dictionary mapping strings to lists
      of strings.
  """
  if not isinstance(cluster_spec, dict):
    raise TypeError("`cluster_spec` must be a dictionary mapping one or more "
                    "job names to lists of network addresses")

  cluster_def = tensorflow_server_pb2.ClusterDef()

  # NOTE(mrry): Sort by job_name to produce deterministic protobufs.
  for job_name, task_list in sorted(cluster_spec.items()):
    try:
      job_name = compat.as_bytes(job_name)
    except TypeError:
      raise TypeError("Job name %r must be bytes or unicode" % job_name)

    job_def = cluster_def.job.add()
    job_def.name = job_name

    for i, task_address in enumerate(task_list):
      try:
        task_address = compat.as_bytes(task_address)
      except TypeError:
        raise TypeError(
            "Task address %r must be bytes or unicode" % task_address)
      job_def.tasks[i] = task_address

  return cluster_def
