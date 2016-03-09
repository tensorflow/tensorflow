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
