# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Debugger wrapper session that sends debug data to file:// URLs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import signal
import sys
import traceback

import six

# Google-internal import(s).
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.wrappers import framework


def publish_traceback(debug_server_urls,
                      graph,
                      feed_dict,
                      fetches,
                      old_graph_version):
  """Publish traceback and source code if graph version is new.

  `graph.version` is compared with `old_graph_version`. If the former is higher
  (i.e., newer), the graph traceback and the associated source code is sent to
  the debug server at the specified gRPC URLs.

  Args:
    debug_server_urls: A single gRPC debug server URL as a `str` or a `list` of
      debug server URLs.
    graph: A Python `tf.Graph` object.
    feed_dict: Feed dictionary given to the `Session.run()` call.
    fetches: Fetches from the `Session.run()` call.
    old_graph_version: Old graph version to compare to.

  Returns:
    If `graph.version > old_graph_version`, the new graph version as an `int`.
    Else, the `old_graph_version` is returned.
  """
  # TODO(cais): Consider moving this back to the top, after grpc becomes a
  # pip dependency of tensorflow or tf_debug.
  # pylint:disable=g-import-not-at-top
  from tensorflow.python.debug.lib import source_remote
  # pylint:enable=g-import-not-at-top
  if graph.version > old_graph_version:
    run_key = common.get_run_key(feed_dict, fetches)
    source_remote.send_graph_tracebacks(
        debug_server_urls, run_key, traceback.extract_stack(), graph,
        send_source=True)
    return graph.version
  else:
    return old_graph_version


class GrpcDebugWrapperSession(framework.NonInteractiveDebugWrapperSession):
  """Debug Session wrapper that send debug data to gRPC stream(s)."""

  def __init__(self,
               sess,
               grpc_debug_server_addresses,
               watch_fn=None,
               thread_name_filter=None,
               log_usage=True):
    """Constructor of DumpingDebugWrapperSession.

    Args:
      sess: The TensorFlow `Session` object being wrapped.
      grpc_debug_server_addresses: (`str` or `list` of `str`) Single or a list
        of the gRPC debug server addresses, in the format of
        <host:port>, with or without the "grpc://" prefix. For example:
          "localhost:7000",
          ["localhost:7000", "192.168.0.2:8000"]
      watch_fn: (`Callable`) A Callable that can be used to define per-run
        debug ops and watched tensors. See the doc of
        `NonInteractiveDebugWrapperSession.__init__()` for details.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
      log_usage: (`bool`) whether the usage of this class is to be logged.

    Raises:
       TypeError: If `grpc_debug_server_addresses` is not a `str` or a `list`
         of `str`.
    """

    if log_usage:
      pass  # No logging for open-source.

    framework.NonInteractiveDebugWrapperSession.__init__(
        self, sess, watch_fn=watch_fn, thread_name_filter=thread_name_filter)

    if isinstance(grpc_debug_server_addresses, str):
      self._grpc_debug_server_urls = [
          self._normalize_grpc_url(grpc_debug_server_addresses)]
    elif isinstance(grpc_debug_server_addresses, list):
      self._grpc_debug_server_urls = []
      for address in grpc_debug_server_addresses:
        if not isinstance(address, str):
          raise TypeError(
              "Expected type str in list grpc_debug_server_addresses, "
              "received type %s" % type(address))
        self._grpc_debug_server_urls.append(self._normalize_grpc_url(address))
    else:
      raise TypeError(
          "Expected type str or list in grpc_debug_server_addresses, "
          "received type %s" % type(grpc_debug_server_addresses))

  def prepare_run_debug_urls(self, fetches, feed_dict):
    """Implementation of abstract method in superclass.

    See doc of `NonInteractiveDebugWrapperSession.prepare_run_debug_urls()`
    for details.

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`
      feed_dict: Same as the `feed_dict` argument to `Session.run()`

    Returns:
      debug_urls: (`str` or `list` of `str`) file:// debug URLs to be used in
        this `Session.run()` call.
    """

    return self._grpc_debug_server_urls

  def _normalize_grpc_url(self, address):
    return (common.GRPC_URL_PREFIX + address
            if not address.startswith(common.GRPC_URL_PREFIX) else address)


def _signal_handler(unused_signal, unused_frame):
  while True:
    response = six.moves.input(
        "\nSIGINT received. Quit program? (Y/n): ").strip()
    if response in ("", "Y", "y"):
      sys.exit(0)
    elif response in ("N", "n"):
      break


def register_signal_handler():
  try:
    signal.signal(signal.SIGINT, _signal_handler)
  except ValueError:
    # This can happen if we are not in the MainThread.
    pass


class TensorBoardDebugWrapperSession(GrpcDebugWrapperSession):
  """A tfdbg Session wrapper that can be used with TensorBoard Debugger Plugin.

  This wrapper is the same as `GrpcDebugWrapperSession`, except that it uses a
    predefined `watch_fn` that
    1) uses `DebugIdentity` debug ops with the `gated_grpc` attribute set to
        `True` to allow the interactive enabling and disabling of tensor
       breakpoints.
    2) watches all tensors in the graph.
  This saves the need for the user to define a `watch_fn`.
  """

  def __init__(self,
               sess,
               grpc_debug_server_addresses,
               thread_name_filter=None,
               send_traceback_and_source_code=True,
               log_usage=True):
    """Constructor of TensorBoardDebugWrapperSession.

    Args:
      sess: The `tf.Session` instance to be wrapped.
      grpc_debug_server_addresses: gRPC address(es) of debug server(s), as a
        `str` or a `list` of `str`s. E.g., "localhost:2333",
        "grpc://localhost:2333", ["192.168.0.7:2333", "192.168.0.8:2333"].
      thread_name_filter: Optional filter for thread names.
      send_traceback_and_source_code: Whether traceback of graph elements and
        the source code are to be sent to the debug server(s).
      log_usage: Whether the usage of this class is to be logged (if
        applicable).
    """
    def _gated_grpc_watch_fn(fetches, feeds):
      del fetches, feeds  # Unused.
      return framework.WatchOptions(
          debug_ops=["DebugIdentity(gated_grpc=true)"])

    super(TensorBoardDebugWrapperSession, self).__init__(
        sess,
        grpc_debug_server_addresses,
        watch_fn=_gated_grpc_watch_fn,
        thread_name_filter=thread_name_filter,
        log_usage=log_usage)

    self._send_traceback_and_source_code = send_traceback_and_source_code
    # Keeps track of the latest version of Python graph object that has been
    # sent to the debug servers.
    self._sent_graph_version = -1

    register_signal_handler()

  def run(self,
          fetches,
          feed_dict=None,
          options=None,
          run_metadata=None,
          callable_runner=None,
          callable_runner_args=None,
          callable_options=None):
    if self._send_traceback_and_source_code:
      self._sent_graph_version = publish_traceback(
          self._grpc_debug_server_urls, self.graph, feed_dict, fetches,
          self._sent_graph_version)
    return super(TensorBoardDebugWrapperSession, self).run(
        fetches,
        feed_dict=feed_dict,
        options=options,
        run_metadata=run_metadata,
        callable_runner=callable_runner,
        callable_runner_args=callable_runner_args,
        callable_options=callable_options)
