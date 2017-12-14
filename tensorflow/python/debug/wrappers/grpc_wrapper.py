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

# Google-internal import(s).
from tensorflow.python.debug.wrappers import framework


class GrpcDebugWrapperSession(framework.NonInteractiveDebugWrapperSession):
  """Debug Session wrapper that send debug data to gRPC stream(s)."""

  _GRPC_URL_PREFIX = "grpc://"

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
    return (self._GRPC_URL_PREFIX + address
            if not address.startswith(self._GRPC_URL_PREFIX) else address)


class TensorBoardDebugWrapperSession(GrpcDebugWrapperSession):
  """A tfdbg Session wrapper that can be used with TensroBoard Debugger Plugin.

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
               log_usage=True):
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
