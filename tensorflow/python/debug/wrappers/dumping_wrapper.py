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
"""Debugger wrapper session that dumps debug data to file:// URLs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import uuid

# Google-internal import(s).
from tensorflow.core.util import event_pb2
from tensorflow.python.debug import debug_data
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.platform import gfile


class DumpingDebugWrapperSession(framework.BaseDebugWrapperSession):
  """Debug Session wrapper that dumps debug data to filesystem."""

  def __init__(self, sess, session_root, watch_fn=None, log_usage=True):
    """Constructor of DumpingDebugWrapperSession.

    Args:
      sess: The TensorFlow `Session` object being wrapped.
      session_root: (`str`) Path to the session root directory. Must be a
        directory that does not exist or an empty directory. If the directory
        does not exist, it will be created by the debugger core during debug
        [`Session.run()`](../../../g3doc/api_docs/python/client.md#session.run)
        calls.
        As the `run()` calls occur, subdirectories will be added to
        `session_root`. The subdirectories' names has the following pattern:
          run_<epoch_time_stamp>_<uuid>
        E.g., run_1480734393835964_ad4c953a85444900ae79fc1b652fb324
      watch_fn: (`Callable`) A Callable of the following signature:
        ```
        def watch_fn(fetches, feeds):
          # Args:
          #   fetches: the fetches to the `Session.run()` call.
          #   feeds: the feeds to the `Session.run()` call.
          #
          # Returns: (node_name_regex_whitelist, op_type_regex_whitelist)
          #   debug_ops: (str or list of str) Debug op(s) to be used by the
          #     debugger in this run() call.
          #   node_name_regex_whitelist: Regular-expression whitelist for node
          #     name. Same as the corresponding arg to `debug_util.watch_graph`.
          #   op_type_regex_whiteslit: Regular-expression whitelist for op type.
          #     Same as the corresponding arg to `debug_util.watch_graph`.
          #
          #   Both or either can be None. If both are set, the two whitelists
          #   will operate in a logical AND relation. This is consistent with
          #   `debug_utils.watch_graph()`.
        ```
      log_usage: (`bool`) whether the usage of this class is to be logged.

    Raises:
       ValueError: If `session_root` is an existing and non-empty directory or
       if
         `session_root` is a file.
       TypeError: If a non-None `watch_fn` is specified and it is not callable.
    """

    if log_usage:
      pass  # No logging for open-source.

    framework.BaseDebugWrapperSession.__init__(self, sess)

    self._watch_fn = None
    if watch_fn is not None:
      if not callable(watch_fn):
        raise TypeError("watch_fn is not callable")
      self._watch_fn = watch_fn

    if gfile.Exists(session_root):
      if not gfile.IsDirectory(session_root):
        raise ValueError(
            "session_root path points to a file: %s" % session_root)
      elif gfile.ListDirectory(session_root):
        raise ValueError(
            "session_root path points to a non-empty directory: %s" %
            session_root)
    self._session_root = session_root

  def on_session_init(self, request):
    """See doc of BaseDebugWrapperSession.on_run_start."""

    return framework.OnSessionInitResponse(
        framework.OnSessionInitAction.PROCEED)

  def on_run_start(self, request):
    """See doc of BaseDebugWrapperSession.on_run_start."""

    (debug_urls, debug_ops, node_name_regex_whitelist,
     op_type_regex_whitelist) = self._prepare_run_watch_config(
         request.fetches, request.feed_dict)

    return framework.OnRunStartResponse(
        framework.OnRunStartAction.DEBUG_RUN,
        debug_urls,
        debug_ops=debug_ops,
        node_name_regex_whitelist=node_name_regex_whitelist,
        op_type_regex_whitelist=op_type_regex_whitelist)

  def _prepare_run_watch_config(self, fetches, feed_dict):
    """Get the debug_urls, and node/op whitelists for the current run() call.

    Prepares a directory with a fixed naming pattern. Saves Event proto files
    of names `_tfdbg_run_fetches_info` and `_tfdbg_run_feed_keys_info` in the
    directory to save information about the `fetches` and `feed_dict.keys()`
    used in this `run()` call, respectively.

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`.
      feed_dict: Same as the `feed_dict argument` to `Session.run()`.

    Returns:
      debug_urls: (str or list of str) Debug URLs for the current run() call.
        Currently, the list consists of only one URL that is a file:// URL.
      debug_ops: (str or list of str) Debug op(s) to be used by the
        debugger.
      node_name_regex_whitelist: (str or regex) Regular-expression whitelist for
        node name. Same as the same-name argument to debug_utils.watch_graph.
      op_type_regex_whitelist: (str or regex) Regular-expression whitelist for
        op type. Same as the same-name argument to debug_utils.watch_graph.
    """

    # Add a UUID to accommodate the possibility of concurrent run() calls.
    run_dir = os.path.join(self._session_root, "run_%d_%s" %
                           (int(time.time() * 1e6), uuid.uuid4().hex))
    gfile.MkDir(run_dir)

    fetches_event = event_pb2.Event()
    fetches_event.log_message.message = repr(fetches)
    fetches_path = os.path.join(
        run_dir,
        debug_data.METADATA_FILE_PREFIX + debug_data.FETCHES_INFO_FILE_TAG)
    with gfile.Open(os.path.join(fetches_path), "wb") as f:
      f.write(fetches_event.SerializeToString())

    feed_keys_event = event_pb2.Event()
    feed_keys_event.log_message.message = (repr(feed_dict.keys()) if feed_dict
                                           else repr(feed_dict))

    feed_keys_path = os.path.join(
        run_dir,
        debug_data.METADATA_FILE_PREFIX + debug_data.FEED_KEYS_INFO_FILE_TAG)
    with gfile.Open(os.path.join(feed_keys_path), "wb") as f:
      f.write(feed_keys_event.SerializeToString())

    debug_ops, node_name_regex_whitelist, op_type_regex_whitelist = (
        "DebugIdentity", None, None)
    if self._watch_fn is not None:
      debug_ops, node_name_regex_whitelist, op_type_regex_whitelist = (
          self._watch_fn(fetches, feed_dict))

    return (["file://" + run_dir], debug_ops, node_name_regex_whitelist,
            op_type_regex_whitelist)

  def on_run_end(self, request):
    """See doc of BaseDebugWrapperSession.on_run_end."""

    return framework.OnRunEndResponse()

  def invoke_node_stepper(self,
                          node_stepper,
                          restore_variable_values_on_exit=True):
    """See doc of BaseDebugWrapperSession.invoke_node_stepper."""

    return NotImplementedError(
        "DumpingDebugWrapperSession does not support node-stepper mode.")
