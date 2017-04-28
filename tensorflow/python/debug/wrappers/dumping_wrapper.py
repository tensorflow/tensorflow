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
import threading
import time

# Google-internal import(s).
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.platform import gfile


class DumpingDebugWrapperSession(framework.NonInteractiveDebugWrapperSession):
  """Debug Session wrapper that dumps debug data to filesystem."""

  def __init__(self,
               sess,
               session_root,
               watch_fn=None,
               thread_name_filter=None,
               log_usage=True):
    """Constructor of DumpingDebugWrapperSession.

    Args:
      sess: The TensorFlow `Session` object being wrapped.
      session_root: (`str`) Path to the session root directory. Must be a
        directory that does not exist or an empty directory. If the directory
        does not exist, it will be created by the debugger core during debug
        @{tf.Session.run}
        calls.
        As the `run()` calls occur, subdirectories will be added to
        `session_root`. The subdirectories' names has the following pattern:
          run_<epoch_time_stamp>_<zero_based_run_counter>
        E.g., run_1480734393835964_ad4c953a85444900ae79fc1b652fb324
      watch_fn: (`Callable`) A Callable that can be used to define per-run
        debug ops and watched tensors. See the doc of
        `NonInteractiveDebugWrapperSession.__init__()` for details.
      thread_name_filter: Regular-expression white list for threads on which the
        wrapper session will be active. See doc of `BaseDebugWrapperSession` for
        more details.
      log_usage: (`bool`) whether the usage of this class is to be logged.

    Raises:
       ValueError: If `session_root` is an existing and non-empty directory or
       if `session_root` is a file.
    """

    if log_usage:
      pass  # No logging for open-source.

    framework.NonInteractiveDebugWrapperSession.__init__(
        self, sess, watch_fn=watch_fn, thread_name_filter=thread_name_filter)

    if gfile.Exists(session_root):
      if not gfile.IsDirectory(session_root):
        raise ValueError(
            "session_root path points to a file: %s" % session_root)
      elif gfile.ListDirectory(session_root):
        raise ValueError(
            "session_root path points to a non-empty directory: %s" %
            session_root)
    self._session_root = session_root

    self._run_counter = 0
    self._run_counter_lock = threading.Lock()

  def prepare_run_debug_urls(self, fetches, feed_dict):
    """Implementation of abstrat method in superclass.

    See doc of `NonInteractiveDebugWrapperSession.prepare_run_debug_urls()`
    for details. This implentation creates a run-specific subdirectory under
    self._session_root and stores information regarding run `fetches` and
    `feed_dict.keys()` in the subdirectory.

    Args:
      fetches: Same as the `fetches` argument to `Session.run()`
      feed_dict: Same as the `feed_dict` argument to `Session.run()`

    Returns:
      debug_urls: (`str` or `list` of `str`) file:// debug URLs to be used in
        this `Session.run()` call.
    """

    # Add a UUID to accommodate the possibility of concurrent run() calls.
    self._run_counter_lock.acquire()
    run_dir = os.path.join(self._session_root, "run_%d_%d" %
                           (int(time.time() * 1e6), self._run_counter))
    self._run_counter += 1
    self._run_counter_lock.release()
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

    return ["file://" + run_dir]
