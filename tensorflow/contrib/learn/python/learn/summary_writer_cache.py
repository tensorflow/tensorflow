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
"""Wrapper for a Session-like object that handles threads and recovery.

Based on an original design of Illia Polosukhin.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading

from tensorflow.python.framework import ops
from tensorflow.python.training import summary_io


class SummaryWriterCache(object):
  """Cache for summary writers.

  This class caches summary writers, one per directory.
  """
  # Cache, keyed by directory.
  _cache = {}

  # Lock protecting _SUMMARY_WRITERS.
  _lock = threading.RLock()

  @staticmethod
  def clear():
    """Clear cached summary writers. Currently only used for unit tests."""
    with SummaryWriterCache._lock:
      SummaryWriterCache._cache = {}

  @staticmethod
  def get(logdir):
    """Returns the SummaryWriter for the specified directory.

    Args:
      logdir: str, name of the directory.

    Returns:
      A `SummaryWriter`.
    """
    with SummaryWriterCache._lock:
      if logdir not in SummaryWriterCache._cache:
        SummaryWriterCache._cache[logdir] = summary_io.SummaryWriter(
            logdir, graph=ops.get_default_graph())
      return SummaryWriterCache._cache[logdir]


# Backward compatible interface.  Remove?
clear_summary_writers = SummaryWriterCache.clear
get_summary_writer = SummaryWriterCache.get
