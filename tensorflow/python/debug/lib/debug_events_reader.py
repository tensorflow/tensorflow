# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Reader class for tfdbg v2 debug events."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import threading

from six.moves import map

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util import compat


class DebugEventsReader(object):
  """Reader class for a tfdbg v2 DebugEvents directory."""

  def __init__(self, dump_root):
    if not os.path.isdir(dump_root):
      raise ValueError("Specified dump_root is not a directory: %s" % dump_root)
    metadata_paths = glob.glob(os.path.join(dump_root, "*.metadata"))
    if not metadata_paths:
      raise ValueError("Cannot find any metadata file in directory: %s" %
                       dump_root)
    elif len(metadata_paths) > 1:
      raise ValueError(
          "Unexpected: Found multiple (%d) metadata in directory: %s" %
          (len(metadata_paths), dump_root))
    self._metadata_path = compat.as_bytes(metadata_paths[0])
    self._metadata_reader = None

    prefix = metadata_paths[0][:-len(".metadata")]
    self._source_files_path = compat.as_bytes("%s.source_files" % prefix)
    self._stack_frames_path = compat.as_bytes("%s.stack_frames" % prefix)
    self._graphs_path = compat.as_bytes("%s.graphs" % prefix)
    self._execution_path = compat.as_bytes("%s.execution" % prefix)
    self._graph_execution_traces_path = compat.as_bytes(
        "%s.graph_execution_traces" % prefix)
    self._readers = dict()  # A map from file path to reader.
    self._readers_lock = threading.Lock()

  def __enter__(self):
    return self

  def __exit__(self, exception_type, exception_value, traceback):
    del exception_type, exception_value, traceback  # Unused
    self.close()

  def _generic_iterator(self, file_path):
    """A helper method that makes an iterator given a debug-events file path."""
    # The following code uses the double-checked locking pattern to optimize
    # the common case (where the reader is already initialized).
    if file_path not in self._readers:  # 1st check, without lock.
      with self._readers_lock:
        if file_path not in self._readers:  # 2nd check, with lock.
          self._readers[file_path] = tf_record.tf_record_iterator(file_path)

    return map(debug_event_pb2.DebugEvent.FromString, self._readers[file_path])

  def metadata_iterator(self):
    return self._generic_iterator(self._metadata_path)

  def source_files_iterator(self):
    return self._generic_iterator(self._source_files_path)

  def stack_frames_iterator(self):
    return self._generic_iterator(self._stack_frames_path)

  def graphs_iterator(self):
    return self._generic_iterator(self._graphs_path)

  def execution_iterator(self):
    return self._generic_iterator(self._execution_path)

  def graph_execution_traces_iterator(self):
    return self._generic_iterator(self._graph_execution_traces_path)

  def close(self):
    with self._readers_lock:
      self._readers.clear()
