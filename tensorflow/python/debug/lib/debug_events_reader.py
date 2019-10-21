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

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.lib.io import tf_record


def _check_debug_event_file_exists(file_path):
  if not os.path.isfile(file_path):
    raise ValueError("DebugEvent data file does not exist: %s" % file_path)


class DebugEventsDir(object):
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
    self._metadata_path = metadata_paths[0]

    prefix = metadata_paths[0][:-len(".metadata")]
    self._source_files_path = "%s.source_files" % prefix
    self._stack_frames_path = "%s.stack_frames" % prefix
    self._graphs_path = "%s.graphs" % prefix
    self._execution_path = "%s.execution" % prefix
    self._graph_execution_traces_path = ("%s.graph_execution_traces" %
                                         prefix)

  def metadata_iterator(self):
    for r in tf_record.tf_record_iterator(self._metadata_path):
      yield debug_event_pb2.DebugEvent.FromString(r)

  def source_files_iterator(self):
    _check_debug_event_file_exists(self._source_files_path)
    for r in tf_record.tf_record_iterator(self._source_files_path):
      yield debug_event_pb2.DebugEvent.FromString(r)

  def stack_frames_iterator(self):
    _check_debug_event_file_exists(self._stack_frames_path)
    for r in tf_record.tf_record_iterator(self._stack_frames_path):
      yield debug_event_pb2.DebugEvent.FromString(r)

  def graphs_iterator(self):
    _check_debug_event_file_exists(self._graphs_path)
    for r in tf_record.tf_record_iterator(self._graphs_path):
      yield debug_event_pb2.DebugEvent.FromString(r)

  def execution_iterator(self):
    _check_debug_event_file_exists(self._execution_path)
    for r in tf_record.tf_record_iterator(self._execution_path):
      yield debug_event_pb2.DebugEvent.FromString(r)

  def graph_execution_traces_iterator(self):
    _check_debug_event_file_exists(self._graph_execution_traces_path)
    for r in tf_record.tf_record_iterator(self._graph_execution_traces_path):
      yield debug_event_pb2.DebugEvent.FromString(r)
