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
"""Writer class for `DebugEvent` protos in tfdbg v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer

# Default size of each circular buffer (unit: number of DebugEvent protos).
DEFAULT_CIRCULAR_BUFFER_SIZE = 1000


class DebugEventsWriter(object):
  """A writer for TF debugging events. Used by tfdbg v2."""

  def __init__(self,
               dump_root,
               tfdbg_run_id,
               circular_buffer_size=DEFAULT_CIRCULAR_BUFFER_SIZE):
    """Construct a DebugEventsWriter object.

    NOTE: Given the same `dump_root`, all objects from this constructor
      will point to the same underlying set of writers. In other words, they
      will write to the same set of debug events files in the `dump_root`
      folder.

    Args:
      dump_root: The root directory for dumping debug data. If `dump_root` does
        not exist as a directory, it will be created.
      tfdbg_run_id: Debugger Run ID.
      circular_buffer_size: Size of the circular buffer for each of the two
        execution-related debug events files: with the following suffixes: -
          .execution - .graph_execution_traces If <= 0, the circular-buffer
          behavior will be abolished in the constructed object.
    """
    if not dump_root:
      raise ValueError("Empty or None dump root")
    self._dump_root = dump_root
    self._tfdbg_run_id = tfdbg_run_id
    _pywrap_debug_events_writer.Init(self._dump_root, self._tfdbg_run_id,
                                     circular_buffer_size)

  def WriteSourceFile(self, source_file):
    """Write a SourceFile proto with the writer.

    Args:
      source_file: A SourceFile proto, describing the content of a source file
        involved in the execution of the debugged TensorFlow program.
    """
    # TODO(cais): Explore performance optimization that avoids memcpy.
    debug_event = debug_event_pb2.DebugEvent(source_file=source_file)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteSourceFile(self._dump_root, debug_event)

  def WriteStackFrameWithId(self, stack_frame_with_id):
    """Write a StackFrameWithId proto with the writer.

    Args:
      stack_frame_with_id: A StackFrameWithId proto, describing the content a
        stack frame involved in the execution of the debugged TensorFlow
        program.
    """
    debug_event = debug_event_pb2.DebugEvent(
        stack_frame_with_id=stack_frame_with_id)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteStackFrameWithId(self._dump_root,
                                                      debug_event)

  def WriteGraphOpCreation(self, graph_op_creation):
    """Write a GraphOpCreation proto with the writer.

    Args:
      graph_op_creation: A GraphOpCreation proto, describing the details of the
        creation of an op inside a TensorFlow Graph.
    """
    debug_event = debug_event_pb2.DebugEvent(
        graph_op_creation=graph_op_creation)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteGraphOpCreation(self._dump_root,
                                                     debug_event)

  def WriteDebuggedGraph(self, debugged_graph):
    """Write a DebuggedGraph proto with the writer.

    Args:
      debugged_graph: A DebuggedGraph proto, describing the details of a
        TensorFlow Graph that has completed its construction.
    """
    debug_event = debug_event_pb2.DebugEvent(debugged_graph=debugged_graph)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteDebuggedGraph(self._dump_root, debug_event)

  def WriteExecution(self, execution):
    """Write a Execution proto with the writer.

    Args:
      execution: An Execution proto, describing a TensorFlow op or graph
        execution event.
    """
    debug_event = debug_event_pb2.DebugEvent(execution=execution)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteExecution(self._dump_root, debug_event)

  def WriteGraphExecutionTrace(self, graph_execution_trace):
    """Write a GraphExecutionTrace proto with the writer.

    Args:
      graph_execution_trace: A GraphExecutionTrace proto, concerning the value
        of an intermediate tensor or a list of intermediate tensors that are
        computed during the graph's execution.
    """
    debug_event = debug_event_pb2.DebugEvent(
        graph_execution_trace=graph_execution_trace)
    self._EnsureTimestampAdded(debug_event)
    _pywrap_debug_events_writer.WriteGraphExecutionTrace(
        self._dump_root, debug_event)

  def RegisterDeviceAndGetId(self, device_name):
    return _pywrap_debug_events_writer.RegisterDeviceAndGetId(
        self._dump_root, device_name)

  def FlushNonExecutionFiles(self):
    """Flush the non-execution debug event files."""
    _pywrap_debug_events_writer.FlushNonExecutionFiles(self._dump_root)

  def FlushExecutionFiles(self):
    """Flush the execution debug event files.

    Causes the current content of the cyclic buffers to be written to
    the .execution and .graph_execution_traces debug events files.
    Also clears those cyclic buffers.
    """
    _pywrap_debug_events_writer.FlushExecutionFiles(self._dump_root)

  def Close(self):
    """Close the writer."""
    _pywrap_debug_events_writer.Close(self._dump_root)

  @property
  def dump_root(self):
    return self._dump_root

  def _EnsureTimestampAdded(self, debug_event):
    if debug_event.wall_time == 0:
      debug_event.wall_time = time.time()
