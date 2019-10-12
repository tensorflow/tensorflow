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
"""Tests for the debug events writer Python class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import tempfile
import threading

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import googletest


def ReadDebugEvents(filename):
  reader = tf_record.tf_record_iterator(filename)

  debug_events = []
  try:
    while True:
      debug_event = debug_event_pb2.DebugEvent()
      debug_event.ParseFromString(next(reader))
      debug_events.append(debug_event)
  except StopIteration:
    return debug_events


class PywrapeventsWriterTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(PywrapeventsWriterTest, self).setUp()
    self.dump_root = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self.dump_root):
      file_io.delete_recursively(self.dump_root)
    super(PywrapeventsWriterTest, self).tearDown()

  def testMultiThreadedConstructorCallWorks(self):

    def InitWriter():
      debug_events_writer.DebugEventsWriter(self.dump_root)

    num_threads = 4
    threads = []
    for _ in range(num_threads):
      thread = threading.Thread(target=InitWriter)
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()

    # Verify that there is only one debug event file of each type.
    metadata_paths = glob.glob(os.path.join(self.dump_root, "*.metadata"))
    self.assertEqual(len(metadata_paths), 1)
    source_files_paths = glob.glob(
        os.path.join(self.dump_root, "*.source_files"))
    self.assertEqual(len(source_files_paths), 1)
    stack_frames_paths = glob.glob(
        os.path.join(self.dump_root, "*.stack_frames"))
    self.assertEqual(len(stack_frames_paths), 1)
    graphs_paths = glob.glob(os.path.join(self.dump_root, "*.graphs"))
    self.assertEqual(len(graphs_paths), 1)

    # Verify the content of the metadata file.
    debug_events = ReadDebugEvents(metadata_paths[0])
    self.assertEqual(len(metadata_paths), 1)
    self.assertTrue(debug_events[0].debug_metadata.tensorflow_version)
    self.assertTrue(
        debug_events[0].debug_metadata.file_version.startswith("debug.Event:"))

  def testWriteSourceFilesAndStackFrames(self):
    writer = debug_events_writer.DebugEventsWriter(self.dump_root)
    num_protos = 10
    for i in range(num_protos):
      source_file = debug_event_pb2.SourceFile()
      source_file.file_path = "/home/tf2user/main.py"
      source_file.host_name = "machine.cluster"
      source_file.lines.append("print(%d)" % i)
      writer.WriteSourceFile(source_file)

      stack_frame = debug_event_pb2.StackFrameWithId()
      stack_frame.id = "stack_%d" % i
      stack_frame.file_line_col.file_index = i * 10
      writer.WriteStackFrameWithId(stack_frame)

    writer.FlushNonExecutionFiles()

    source_files_paths = glob.glob(
        os.path.join(self.dump_root, "*.source_files"))
    self.assertEqual(len(source_files_paths), 1)
    actuals = ReadDebugEvents(source_files_paths[0])
    self.assertEqual(len(actuals), num_protos)
    for i in range(num_protos):
      self.assertEqual(actuals[i].source_file.file_path,
                       "/home/tf2user/main.py")
      self.assertEqual(actuals[i].source_file.host_name, "machine.cluster")
      self.assertEqual(actuals[i].source_file.lines, ["print(%d)" % i])

    stack_frames_paths = glob.glob(
        os.path.join(self.dump_root, "*.stack_frames"))
    self.assertEqual(len(stack_frames_paths), 1)
    actuals = ReadDebugEvents(stack_frames_paths[0])
    self.assertEqual(len(actuals), num_protos)
    for i in range(num_protos):
      self.assertEqual(actuals[i].stack_frame_with_id.id, "stack_%d" % i)
      self.assertEqual(actuals[i].stack_frame_with_id.file_line_col.file_index,
                       i * 10)

  def testWriteGraphOpCreationAndDebuggedGraphs(self):
    writer = debug_events_writer.DebugEventsWriter(self.dump_root)
    num_op_creations = 10
    for i in range(num_op_creations):
      graph_op_creation = debug_event_pb2.GraphOpCreation()
      graph_op_creation.op_type = "Conv2D"
      graph_op_creation.op_name = "Conv2D_%d" % i
      writer.WriteGraphOpCreation(graph_op_creation)
    debugged_graph = debug_event_pb2.DebuggedGraph()
    debugged_graph.graph_id = "deadbeaf"
    debugged_graph.graph_name = "MyGraph1"
    writer.WriteDebuggedGraph(debugged_graph)
    writer.FlushNonExecutionFiles()

    source_files_paths = glob.glob(os.path.join(self.dump_root, "*.graphs"))
    self.assertEqual(len(source_files_paths), 1)
    actuals = ReadDebugEvents(source_files_paths[0])
    self.assertEqual(len(actuals), num_op_creations + 1)
    for i in range(num_op_creations):
      self.assertEqual(actuals[i].graph_op_creation.op_type, "Conv2D")
      self.assertEqual(actuals[i].graph_op_creation.op_name, "Conv2D_%d" % i)
    self.assertEqual(actuals[num_op_creations].debugged_graph.graph_id,
                     "deadbeaf")

  def testConcurrentWritesToNonExecutionFilesWorks(self):
    writer = debug_events_writer.DebugEventsWriter(self.dump_root)

    source_file_state = {"counter": 0, "lock": threading.Lock()}

    def WriteSourceFile():
      source_file = debug_event_pb2.SourceFile()
      with source_file_state["lock"]:
        source_file.file_path = "/home/tf2user/file_%d.py" % source_file_state[
            "counter"]
        source_file_state["counter"] += 1
      writer.WriteSourceFile(source_file)
      # More-frequent-than-necessary concurrent flushing is not recommended,
      # but tolerated.
      writer.FlushNonExecutionFiles()

    stack_frame_state = {"counter": 0, "lock": threading.Lock()}

    def WriteStackFrame():
      stack_frame = debug_event_pb2.StackFrameWithId()
      with stack_frame_state["lock"]:
        stack_frame.id = "stack_frame_%d" % stack_frame_state["counter"]
        stack_frame_state["counter"] += 1
      writer.WriteStackFrameWithId(stack_frame)
      # More-frequent-than-necessary concurrent flushing is not recommended,
      # but tolerated.
      writer.FlushNonExecutionFiles()

    graph_op_state = {"counter": 0, "lock": threading.Lock()}

    def WriteGraphOpCreation():
      graph_op_creation = debug_event_pb2.GraphOpCreation()
      with graph_op_state["lock"]:
        graph_op_creation.op_name = "Op%d" % graph_op_state["counter"]
        graph_op_state["counter"] += 1
      writer.WriteGraphOpCreation(graph_op_creation)
      # More-frequent-than-necessary concurrent flushing is not recommended,
      # but tolerated.
      writer.FlushNonExecutionFiles()

    num_threads = 9
    threads = []
    for i in range(num_threads):
      if i % 3 == 0:
        target = WriteSourceFile
      elif i % 3 == 1:
        target = WriteStackFrame
      else:
        target = WriteGraphOpCreation
      thread = threading.Thread(target=target)
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()

    # Verify the content of the .source_files file.
    source_files_paths = glob.glob(
        os.path.join(self.dump_root, "*.source_files"))
    self.assertEqual(len(source_files_paths), 1)
    actuals = ReadDebugEvents(source_files_paths[0])
    file_paths = sorted([actual.source_file.file_path for actual in actuals])
    self.assertEqual(file_paths, [
        "/home/tf2user/file_0.py", "/home/tf2user/file_1.py",
        "/home/tf2user/file_2.py"
    ])

    # Verify the content of the .stack_frames file.
    stack_frames_paths = glob.glob(
        os.path.join(self.dump_root, "*.stack_frames"))
    self.assertEqual(len(stack_frames_paths), 1)
    actuals = ReadDebugEvents(stack_frames_paths[0])
    stack_frame_ids = sorted(
        [actual.stack_frame_with_id.id for actual in actuals])
    self.assertEqual(stack_frame_ids,
                     ["stack_frame_0", "stack_frame_1", "stack_frame_2"])

    # Verify the content of the .graphs file.
    graphs_paths = glob.glob(os.path.join(self.dump_root, "*.graphs"))
    self.assertEqual(len(graphs_paths), 1)
    actuals = ReadDebugEvents(graphs_paths[0])
    graph_op_names = sorted(
        [actual.graph_op_creation.op_name for actual in actuals])
    self.assertEqual(graph_op_names, ["Op0", "Op1", "Op2"])

  def testWriteExecutionEventsWithCyclicBuffer(self):
    writer = debug_events_writer.DebugEventsWriter(self.dump_root)
    num_execution_events = debug_events_writer.DEFAULT_CYCLIC_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      execution = debug_event_pb2.Execution()
      execution.op_type = "OpType%d" % i
      writer.WriteExecution(execution)

    execution_paths = glob.glob(os.path.join(self.dump_root, "*.execution"))
    self.assertEqual(len(execution_paths), 1)
    actuals = ReadDebugEvents(execution_paths[0])
    # Before FlushExecutionFiles() is called. No data should have been written
    # to the file.
    self.assertEqual(len(actuals), 0)
    writer.FlushExecutionFiles()

    actuals = ReadDebugEvents(execution_paths[0])
    self.assertEqual(
        len(actuals), debug_events_writer.DEFAULT_CYCLIC_BUFFER_SIZE)
    for i in range(debug_events_writer.DEFAULT_CYCLIC_BUFFER_SIZE):
      self.assertEqual(
          actuals[i].execution.op_type,
          "OpType%d" % (i + debug_events_writer.DEFAULT_CYCLIC_BUFFER_SIZE))

  def testWriteExecutionEventsWithoutCyclicBufferBehavior(self):
    # A cyclic buffer size of 0 abolishes the cyclic buffer behavior.
    writer = debug_events_writer.DebugEventsWriter(self.dump_root, 0)
    num_execution_events = debug_events_writer.DEFAULT_CYCLIC_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      execution = debug_event_pb2.Execution()
      execution.op_type = "OpType%d" % i
      writer.WriteExecution(execution)
    writer.FlushExecutionFiles()

    execution_paths = glob.glob(os.path.join(self.dump_root, "*.execution"))
    self.assertEqual(len(execution_paths), 1)
    actuals = ReadDebugEvents(execution_paths[0])
    self.assertEqual(len(actuals), num_execution_events)
    for i in range(num_execution_events):
      self.assertEqual(actuals[i].execution.op_type, "OpType%d" % i)

  def testWriteGraphExecutionTraceEventsWithCyclicBuffer(self):
    writer = debug_events_writer.DebugEventsWriter(self.dump_root)
    num_execution_events = debug_events_writer.DEFAULT_CYCLIC_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      trace = debug_event_pb2.GraphExecutionTrace()
      trace.op_name = "Op%d" % i
      writer.WriteGraphExecutionTrace(trace)

    trace_paths = glob.glob(
        os.path.join(self.dump_root, "*.graph_execution_traces"))
    self.assertEqual(len(trace_paths), 1)
    actuals = ReadDebugEvents(trace_paths[0])
    # Before FlushExecutionFiles() is called. No data should have been written
    # to the file.
    self.assertEqual(len(actuals), 0)

    writer.FlushExecutionFiles()
    actuals = ReadDebugEvents(trace_paths[0])
    self.assertEqual(
        len(actuals), debug_events_writer.DEFAULT_CYCLIC_BUFFER_SIZE)

    for i in range(debug_events_writer.DEFAULT_CYCLIC_BUFFER_SIZE):
      self.assertEqual(
          actuals[i].graph_execution_trace.op_name,
          "Op%d" % (i + debug_events_writer.DEFAULT_CYCLIC_BUFFER_SIZE))

  def testWriteGraphExecutionTraceEventsWithoutCyclicBufferBehavior(self):
    # A cyclic buffer size of 0 abolishes the cyclic buffer behavior.
    writer = debug_events_writer.DebugEventsWriter(self.dump_root, 0)
    num_execution_events = debug_events_writer.DEFAULT_CYCLIC_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      trace = debug_event_pb2.GraphExecutionTrace()
      trace.op_name = "Op%d" % i
      writer.WriteGraphExecutionTrace(trace)
    writer.FlushExecutionFiles()

    trace_paths = glob.glob(
        os.path.join(self.dump_root, "*.graph_execution_traces"))
    self.assertEqual(len(trace_paths), 1)
    actuals = ReadDebugEvents(trace_paths[0])
    self.assertEqual(len(actuals), num_execution_events)
    for i in range(num_execution_events):
      self.assertEqual(actuals[i].graph_execution_trace.op_name, "Op%d" % i)

  def testConcurrentWritesToExecutionFiles(self):
    cyclic_buffer_size = 5
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   cyclic_buffer_size)

    execution_state = {"counter": 0, "lock": threading.Lock()}

    def WriteExecution():
      execution = debug_event_pb2.Execution()
      with execution_state["lock"]:
        execution.op_type = "OpType%d" % execution_state["counter"]
        execution_state["counter"] += 1
      writer.WriteExecution(execution)

    graph_execution_trace_state = {"counter": 0, "lock": threading.Lock()}

    def WriteGraphExecutionTrace():
      trace = debug_event_pb2.GraphExecutionTrace()
      with graph_execution_trace_state["lock"]:
        trace.op_name = "Op%d" % graph_execution_trace_state["counter"]
        graph_execution_trace_state["counter"] += 1
      writer.WriteGraphExecutionTrace(trace)

    threads = []
    for i in range(cyclic_buffer_size * 4):
      if i % 2 == 0:
        target = WriteExecution
      else:
        target = WriteGraphExecutionTrace
      thread = threading.Thread(target=target)
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()
    writer.FlushExecutionFiles()

    # Verify the content of the .execution file.
    execution_paths = glob.glob(os.path.join(self.dump_root, "*.execution"))
    self.assertEqual(len(execution_paths), 1)
    actuals = ReadDebugEvents(execution_paths[0])
    op_types = sorted([actual.execution.op_type for actual in actuals])
    self.assertEqual(len(op_types), cyclic_buffer_size)
    self.assertEqual(len(op_types), len(set(op_types)))

    # Verify the content of the .execution file.
    traces_paths = glob.glob(
        os.path.join(self.dump_root, "*.graph_execution_traces"))
    self.assertEqual(len(traces_paths), 1)
    actuals = ReadDebugEvents(traces_paths[0])
    op_names = sorted(
        [actual.graph_execution_trace.op_name for actual in actuals])
    self.assertEqual(len(op_names), cyclic_buffer_size)
    self.assertEqual(len(op_names), len(set(op_names)))


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
