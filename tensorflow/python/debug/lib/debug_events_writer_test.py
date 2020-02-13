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
import json as json_lib
import os
import threading
import time

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import dumping_callback_test_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.platform import googletest


class DebugEventsWriterTest(dumping_callback_test_lib.DumpingCallbackTestBase):

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
    self._readAndCheckMetadataFile()

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

    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      actuals = list(item.debug_event.source_file
                     for item in reader.source_files_iterator())
      self.assertLen(actuals, num_protos)
      for i in range(num_protos):
        self.assertEqual(actuals[i].file_path, "/home/tf2user/main.py")
        self.assertEqual(actuals[i].host_name, "machine.cluster")
        self.assertEqual(actuals[i].lines, ["print(%d)" % i])

      actuals = list(item.debug_event.stack_frame_with_id
                     for item in reader.stack_frames_iterator())
      self.assertLen(actuals, num_protos)
      for i in range(num_protos):
        self.assertEqual(actuals[i].id, "stack_%d" % i)
        self.assertEqual(actuals[i].file_line_col.file_index, i * 10)

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

    reader = debug_events_reader.DebugEventsReader(self.dump_root)
    actuals = list(item.debug_event for item in reader.graphs_iterator())
    self.assertLen(actuals, num_op_creations + 1)
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
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      source_files_iter = reader.source_files_iterator()
      actuals = list(item.debug_event.source_file for item in source_files_iter)
      file_paths = sorted([actual.file_path for actual in actuals])
      self.assertEqual(file_paths, [
          "/home/tf2user/file_0.py", "/home/tf2user/file_1.py",
          "/home/tf2user/file_2.py"
      ])

    # Verify the content of the .stack_frames file.
    actuals = list(item.debug_event.stack_frame_with_id
                   for item in reader.stack_frames_iterator())
    stack_frame_ids = sorted([actual.id for actual in actuals])
    self.assertEqual(stack_frame_ids,
                     ["stack_frame_0", "stack_frame_1", "stack_frame_2"])

    # Verify the content of the .graphs file.
    actuals = list(item.debug_event.graph_op_creation
                   for item in reader.graphs_iterator())
    graph_op_names = sorted([actual.op_name for actual in actuals])
    self.assertEqual(graph_op_names, ["Op0", "Op1", "Op2"])

  def testWriteAndReadMetadata(self):
    t0 = time.time()
    writer = debug_events_writer.DebugEventsWriter(self.dump_root)
    writer.Close()
    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      self.assertIsInstance(reader.starting_wall_time(), float)
      self.assertGreaterEqual(reader.starting_wall_time(), t0)
      self.assertEqual(reader.tensorflow_version(), versions.__version__)

  def testWriteExecutionEventsWithCircularBuffer(self):
    writer = debug_events_writer.DebugEventsWriter(self.dump_root)
    num_execution_events = debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      execution = debug_event_pb2.Execution()
      execution.op_type = "OpType%d" % i
      writer.WriteExecution(execution)

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      # Before FlushExecutionFiles() is called. No data should have been written
      # to the file.
      reader.update()
      self.assertFalse(reader.executions())

      writer.FlushExecutionFiles()
      reader.update()
      executions = reader.executions()
      for i, execution in enumerate(executions):
        self.assertEqual(
            execution.op_type,
            "OpType%d" % (i + debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE))

  def testWriteExecutionEventsWithoutCircularBufferBehavior(self):
    # A circular buffer size of 0 abolishes the circular buffer behavior.
    writer = debug_events_writer.DebugEventsWriter(self.dump_root, 0)
    num_execution_events = debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      execution = debug_event_pb2.Execution()
      execution.op_type = "OpType%d" % i
      writer.WriteExecution(execution)
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      executions = reader.executions()
      self.assertLen(executions, num_execution_events)
      for i, execution in enumerate(executions):
        self.assertEqual(execution.op_type, "OpType%d" % i)

  def testWriteGraphExecutionTraceEventsWithCircularBuffer(self):
    writer = debug_events_writer.DebugEventsWriter(self.dump_root)
    num_execution_events = debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      trace = debug_event_pb2.GraphExecutionTrace()
      trace.op_name = "Op%d" % i
      writer.WriteGraphExecutionTrace(trace)

    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      actuals = list(reader.graph_execution_traces_iterator())
      # Before FlushExecutionFiles() is called. No data should have been written
      # to the file.
      self.assertEqual(len(actuals), 0)

      writer.FlushExecutionFiles()
      actuals = list(item.debug_event.graph_execution_trace
                     for item in reader.graph_execution_traces_iterator())
      self.assertLen(actuals, debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE)
      for i in range(debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE):
        self.assertEqual(
            actuals[i].op_name,
            "Op%d" % (i + debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE))

  def testWriteGraphExecutionTraceEventsWithoutCircularBufferBehavior(self):
    # A circular buffer size of 0 abolishes the circular buffer behavior.
    writer = debug_events_writer.DebugEventsWriter(self.dump_root, 0)
    num_execution_events = debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      trace = debug_event_pb2.GraphExecutionTrace()
      trace.op_name = "Op%d" % i
      writer.WriteGraphExecutionTrace(trace)
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      actuals = list(item.debug_event.graph_execution_trace
                     for item in reader.graph_execution_traces_iterator())
    self.assertLen(actuals, num_execution_events)
    for i in range(num_execution_events):
      self.assertEqual(actuals[i].op_name, "Op%d" % i)

  def testConcurrentWritesToExecutionFiles(self):
    circular_buffer_size = 5
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   circular_buffer_size)
    debugged_graph = debug_event_pb2.DebuggedGraph(graph_id="graph1",
                                                   graph_name="graph1")
    writer.WriteDebuggedGraph(debugged_graph)

    execution_state = {"counter": 0, "lock": threading.Lock()}

    def WriteExecution():
      execution = debug_event_pb2.Execution()
      with execution_state["lock"]:
        execution.op_type = "OpType%d" % execution_state["counter"]
        execution_state["counter"] += 1
      writer.WriteExecution(execution)

    graph_execution_trace_state = {"counter": 0, "lock": threading.Lock()}

    def WriteGraphExecutionTrace():
      with graph_execution_trace_state["lock"]:
        op_name = "Op%d" % graph_execution_trace_state["counter"]
        graph_op_creation = debug_event_pb2.GraphOpCreation(
            op_type="FooOp", op_name=op_name, graph_id="graph1")
        trace = debug_event_pb2.GraphExecutionTrace(
            op_name=op_name, tfdbg_context_id="graph1")
        graph_execution_trace_state["counter"] += 1
      writer.WriteGraphOpCreation(graph_op_creation)
      writer.WriteGraphExecutionTrace(trace)

    threads = []
    for i in range(circular_buffer_size * 4):
      if i % 2 == 0:
        target = WriteExecution
      else:
        target = WriteGraphExecutionTrace
      thread = threading.Thread(target=target)
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      # Verify the content of the .execution file.
      executions = reader.executions()
      executed_op_types = [execution.op_type for execution in executions]
      self.assertLen(executed_op_types, circular_buffer_size)
      self.assertLen(executed_op_types, len(set(executed_op_types)))

      # Verify the content of the .graph_execution_traces file.
      op_names = [trace.op_name for trace in reader.graph_execution_traces()]
      self.assertLen(op_names, circular_buffer_size)
      self.assertLen(op_names, len(set(op_names)))


class DataObjectsTest(test_util.TensorFlowTestCase):

  def jsonRoundTripCheck(self, obj):
    self.assertEqual(
        json_lib.dumps(json_lib.loads(json_lib.dumps(obj)), sort_keys=True),
        json_lib.dumps(obj, sort_keys=True))

  def testExecutionDigestWithNoOutputToJson(self):
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 5678, "FooOp", output_tensor_device_ids=None)
    json = execution_digest.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["output_tensor_device_ids"], None)

  def testExecutionDigestWithTwoOutputsToJson(self):
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 5678, "FooOp", output_tensor_device_ids=[1357, 2468])
    json = execution_digest.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["output_tensor_device_ids"], (1357, 2468))

  def testExecutionNoGraphNoInputToJson(self):
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 5678, "FooOp", output_tensor_device_ids=[1357])
    execution = debug_events_reader.Execution(
        execution_digest,
        "localhost",
        ("a1", "b2"),
        debug_event_pb2.TensorDebugMode.CURT_HEALTH,
        graph_id=None,
        input_tensor_ids=None,
        output_tensor_ids=[2468],
        debug_tensor_values=([1, 0],))
    json = execution.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["output_tensor_device_ids"], (1357,))
    self.assertEqual(json["host_name"], "localhost")
    self.assertEqual(json["stack_frame_ids"], ("a1", "b2"))
    self.assertEqual(json["tensor_debug_mode"],
                     debug_event_pb2.TensorDebugMode.CURT_HEALTH)
    self.assertIsNone(json["graph_id"])
    self.assertIsNone(json["input_tensor_ids"])
    self.assertEqual(json["output_tensor_ids"], (2468,))
    self.assertEqual(json["debug_tensor_values"], ([1, 0],))

  def testExecutionNoGraphNoInputButWithOutputToJson(self):
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 5678, "FooOp", output_tensor_device_ids=[1357])
    execution = debug_events_reader.Execution(
        execution_digest,
        "localhost",
        ("a1", "b2"),
        debug_event_pb2.TensorDebugMode.FULL_HEALTH,
        graph_id="abcd",
        input_tensor_ids=[13, 37],
        output_tensor_ids=None,
        debug_tensor_values=None)
    json = execution.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["output_tensor_device_ids"], (1357,))
    self.assertEqual(json["host_name"], "localhost")
    self.assertEqual(json["stack_frame_ids"], ("a1", "b2"))
    self.assertEqual(json["tensor_debug_mode"],
                     debug_event_pb2.TensorDebugMode.FULL_HEALTH)
    self.assertEqual(json["graph_id"], "abcd")
    self.assertEqual(json["input_tensor_ids"], (13, 37))
    self.assertIsNone(json["output_tensor_ids"])
    self.assertIsNone(json["debug_tensor_values"])

  def testGraphOpCreationDigestNoInputNoDeviceNameToJson(self):
    op_creation_digest = debug_events_reader.GraphOpCreationDigest(
        1234, 5678, "deadbeef", "FooOp", "Model_1/Foo_2",
        [135], input_names=None, device_name=None)
    json = op_creation_digest.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["graph_id"], "deadbeef")
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["op_name"], "Model_1/Foo_2")
    self.assertEqual(json["output_tensor_ids"], (135,))
    self.assertIsNone(json["input_names"])
    self.assertIsNone(json["device_name"])

  def testGraphOpCreationDigestWithInputsAndDeviceNameToJson(self):
    op_creation_digest = debug_events_reader.GraphOpCreationDigest(
        1234, 5678, "deadbeef", "FooOp", "Model_1/Foo_2",
        [135], input_names=["Bar_1", "Qux_2"], device_name="/device:GPU:0")
    json = op_creation_digest.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["graph_id"], "deadbeef")
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["op_name"], "Model_1/Foo_2")
    self.assertEqual(json["output_tensor_ids"], (135,))
    self.assertEqual(json["input_names"], ("Bar_1", "Qux_2"))
    self.assertEqual(json["device_name"], "/device:GPU:0")

  def testGraphExecutionTraceDigestToJson(self):
    trace_digest = debug_events_reader.GraphExecutionTraceDigest(
        1234, 5678, "FooOp", "Model_1/Foo_2", 1, "deadbeef")
    json = trace_digest.to_json()
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["op_name"], "Model_1/Foo_2")
    self.assertEqual(json["output_slot"], 1)
    self.assertEqual(json["graph_id"], "deadbeef")

  def testGraphExecutionTraceWithTensorDebugValueAndDeviceNameToJson(self):
    trace_digest = debug_events_reader.GraphExecutionTraceDigest(
        1234, 5678, "FooOp", "Model_1/Foo_2", 1, "deadbeef")
    trace = debug_events_reader.GraphExecutionTrace(
        trace_digest, ["g1", "g2", "deadbeef"],
        debug_event_pb2.TensorDebugMode.CURT_HEALTH,
        debug_tensor_value=[3, 1], device_name="/device:GPU:0")
    json = trace.to_json()
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["op_name"], "Model_1/Foo_2")
    self.assertEqual(json["output_slot"], 1)
    self.assertEqual(json["graph_id"], "deadbeef")
    self.assertEqual(json["graph_ids"], ("g1", "g2", "deadbeef"))
    self.assertEqual(json["tensor_debug_mode"],
                     debug_event_pb2.TensorDebugMode.CURT_HEALTH)
    self.assertEqual(json["debug_tensor_value"], (3, 1))
    self.assertEqual(json["device_name"], "/device:GPU:0")

  def testGraphExecutionTraceNoTensorDebugValueNoDeviceNameToJson(self):
    trace_digest = debug_events_reader.GraphExecutionTraceDigest(
        1234, 5678, "FooOp", "Model_1/Foo_2", 1, "deadbeef")
    trace = debug_events_reader.GraphExecutionTrace(
        trace_digest, ["g1", "g2", "deadbeef"],
        debug_event_pb2.TensorDebugMode.NO_TENSOR,
        debug_tensor_value=None, device_name=None)
    json = trace.to_json()
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["op_name"], "Model_1/Foo_2")
    self.assertEqual(json["output_slot"], 1)
    self.assertEqual(json["graph_id"], "deadbeef")
    self.assertEqual(json["graph_ids"], ("g1", "g2", "deadbeef"))
    self.assertEqual(json["tensor_debug_mode"],
                     debug_event_pb2.TensorDebugMode.NO_TENSOR)
    self.assertIsNone(json["debug_tensor_value"])
    self.assertIsNone(json["device_name"])


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
