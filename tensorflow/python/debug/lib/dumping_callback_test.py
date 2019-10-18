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
"""Unit tests for tfdbg v2 dumping callback."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import socket
import tempfile
import threading

import numpy as np

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.keras import models
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


def _create_simple_recurrent_keras_model():
  """Create a simple tf.keras model containing a recurrent layer for testing."""
  model = models.Sequential()
  model.add(recurrent_v2.LSTM(
      10,
      input_shape=[8, 4],
      kernel_initializer="zeros",
      recurrent_initializer="zeros"))
  model.add(core.Dense(1, kernel_initializer="zeros"))
  model.compile(loss="mse", optimizer="sgd")
  return model


class TracingCallbackTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(TracingCallbackTest, self).setUp()
    self.dump_root = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self.dump_root):
      shutil.rmtree(self.dump_root, ignore_errors=True)
    dumping_callback.disable_dumping()
    super(TracingCallbackTest, self).tearDown()

  def _readAndCheckMetadataFile(self):
    """Read and check the .metadata debug-events file."""
    reader = debug_events_reader.DebugEventsDir(self.dump_root)
    metadata_iter = reader.metadata_iterator()
    metadata = next(metadata_iter).debug_metadata
    self.assertEqual(metadata.tensorflow_version, versions.__version__)
    self.assertTrue(metadata.file_version.startswith("debug.Event"))

  def _readAndCheckSourceFilesAndStackFrames(self):
    """Read and verify the .source_files & .stack_frames debug-event files.

    Returns:
      A dict mapping stack frame IDs to stack frames (FileLineCol).
    """
    reader = debug_events_reader.DebugEventsDir(self.dump_root)
    # Check the content of the .source_files file.
    source_files_iter = reader.source_files_iterator()
    source_file_paths = []
    prev_wall_time = 1
    for debug_event in source_files_iter:
      self.assertGreaterEqual(debug_event.wall_time, prev_wall_time)
      prev_wall_time = debug_event.wall_time
      source_file = debug_event.source_file
      self.assertEqual(source_file.host_name, socket.gethostname())
      self.assertTrue(source_file.file_path)
      if source_file.lines:
        self.assertTrue(os.path.isfile(source_file.file_path))
      source_file_paths.append(source_file.file_path)
    # Assert the file paths are unique.
    self.assertEqual(len(source_file_paths), len(set(source_file_paths)))

    # Check the content of the .stack_frames file.
    stack_frame_by_id = dict()  # A map from ID to stack frame.
    stack_frames_iter = reader.stack_frames_iterator()
    prev_wall_time = 0
    for debug_event in stack_frames_iter:
      self.assertGreaterEqual(debug_event.wall_time, prev_wall_time)
      prev_wall_time = debug_event.wall_time
      stack_frame_with_id = debug_event.stack_frame_with_id
      stack_frame_id = stack_frame_with_id.id
      file_line_col = stack_frame_with_id.file_line_col
      self.assertTrue(stack_frame_id)
      self.assertNotIn(stack_frame_id, stack_frame_by_id,
                       "Duplicate stack frame ID: %s" % id)
      stack_frame_by_id[stack_frame_id] = (file_line_col.file_index,
                                           file_line_col.line,
                                           file_line_col.func)
      self.assertGreaterEqual(file_line_col.file_index, 0)
      self.assertLess(file_line_col.file_index, len(source_file_paths))
      self.assertTrue(file_line_col.line)  # Line numbers are 1-based.
      self.assertTrue(file_line_col.func)
    # Assert the stack frames are unique.
    self.assertEqual(
        len(stack_frame_by_id.values()), len(set(stack_frame_by_id.values())))
    return stack_frame_by_id

  def _readAndCheckGraphsFile(self, stack_frame_by_id):
    """Read and verify the content of the .graphs debug-event file.

    Args:
      stack_frame_by_id: A dict mapping unique string IDs to stack frames.
        It is used by this method to look up stack frames.

    Returns:
      context_ids: IDs of op creation contexts (e.g., TensorFlow graphs), as a
        `list` of `str`s.
      op_types: Types of the ops that are created, as a `list` of `str`s with
        the same length as `context_ids`.
      op_name_to_op_type: A `dict` mapping op name to op type.
    """
    reader = debug_events_reader.DebugEventsDir(self.dump_root)
    graphs_iter = reader.graphs_iterator()
    prev_wall_time = 0
    op_types = []
    op_name_to_op_type = dict()
    context_ids = set()
    for debug_event in graphs_iter:
      self.assertGreaterEqual(debug_event.wall_time, prev_wall_time)
      prev_wall_time = debug_event.wall_time
      graph_op_creation = debug_event.graph_op_creation
      self.assertTrue(graph_op_creation.op_type)
      op_types.append(graph_op_creation.op_type)
      self.assertTrue(graph_op_creation.op_name)
      op_name_to_op_type[graph_op_creation.op_name] = graph_op_creation.op_type
      self.assertTrue(graph_op_creation.graph_id)
      context_ids.add(graph_op_creation.graph_id)
      self.assertTrue(graph_op_creation.code_location)
      for stack_frame_id in graph_op_creation.code_location.stack_frame_ids:
        self.assertIn(stack_frame_id, stack_frame_by_id)
    return context_ids, op_types, op_name_to_op_type

  def _readAndCheckExecutionFile(self):
    """Read and verify the content of the .execution debug-event file.

    Returns:
      executed_op_types: Types of ops that are created, as a `list` of `str`.
      input_tensor_ids: Input tensor IDs for each of the ops executed, as a
        `list` of `list` of `int`s, with the same length as `executed_op_types`.
      output_tensor_ids: Output tensor IDs for each of the ops executed, as a
        `list` of `list` of `int`s, with the same length as `executed_op_types`.
      tensor_debug_modes: Tensor debug modes used to instrument each of ops
        executed.
    """
    reader = debug_events_reader.DebugEventsDir(self.dump_root)
    execution_iter = reader.execution_iterator()
    prev_wall_time = 1
    executed_op_types = []
    input_tensor_ids = []
    output_tensor_ids = []
    tensor_debug_modes = []
    for debug_event in execution_iter:
      self.assertGreaterEqual(debug_event.wall_time, prev_wall_time)
      prev_wall_time = debug_event.wall_time
      execution = debug_event.execution
      executed_op_types.append(execution.op_type)
      input_tensor_ids.append(execution.input_tensor_ids)
      output_tensor_ids.append(execution.output_tensor_ids)
      tensor_debug_modes.append(execution.tensor_debug_mode)

    # TODO(cais): When tensor debug modes other than NO_TENSOR is supported,
    # return tensor_values as well.
    return (executed_op_types, input_tensor_ids, output_tensor_ids,
            tensor_debug_modes)

  def _readAndCheckGraphExecutionTracesFile(self, context_ids):
    """Read & verify the content of the .graph_execution_trace debug-event file.

    Args:
      context_ids: Op-creation context IDs from _readAndCheckGraphsFile().

    Returns:
      op_names: Names of the ops that are executed, as a `list` of `str`s.
      output_slots: Output slots, as a `list` of `int`s, with the same length as
        `op_names`. In other words, for an executed op with N output tensors,
        there will be N entries in this `list` and in `op_names`, at
        corresponding indices.
      tensor_values: Tensor values or their concise summaries, depending on
        TensorDebugMode.
    """
    reader = debug_events_reader.DebugEventsDir(self.dump_root)
    graph_execution_traces_iter = reader.graph_execution_traces_iterator()
    op_names = []
    output_slots = []
    tensor_values = []
    for debug_event in graph_execution_traces_iter:
      self.assertGreaterEqual(debug_event.wall_time, 0)
      graph_execution_trace = debug_event.graph_execution_trace
      op_names.append(graph_execution_trace.op_name)
      # All the ops in the graph have only one output.
      self.assertTrue(graph_execution_trace.tfdbg_context_id)
      self.assertIn(graph_execution_trace.tfdbg_context_id, context_ids)
      output_slots.append(graph_execution_trace.output_slot)
      # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought to
      # be an empty float32 tensor.
      tensor_values.append(
          tensor_util.MakeNdarray(graph_execution_trace.tensor_proto))
    return op_names, output_slots, tensor_values

  def testInvalidTensorDebugModeCausesError(self):
    with self.assertRaisesRegexp(
        ValueError,
        r"Invalid value in tensor_debug_mode \(\'NONSENSICAL\'\).*"
        r"Valid options.*NO_TENSOR.*"):
      dumping_callback.enable_dumping(
          self.dump_root, tensor_debug_mode="NONSENSICAL")

  def testDisablingTracingCallbackWithoutEnablingFirstIsTolerated(self):
    dumping_callback.disable_dumping()

  def testPureEagerOpExecution(self):
    """Test catching Infinity in eager op execution: float32."""
    writer = dumping_callback.enable_dumping(
        self.dump_root, tensor_debug_mode="NO_TENSOR")

    x = constant_op.constant(10.0)
    zero = constant_op.constant(0.0)
    one = constant_op.constant(1.0)
    two = constant_op.constant(2.0)
    three = constant_op.constant(3.0)
    # Use Collatz conjecture as a test case.
    while x > one:
      if math_ops.equal(x % two, zero):
        x = x / two
      else:
        x = x * three + one

    writer.FlushNonExecutionFiles()
    self._readAndCheckMetadataFile()
    stack_frame_by_id = self._readAndCheckSourceFilesAndStackFrames()

    # Before FlushExecutionFiles() is called, the .execution file should be
    # empty.
    reader = debug_events_reader.DebugEventsDir(self.dump_root)
    execution_iter = reader.execution_iterator()
    with self.assertRaises(StopIteration):
      next(execution_iter)

    # After the flushing, the .execution file should hold the appropriate
    # contents.
    writer.FlushExecutionFiles()
    execution_iter = reader.execution_iterator()
    prev_wall_time = 1
    executed_op_types = []
    for debug_event in execution_iter:
      self.assertGreaterEqual(debug_event.wall_time, prev_wall_time)
      prev_wall_time = debug_event.wall_time
      execution = debug_event.execution
      executed_op_types.append(execution.op_type)
      self.assertTrue(execution.input_tensor_ids)
      self.assertTrue(execution.output_tensor_ids)
      # Due to the default NO_TENSOR tensor debug mode, tensor_protos ought to
      # be empty.
      self.assertFalse(execution.tensor_protos)
      # Verify the code_location field.
      self.assertTrue(execution.code_location.stack_frame_ids)
      for stack_frame_id in execution.code_location.stack_frame_ids:
        self.assertIn(stack_frame_id, stack_frame_by_id)
    self.assertEqual(
        executed_op_types,
        [
            "Greater",
            "FloorMod",
            "Equal",
            "RealDiv",  # 10 --> 5
            "Greater",
            "FloorMod",
            "Equal",
            "Mul",
            "AddV2",  # 5 --> 16
            "Greater",
            "FloorMod",
            "Equal",
            "RealDiv",  # 16 --> 8
            "Greater",
            "FloorMod",
            "Equal",
            "RealDiv",  # 8 --> 4
            "Greater",
            "FloorMod",
            "Equal",
            "RealDiv",  # 4 --> 2
            "Greater",
            "FloorMod",
            "Equal",
            "RealDiv",  # 2 --> 1
            "Greater"
        ])

    # Due to the pure eager op execution, the .graph file and the
    # .graph_execution_traces file ought to be empty.
    graphs_iterator = reader.graphs_iterator()
    with self.assertRaises(StopIteration):
      next(graphs_iterator)
    graph_trace_iter = reader.graph_execution_traces_iterator()
    with self.assertRaises(StopIteration):
      next(graph_trace_iter)

  @test_util.run_in_graph_and_eager_modes
  def testNestedFunctionExecutionWithoutControlFlow(self):
    writer = dumping_callback.enable_dumping(self.dump_root)

    @def_function.function
    def log_sum(x, y):
      return math_ops.log(x + y)

    @def_function.function
    def sin1p_log_sum(x, y):
      return math_ops.sin(1.0 + log_sum(x, y))

    x = constant_op.constant(2.0)
    y = constant_op.constant(3.0)
    self.assertAllClose(sin1p_log_sum(x, y), np.sin(1.0 + np.log(5.0)))
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    if context.executing_eagerly():
      # NOTE(b/142486213): Execution of the TF function happens with
      # Session.run() in v1 graph mode, so doesn't get logged to the
      # .execution file.
      executed_op_types, _, _, _ = self._readAndCheckExecutionFile()
      self.assertEqual(len(executed_op_types), 1)
      self.assertIn("sin1p_log_sum", executed_op_types[0])

    stack_frame_by_id = self._readAndCheckSourceFilesAndStackFrames()
    (context_ids, op_types,
     op_name_to_op_type) = self._readAndCheckGraphsFile(stack_frame_by_id)
    self.assertIn("AddV2", op_types)
    self.assertIn("Log", op_types)
    self.assertIn("Sin", op_types)

    (op_names, _,
     tensor_values) = self._readAndCheckGraphExecutionTracesFile(context_ids)
    executed_op_types = [op_name_to_op_type[op_name] for op_name in op_names]
    self.assertEqual(executed_op_types, ["AddV2", "Log", "AddV2", "Sin"])

    # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought to
    # be an empty float32 tensor.
    for tensor_value in tensor_values:
      self.assertEqual(tensor_value.dtype, np.float32)
      self.assertEqual(tensor_value.shape, (0,))

  @test_util.run_in_graph_and_eager_modes
  def testFunctionExecutionWithControlFlow(self):
    writer = dumping_callback.enable_dumping(self.dump_root)

    @def_function.function
    def iterative_doubling(x, times):
      i = constant_op.constant(0, dtype=dtypes.int32)
      while i < times:
        x = x * 2.0
        i += 1
      return x

    x = constant_op.constant(0.5, dtype=dtypes.float32)
    times = constant_op.constant(4, dtype=dtypes.int32)
    self.assertAllClose(self.evaluate(iterative_doubling(x, times)), 8.0)

    writer.FlushNonExecutionFiles()
    stack_frame_by_id = self._readAndCheckSourceFilesAndStackFrames()

    # Verify the content of the .graphs file.
    context_ids, op_types, op_name_to_op_type = (
        self._readAndCheckGraphsFile(stack_frame_by_id))
    self.assertIn("Less", op_types)
    self.assertIn("Mul", op_types)
    self.assertIn("AddV2", op_types)

    # Before FlushExecutionFiles() is called, the .execution and
    # .graph_execution_traces files should be both empty.
    reader = debug_events_reader.DebugEventsDir(self.dump_root)
    execution_iter = reader.execution_iterator()
    graph_execution_traces_iter = reader.graph_execution_traces_iterator()
    with self.assertRaises(StopIteration):
      next(execution_iter)
    with self.assertRaises(StopIteration):
      next(graph_execution_traces_iter)

    # TODO(cais): Backport execution instrumentation to tf.Session.
    writer.FlushExecutionFiles()
    # After the flushing, the .execution file should hold the appropriate
    # contents.
    if context.executing_eagerly():
      (executed_op_types, input_tensor_ids, output_tensor_ids,
       tensor_debug_modes) = self._readAndCheckExecutionFile()
      # NOTE(b/142486213): Execution of the TF function happens with
      # Session.run() in v1 graph mode, hence it doesn't get logged to the
      # .execution file.
      self.assertEqual(len(executed_op_types), 1)
      self.assertIn("iterative_doubling", executed_op_types[0])
      self.assertEqual(len(input_tensor_ids[0]), 2)
      self.assertEqual(len(output_tensor_ids[0]), 1)
      self.assertEqual(tensor_debug_modes[0],
                       debug_event_pb2.TensorDebugMode.NO_TENSOR)

    (op_names, output_slots,
     tensor_values) = self._readAndCheckGraphExecutionTracesFile(context_ids)
    executed_op_types = [op_name_to_op_type[op_name] for op_name in op_names]
    # The Less op should have been executed 5 times.
    self.assertEqual(executed_op_types.count("Less"), 5)
    # The last executed op should be Less.
    self.assertEqual(executed_op_types[-1], "Less")
    # The Mul op should have been executed 4 times.
    self.assertEqual(executed_op_types.count("Mul"), 4)
    # The AddV2 op should have been run, but we refrain from asserting on how
    # many times it's executed.
    self.assertIn("AddV2", executed_op_types)
    for output_slot in output_slots:
      self.assertEqual(output_slot, 0)
    # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought to
    # be an empty float32 tensor.
    for tensor_value in tensor_values:
      self.assertEqual(tensor_value.dtype, np.float32)
      self.assertEqual(tensor_value.shape, (0,))

  def testCallingEnableTracingTwiceWithTheSameDumpRootIsIdempotent(self):
    dumping_callback.enable_dumping(self.dump_root)
    writer = dumping_callback.enable_dumping(self.dump_root)

    x = constant_op.constant([10.0, 12.0, 10.0])
    for _ in range(2):
      array_ops.unique(x)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    reader = debug_events_reader.DebugEventsDir(self.dump_root)
    execution_iter = reader.execution_iterator()
    for _ in range(2):
      debug_event = next(execution_iter)
      self.assertGreater(debug_event.wall_time, 0)
      execution = debug_event.execution
      self.assertEqual(execution.op_type, "Unique")
      self.assertEqual(execution.num_outputs, 2)
      self.assertTrue(execution.code_location)
    with self.assertRaises(StopIteration):
      next(execution_iter)

  def testCallingEnableTracingTwiceWithDifferentDumpRootsOverwrites(self):
    dumping_callback.enable_dumping(self.dump_root)
    new_dump_root = self.dump_root + "_new_dump_root"
    writer = dumping_callback.enable_dumping(new_dump_root)

    x = constant_op.constant([10.0, 12.0, 10.0])
    for _ in range(2):
      array_ops.unique(x)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    reader = debug_events_reader.DebugEventsDir(new_dump_root)
    execution_iter = reader.execution_iterator()
    for _ in range(2):
      debug_event = next(execution_iter)
      self.assertGreater(debug_event.wall_time, 0)
      execution = debug_event.execution
      self.assertEqual(execution.op_type, "Unique")
      self.assertEqual(execution.num_outputs, 2)
      self.assertTrue(execution.code_location)
    with self.assertRaises(StopIteration):
      next(execution_iter)

    old_dump_root_reader = debug_events_reader.DebugEventsDir(self.dump_root)
    execution_iter = old_dump_root_reader.execution_iterator()
    # The old dump root shouldn't have been written to.
    with self.assertRaises(StopIteration):
      next(execution_iter)

  def testDisableTracingWorks(self):
    writer = dumping_callback.enable_dumping(self.dump_root)
    dumping_callback.disable_dumping()

    x = constant_op.constant([10.0, 12.0, 10.0])
    for _ in range(2):
      array_ops.unique(x)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    reader = debug_events_reader.DebugEventsDir(self.dump_root)
    source_files_iter = reader.source_files_iterator()
    stack_frames_iter = reader.stack_frames_iterator()
    execution_iter = reader.execution_iterator()
    # No source-file, stack-frame or execution data should have been dumped.
    with self.assertRaises(StopIteration):
      next(source_files_iter)
    with self.assertRaises(StopIteration):
      next(stack_frames_iter)
    with self.assertRaises(StopIteration):
      next(execution_iter)

  def testMultiThreadedExecution(self):
    writer = dumping_callback.enable_dumping(self.dump_root)
    x = variables.Variable(10.0, dtype=dtypes.float32)
    y = variables.Variable(3.0, dtype=dtypes.float32)

    @def_function.function
    def increase_x():
      return x.assign_add(y * 2.0)

    increase_x()

    num_threads = 3
    threads = []
    for _ in range(num_threads):
      threads.append(threading.Thread(target=increase_x))
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    # 10 --> 16 --> 22 --> 28 --> 34.
    self.assertAllClose(x.read_value(), 34.0)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    stack_frame_by_id = self._readAndCheckSourceFilesAndStackFrames()
    reader = debug_events_reader.DebugEventsDir(self.dump_root)
    execution_iter = reader.execution_iterator()
    prev_wall_time = 1
    for debug_event in execution_iter:
      self.assertGreaterEqual(debug_event.wall_time, prev_wall_time)
      prev_wall_time = debug_event.wall_time

    (context_ids, _,
     op_name_to_op_type) = self._readAndCheckGraphsFile(stack_frame_by_id)

    (op_names, output_slots,
     tensor_values) = self._readAndCheckGraphExecutionTracesFile(context_ids)
    executed_op_types = [op_name_to_op_type[op_name] for op_name in op_names]
    self.assertEqual(executed_op_types.count("Mul"), 1 + num_threads)
    self.assertEqual(
        executed_op_types.count("ReadVariableOp"), 2 * (1 + num_threads))
    for output_slot in output_slots:
      self.assertEqual(output_slot, 0)
    for tensor_value in tensor_values:
      self.assertEqual(tensor_value.dtype, np.float32)
      self.assertEqual(tensor_value.shape, (0,))

  @test_util.run_in_graph_and_eager_modes
  def testSimpleKerasRecurrentModelPredict(self):
    writer = dumping_callback.enable_dumping(self.dump_root)
    model = _create_simple_recurrent_keras_model()
    xs = np.ones([5, 8, 4])
    self.assertAllClose(model.predict(xs), np.zeros([5, 1]))

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    stack_frame_by_id = self._readAndCheckSourceFilesAndStackFrames()
    (context_ids, op_types,
     op_name_to_op_type) = self._readAndCheckGraphsFile(stack_frame_by_id)
    # Simply assert that graph are recorded and refrain from asserting on the
    # internal details of the Keras model.
    self.assertTrue(context_ids)
    self.assertTrue(op_types)
    self.assertTrue(op_name_to_op_type)

    if context.executing_eagerly():
      # NOTE(b/142486213): Execution of the TF function happens with
      # Session.run() in v1 graph mode, hence it doesn't get logged to the
      # .execution file.
      executed_op_types, _, _, _ = self._readAndCheckExecutionFile()
      self.assertTrue(executed_op_types)

    (op_names, _,
     tensor_values) = self._readAndCheckGraphExecutionTracesFile(context_ids)
    executed_op_types = [op_name_to_op_type[op_name] for op_name in op_names]
    # These are the ops that we can safely assume to have been executed during
    # the model prediction.
    self.assertIn("MatMul", executed_op_types)
    self.assertIn("BiasAdd", executed_op_types)
    # On the GPU, CudnnRNN is used in lieu of the default op-by-op
    # implementation.
    self.assertTrue(
        ("Sigmoid" in executed_op_types and "Tanh" in executed_op_types or
         "CudnnRNN" in executed_op_types))
    # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought to
    # be an empty float32 tensor.
    for tensor_value in tensor_values:
      self.assertEqual(tensor_value.dtype, np.float32)
      self.assertEqual(tensor_value.shape, (0,))

  @test_util.run_in_graph_and_eager_modes
  def testSimpleKerasRecurrentModelFit(self):
    writer = dumping_callback.enable_dumping(self.dump_root)
    model = _create_simple_recurrent_keras_model()
    xs = np.ones([5, 8, 4])
    ys = np.ones([5, 1])

    history = model.fit(xs, ys, epochs=3, verbose=0)
    self.assertAllClose(
        history.history["loss"], [1.0, 0.9603999853134155, 0.9223681688308716])

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    stack_frame_by_id = self._readAndCheckSourceFilesAndStackFrames()
    (context_ids, op_types,
     op_name_to_op_type) = self._readAndCheckGraphsFile(stack_frame_by_id)
    # Simply assert that graph are recorded and refrain from asserting on the
    # internal details of the Keras model.
    self.assertTrue(context_ids)
    self.assertTrue(op_types)
    self.assertTrue(op_name_to_op_type)

    if context.executing_eagerly():
      # NOTE(b/142486213): Execution of the TF function happens with
      # Session.run() in v1 graph mode, hence it doesn't get logged to the
      # .execution file.
      executed_op_types, _, _, _ = self._readAndCheckExecutionFile()
      self.assertTrue(executed_op_types)

    (op_names, _,
     tensor_values) = self._readAndCheckGraphExecutionTracesFile(context_ids)
    executed_op_types = [op_name_to_op_type[op_name] for op_name in op_names]
    # These are the ops that we can safely assume to have been executed during
    # the recurrent model's fit() call.
    self.assertIn("MatMul", executed_op_types)
    self.assertIn("BiasAdd", executed_op_types)
    # On the GPU, CudnnRNN is used in lieu of the default op-by-op
    # implementation.
    self.assertTrue(
        ("Sigmoid" in executed_op_types and "Tanh" in executed_op_types or
         "CudnnRNN" in executed_op_types))
    self.assertTrue(
        ("SigmoidGrad" in executed_op_types and
         "TanhGrad" in executed_op_types or
         "CudnnRNNBackprop" in executed_op_types))
    # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought to
    # be an empty float32 tensor.
    for tensor_value in tensor_values:
      self.assertEqual(tensor_value.dtype, np.float32)
      self.assertEqual(tensor_value.shape, (0,))

  @test_util.run_in_graph_and_eager_modes
  @test_util.disable_xla("TODO(cais): Investigate timeout.")
  def testMobileNetV2Fit(self):
    """Test training Keras MobileNetV2 application works w/ check numerics."""
    # Use a large circular-buffer to make sure we capture all the executed ops.
    writer = dumping_callback.enable_dumping(self.dump_root,
                                             circular_buffer_size=100000)
    model = mobilenet_v2.MobileNetV2(alpha=0.1, weights=None)

    xs = np.zeros([2] + list(model.input_shape[1:]))
    ys = np.zeros([2] + list(model.output_shape[1:]))
    model.compile(optimizer="sgd", loss="categorical_crossentropy")
    epochs = 1
    history = model.fit(xs, ys, epochs=epochs, verbose=0)
    self.assertEqual(len(history.history["loss"]), epochs)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    stack_frame_by_id = self._readAndCheckSourceFilesAndStackFrames()
    (context_ids, op_types,
     op_name_to_op_type) = self._readAndCheckGraphsFile(stack_frame_by_id)
    # Simply assert that graph are recorded and refrain from asserting on the
    # internal details of the Keras model.
    self.assertTrue(context_ids)
    self.assertTrue(op_types)
    self.assertTrue(op_name_to_op_type)

    if context.executing_eagerly():
      # NOTE(b/142486213): Execution of the TF function happens with
      # Session.run() in v1 graph mode, hence it doesn't get logged to the
      # .execution file.
      executed_op_types, _, _, _ = self._readAndCheckExecutionFile()
      self.assertTrue(executed_op_types)

    (op_names, _,
     tensor_values) = self._readAndCheckGraphExecutionTracesFile(context_ids)
    executed_op_types = [op_name_to_op_type[op_name] for op_name in op_names]
    # These are the ops that we can safely assume to have been executed during
    # the model's fit() call.
    self.assertIn("Conv2D", executed_op_types)
    self.assertIn("Relu6", executed_op_types)
    self.assertIn("Conv2DBackpropFilter", executed_op_types)
    self.assertIn("Relu6Grad", executed_op_types)
    # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought to
    # be an empty float32 tensor.
    for tensor_value in tensor_values:
      self.assertEqual(tensor_value.dtype, np.float32)
      self.assertEqual(tensor_value.shape, (0,))


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
