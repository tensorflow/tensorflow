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

import collections
import os
import shutil
import socket
import tempfile
import threading

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.debug.lib import dumping_callback_test_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.keras import models
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.layers import recurrent_v2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


def _create_simple_recurrent_keras_model(input_shape):
  """Create a simple tf.keras model containing a recurrent layer for testing."""
  model = models.Sequential()
  model.add(recurrent_v2.LSTM(
      10,
      input_shape=input_shape,
      kernel_initializer="zeros",
      recurrent_initializer="zeros"))
  model.add(core.Dense(1, kernel_initializer="zeros"))
  model.compile(loss="mse", optimizer="sgd")
  return model


_host_name = socket.gethostname()
_current_file_full_path = os.path.abspath(__file__)


class TracingCallbackTest(
    dumping_callback_test_lib.DumpingCallbackTestBase, parameterized.TestCase):

  def setUp(self):
    super(TracingCallbackTest, self).setUp()
    self.dump_root = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self.dump_root):
      shutil.rmtree(self.dump_root, ignore_errors=True)
    dumping_callback.disable_dump_debug_info()
    super(TracingCallbackTest, self).tearDown()

  def _verifyStackFrames(self, stack_frames):
    """Verify the correctness of the stack frames.

    Currently, it simply asserts that the current file is found in the stack
    frames.
    TODO(cais): Perhaps implement a stricter check later.

    Args:
      stack_frames: The stack frames to verify.
    """
    self.assertTrue([
        frame for frame in stack_frames if frame[0] == _current_file_full_path])

  def _expectedDefaultDeviceName(self):
    gpu_name = test_util.gpu_device_name()
    if gpu_name:
      return "/job:localhost/replica:0/task:0" + gpu_name
    else:
      return "/job:localhost/replica:0/task:0/device:CPU:0"

  def testInvalidTensorDebugModeCausesError(self):
    with self.assertRaisesRegexp(
        ValueError,
        r"Invalid value in tensor_debug_mode \(\'NONSENSICAL\'\).*"
        r"Valid options.*NO_TENSOR.*"):
      dumping_callback.enable_dump_debug_info(
          self.dump_root, tensor_debug_mode="NONSENSICAL")

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("CurtHealth", "CURT_HEALTH"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("Shape", "SHAPE"),
      ("FulHealth", "FULL_HEALTH"),
      ("FullTensor", "FULL_TENSOR"),
  )
  def testEnableDumpDebugInfoLogsTensorDebugModeAsStringName(self,
                                                             tensor_debug_mode):
    log_messages = []
    def fake_logging_info(*args):
      log_messages.append(args)
    with test.mock.patch.object(
        tf_logging, "info", side_effect=fake_logging_info):
      dumping_callback.enable_dump_debug_info(
          self.dump_root, tensor_debug_mode=tensor_debug_mode)
      self.assertLen(log_messages, 1)
      self.assertIn(self.dump_root, log_messages[0])
      self.assertIn(tensor_debug_mode, log_messages[0])

  def testDisablingTracingCallbackWithoutEnablingFirstIsTolerated(self):
    dumping_callback.disable_dump_debug_info()

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("CurtHealth", "CURT_HEALTH"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("Shape", "SHAPE"),
      ("FullHealth", "FULL_HEALTH"),
      ("FullTensor", "FULL_TENSOR"),
  )
  def testPureEagerOpExecution(self, tensor_debug_mode):
    """Test dumping data from eager op execution: float32."""
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

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

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      # Before FlushExecutionFiles() is called, the .execution file should be
      # empty.
      self.assertFalse(reader.executions())

      # After the flushing, the .execution file should hold the appropriate
      # contents.
      writer.FlushExecutionFiles()
      reader.update()
      executions = reader.executions()
      prev_wall_time = 1
      executed_op_types = []
      tensor_values = collections.defaultdict(lambda: [])
      for execution in executions:
        self.assertGreaterEqual(execution.wall_time, prev_wall_time)
        prev_wall_time = execution.wall_time
        executed_op_types.append(execution.op_type)
        # Check the device name.
        if execution.op_type in ("AddV2", "Mul", "RealDiv"):
          self.assertLen(execution.output_tensor_device_ids, 1)
          self.assertEqual(
              reader.device_name_by_id(execution.output_tensor_device_ids[0]),
              self._expectedDefaultDeviceName(),
              "Unexpected device name from eager op %s" % execution.op_type)

        # No graph IDs should have been logged for eager op executions.
        self.assertFalse(execution.graph_id)
        self.assertTrue(execution.input_tensor_ids)
        self.assertTrue(execution.output_tensor_ids)
        self.assertEqual(
            debug_event_pb2.TensorDebugMode.keys()[execution.tensor_debug_mode],
            tensor_debug_mode)
        if tensor_debug_mode == "NO_TENSOR":
          # Due to the NO_TENSOR tensor debug mode, tensor_protos ought to
          # be empty.
          self.assertFalse(execution.debug_tensor_values)
        elif tensor_debug_mode == "CURT_HEALTH":
          self.assertLen(execution.debug_tensor_values, 1)
          if execution.op_type in ("AddV2", "Mul", "RealDiv"):
            # 1st element: -1 is the unset tensor_id for eager op execution.
            # 2nd element: 0 means there is no inf or nan.
            self.assertAllClose(execution.debug_tensor_values, [[-1.0, 0.0]])
        elif tensor_debug_mode == "CONCISE_HEALTH":
          if execution.op_type in ("AddV2", "Mul", "RealDiv"):
            # 1st element: -1 is the unset tensor_id for eager op execution.
            # 2nd element: each scalar tensor has 1 element.
            # Remaining elements: no -inf, inf or nan in these
            self.assertAllClose(
                execution.debug_tensor_values, [[-1, 1, 0, 0, 0]])
        elif tensor_debug_mode == "FULL_HEALTH":
          if execution.op_type in ("AddV2", "Mul", "RealDiv"):
            # Elements: [
            #   -1 is the unset tensor_id for eager op execution,
            #   device ID (set to -1 for now),
            #   dtype, rank, element_count,
            #   neg_inf_count, pos_inf_count, nan_count
            #   neg_finite_count, zero_count, pos_finite_count]
            self.assertAllClose(
                execution.debug_tensor_values,
                [[-1, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1]])
        elif tensor_debug_mode == "SHAPE":
          if execution.op_type in ("AddV2", "Mul", "RealDiv"):
            # 1st element: -1 is the unset tensor_id for eager op execution.
            # 2nd element: dtype enum value (float32).
            # 3rd element: rank (scalar).
            # 4th element: element count (4).
            # Remaining elements: shape at fixed length (6).
            self.assertAllClose(execution.debug_tensor_values,
                                [[-1, 1, 0, 1, 0, 0, 0, 0, 0, 0]])
        elif tensor_debug_mode == "FULL_TENSOR":
          tensor_values[execution.op_type].append(
              reader.execution_to_tensor_values(execution)[0])

        host_name, stack_frames = reader.read_execution_stack_trace(execution)
        self.assertEqual(host_name, _host_name)
        self._verifyStackFrames(stack_frames)

      if tensor_debug_mode == "FULL_TENSOR":
        self.assertAllClose(tensor_values["Greater"], [1, 1, 1, 1, 1, 1, 0])
        self.assertAllClose(tensor_values["RealDiv"], [5, 8, 4, 2, 1])
        self.assertAllClose(tensor_values["Mul"], [15])
        self.assertAllClose(tensor_values["AddV2"], [16])

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
      self.assertFalse(reader.outermost_graphs())
      self.assertEqual(reader.num_graph_execution_traces(), 0)

  @parameterized.named_parameters(
      ("CurtHealth", "CURT_HEALTH"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("FullHealth", "FULL_HEALTH"),
      ("Shape", "SHAPE"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testModesSummarizingBadNumericalValue(self, tensor_debug_mode):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

    @def_function.function
    def func(x, y):
      return (x + y) / (x - y)

    x = np.array([-3, -1, 0, 0, 1, 1, 1, 2], dtype=np.float16)
    y = np.array([2, -1, 0, 0, 1, 1, 1, 3], dtype=np.float16)
    # x - y = [-5, 0, 0, 0, 0, 0, 0, -1]
    # (x + y) / (x - y) = [0.2, -inf, nan, nan, inf, inf, inf, -5].
    self.evaluate(func(x, y))
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_exec_traces = reader.graph_execution_traces()
      executed_op_types = [trace.op_type for trace in graph_exec_traces]
      self.assertCountEqual(
          executed_op_types,
          ["Placeholder", "Placeholder", "AddV2", "Sub", "RealDiv"])
      if tensor_debug_mode == "CURT_HEALTH":
        for trace in graph_exec_traces:
          # 1st element: tensor_id, should be >= 0.
          # 2nd element: indicates if there is any inf or nan.
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          self.assertGreaterEqual(tensor_id, 0)
          if trace.op_type == "RealDiv":
            self.assertAllClose(trace.debug_tensor_value, [tensor_id, 1])
          else:
            self.assertAllClose(trace.debug_tensor_value, [tensor_id, 0])
      elif tensor_debug_mode == "CONCISE_HEALTH":
        for trace in graph_exec_traces:
          # 1st element: tensor_id, should be >= 0.
          # 2nd element: element count (8).
          # Remaining 3 elements: The counts of -inf, inf and nan.
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          self.assertGreaterEqual(tensor_id, 0)
          if trace.op_type == "RealDiv":
            self.assertAllClose(trace.debug_tensor_value,
                                [tensor_id, 8, 1, 3, 2])
          else:
            self.assertAllClose(trace.debug_tensor_value,
                                [tensor_id, 8, 0, 0, 0])
      elif tensor_debug_mode == "FULL_HEALTH":
        for trace in graph_exec_traces:
          # Elements: [
          #   -1 is the unset tensor_id for eager op execution,
          #   device ID (set to -1 for now),
          #   dtype, rank, element_count,
          #   neg_inf_count, pos_inf_count, nan_count
          #   neg_finite_count, zero_count, pos_finite_count]
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          self.assertGreaterEqual(tensor_id, 0)
          if trace.op_type == "RealDiv":
            self.assertAllClose(trace.debug_tensor_value,
                                [tensor_id, -1, 19, 1, 8, 1, 3, 2, 1, 0, 1])
          elif trace.op_type == "Sub":
            self.assertAllClose(trace.debug_tensor_value,
                                [tensor_id, -1, 19, 1, 8, 0, 0, 0, 2, 6, 0])
      else:  # SHAPE.
        for trace in graph_exec_traces:
          # 1st element: tensor_id, should be >= 0.
          # 2nd element: dtype enum value (float16 = 19).
          # 3rd element: rank (1)
          # 4th element: element count (8).
          # Remaining elements: shape at fixed length (6).
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          self.assertGreaterEqual(tensor_id, 0)
          self.assertAllClose(trace.debug_tensor_value,
                              [tensor_id, 19, 1, 8, 8, 0, 0, 0, 0, 0])

  @parameterized.named_parameters(
      ("Shape", "SHAPE"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testBooleanTensors(self, tensor_debug_mode):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

    @def_function.function
    def func(x, y):
      return math_ops.logical_not(math_ops.logical_and(x, y))

    x = np.array([[False, False], [True, True]], dtype=np.bool)
    y = np.array([[False, True], [False, True]], dtype=np.bool)
    self.assertAllEqual(
        self.evaluate(func(x, y)), [[True, True], [True, False]])

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_exec_traces = reader.graph_execution_traces()
      executed_op_types = [trace.op_type for trace in graph_exec_traces]
      self.assertEqual(
          executed_op_types,
          ["Placeholder", "Placeholder", "LogicalAnd", "LogicalNot"])
      for trace in graph_exec_traces:
        tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
        self.assertGreaterEqual(tensor_id, 0)
        # 1st element: tensor_id, should be >= 0.
        # 2nd element: dtype enum value (bool).
        # 3rd element: rank (2).
        # 4th element: element count (4).
        # Remaining elements: shape at fixed length.
        self.assertAllClose(
            trace.debug_tensor_value, [tensor_id, 10, 2, 4, 2, 2, 0, 0, 0, 0])

  def testListingSourceFiles(self):
    writer = dumping_callback.enable_dump_debug_info(self.dump_root)
    # Run a simple eager execution event, so that the source files are dumped.
    self.assertAllClose(math_ops.truediv(7.0, 1.0 / 6.0), 42.0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()
    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      source_file_list = reader.source_file_list()
      self.assertIsInstance(source_file_list, tuple)
      for item in source_file_list:
        self.assertIsInstance(item, tuple)
        self.assertLen(item, 2)
      self.assertIn((_host_name, _current_file_full_path), source_file_list)

  def testReadingSourceLines(self):
    writer = dumping_callback.enable_dump_debug_info(self.dump_root)
    # Run a simple eager execution event, so that the source-file contents are
    # dumped.
    self.assertAllClose(math_ops.truediv(7.0, 1.0 / 6.0), 42.0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()
    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      with open(_current_file_full_path, "rt") as f:
        file_lines = f.read().split("\n")
      self.assertEqual(
          reader.source_lines(_host_name, _current_file_full_path), file_lines)

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("CurtHealth", "CURT_HEALTH"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("FullHealth", "FULL_HEALTH"),
      ("Shape", "SHAPE"),
      ("FullTensor", "FULL_TENSOR"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testNestedFunctionExecutionWithoutControlFlow(self, tensor_debug_mode):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

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

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      outermost_graphs = reader.outermost_graphs()
      self.assertLen(outermost_graphs, 1)

      if context.executing_eagerly():
        # NOTE(b/142486213): Execution of the TF function happens with
        # Session.run() in v1 graph mode, so doesn't get logged to the
        # .execution file.
        executions = reader.executions()
        self.assertLen(executions, 1)
        self.assertIn("sin1p_log_sum", executions[0].op_type)
        # Get the executed graph and verify its identity and inner graph.
        graph = reader.graph_by_id(executions[0].graph_id)
        self.assertEqual(graph.name, "sin1p_log_sum")
        self.assertLen(graph.inner_graph_ids, 1)
        inner_graph = reader.graph_by_id(graph.inner_graph_ids[0])
        self.assertEqual(inner_graph.name, "log_sum")
        # Check device names.
        self.assertLen(executions[0].output_tensor_device_ids, 1)
        self.assertEqual(
            reader.device_name_by_id(executions[0].output_tensor_device_ids[0]),
            self._expectedDefaultDeviceName())
        self.assertIn(self._expectedDefaultDeviceName(),
                      set(reader.device_name_map().values()))

      # Verify the recorded graph-building history.
      placeholder_op_digests = reader.graph_op_digests(op_type="Placeholder")
      add_op_digests = reader.graph_op_digests(op_type="AddV2")
      self.assertLen(add_op_digests, 2)
      self.assertEqual(
          reader.graph_by_id(add_op_digests[0].graph_id).name, "log_sum")
      self.assertEqual(
          reader.graph_by_id(add_op_digests[1].graph_id).name, "sin1p_log_sum")
      log_op_digests = reader.graph_op_digests(op_type="Log")
      self.assertLen(log_op_digests, 1)
      self.assertEqual(
          reader.graph_by_id(log_op_digests[0].graph_id).name, "log_sum")
      sin_op_digests = reader.graph_op_digests(op_type="Sin")
      self.assertLen(sin_op_digests, 1)
      self.assertEqual(
          reader.graph_by_id(sin_op_digests[0].graph_id).name, "sin1p_log_sum")

      # Verify the output tensor IDs and the stack traces.
      for op_digest in add_op_digests + log_op_digests + sin_op_digests:
        # These are all single-output ops.
        self.assertLen(op_digest.output_tensor_ids, 1)
        self.assertGreaterEqual(op_digest.output_tensor_ids[0], 0)
        _, stack_frames = reader.read_graph_op_creation_stack_trace(op_digest)
        self._verifyStackFrames(stack_frames)

      graph_exec_traces = reader.graph_execution_traces()
      executed_op_types = [digest.op_type for digest in graph_exec_traces]
      self.assertEqual(
          executed_op_types,
          ["Placeholder", "Placeholder", "Placeholder", "Placeholder",
           "AddV2", "Log", "AddV2", "Sin"])
      placeholder_traces = graph_exec_traces[:4]
      non_placeholder_traces = graph_exec_traces[4:]

      # Verify the graph ID stack of each op.
      # The outer function's 1st Placeholder.
      self.assertEqual(
          reader.graph_by_id(placeholder_traces[0].graph_ids[-1]).name,
          "sin1p_log_sum")
      # The outer function's 2nd Placeholder.
      self.assertEqual(
          reader.graph_by_id(placeholder_traces[1].graph_ids[-1]).name,
          "sin1p_log_sum")
      # The inner function's 1st Placeholder.
      self.assertEqual(
          reader.graph_by_id(placeholder_traces[2].graph_ids[-1]).name,
          "log_sum")
      self.assertEqual(
          reader.graph_by_id(placeholder_traces[2].graph_ids[-2]).name,
          "sin1p_log_sum")
      # The inner function's 2nd Placeholder.
      self.assertEqual(
          reader.graph_by_id(placeholder_traces[3].graph_ids[-1]).name,
          "log_sum")
      self.assertEqual(
          reader.graph_by_id(placeholder_traces[3].graph_ids[-2]).name,
          "sin1p_log_sum")
      # 1st AddV2 op.
      self.assertEqual(
          reader.graph_by_id(non_placeholder_traces[0].graph_ids[-1]).name,
          "log_sum")
      self.assertEqual(
          reader.graph_by_id(non_placeholder_traces[0].graph_ids[-2]).name,
          "sin1p_log_sum")
      # Log op.
      self.assertEqual(
          reader.graph_by_id(non_placeholder_traces[1].graph_ids[-1]).name,
          "log_sum")
      self.assertEqual(
          reader.graph_by_id(non_placeholder_traces[1].graph_ids[-2]).name,
          "sin1p_log_sum")
      # 2nd AddV2 op.
      self.assertEqual(
          reader.graph_by_id(non_placeholder_traces[2].graph_ids[-1]).name,
          "sin1p_log_sum")
      # Sin op.
      self.assertEqual(
          reader.graph_by_id(non_placeholder_traces[3].graph_ids[-1]).name,
          "sin1p_log_sum")

      if tensor_debug_mode == "NO_TENSOR":
        # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought
        # to be an empty float32 tensor.
        for trace in graph_exec_traces:
          self.assertIsNone(trace.debug_tensor_value)
      elif tensor_debug_mode == "CURT_HEALTH":
        # Test the association between graph exec and prior graph building.
        # In each case, the 1st element of debug_tensor_value is the ID of the
        # symbolic tenosr and the 2nd element is a zero indicating there is no
        # inf or nan.
        self.assertAllClose(  # 1st outer placeholder.
            placeholder_traces[0].debug_tensor_value,
            [placeholder_op_digests[0].output_tensor_ids[0], 0.0])
        self.assertAllClose(  # 2nd outer placeholder.
            placeholder_traces[1].debug_tensor_value,
            [placeholder_op_digests[1].output_tensor_ids[0], 0.0])
        self.assertAllClose(  # 1st inner placeholder.
            placeholder_traces[2].debug_tensor_value,
            [placeholder_op_digests[2].output_tensor_ids[0], 0.0])
        self.assertAllClose(  # 2nd outer placeholder.
            placeholder_traces[3].debug_tensor_value,
            [placeholder_op_digests[3].output_tensor_ids[0], 0.0])
        self.assertAllClose(  # 1st AddV2 op.
            non_placeholder_traces[0].debug_tensor_value,
            [add_op_digests[0].output_tensor_ids[0], 0.0])
        self.assertAllClose(  # Log op.
            non_placeholder_traces[1].debug_tensor_value,
            [log_op_digests[0].output_tensor_ids[0], 0.0])
        self.assertAllClose(  # 2nd AddV2 op.
            non_placeholder_traces[2].debug_tensor_value,
            [add_op_digests[1].output_tensor_ids[0], 0.0])
        self.assertAllClose(  # Sin op.
            non_placeholder_traces[3].debug_tensor_value,
            [sin_op_digests[0].output_tensor_ids[0], 0.0])
      elif tensor_debug_mode == "CONCISE_HEALTH":
        # 1st element: tensor_id.
        # 2nd element: element count. Remaining elements: all zero because there
        # is no -inf, inf or nan.
        self.assertAllClose(  # 1st outer placeholder.
            placeholder_traces[0].debug_tensor_value,
            [placeholder_op_digests[0].output_tensor_ids[0], 1., 0., 0., 0.])
        self.assertAllClose(  # 2nd outer placeholder.
            placeholder_traces[1].debug_tensor_value,
            [placeholder_op_digests[1].output_tensor_ids[0], 1., 0., 0., 0.])
        self.assertAllClose(  # 1st inner placeholder.
            placeholder_traces[2].debug_tensor_value,
            [placeholder_op_digests[2].output_tensor_ids[0], 1., 0., 0., 0.])
        self.assertAllClose(  # 2nd outer placeholder.
            placeholder_traces[3].debug_tensor_value,
            [placeholder_op_digests[3].output_tensor_ids[0], 1., 0., 0., 0.])
        # 1st AddV2 op.
        self.assertAllClose(
            non_placeholder_traces[0].debug_tensor_value,
            [add_op_digests[0].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
        # Log op.
        self.assertAllClose(
            non_placeholder_traces[1].debug_tensor_value,
            [log_op_digests[0].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
        # 2nd AddV2 op.
        self.assertAllClose(
            non_placeholder_traces[2].debug_tensor_value,
            [add_op_digests[1].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
        # Sin op.
        self.assertAllClose(
            non_placeholder_traces[3].debug_tensor_value,
            [sin_op_digests[0].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
      elif tensor_debug_mode == "FULL_HEALTH":
        # Elements: [
        #   -1 is the unset tensor_id for eager op execution,
        #   device ID (set to -1 for now),
        #   dtype, rank, element_count,
        #   neg_inf_count, pos_inf_count, nan_count
        #   neg_finite_count, zero_count, pos_finite_count]
        self.assertAllClose(  # 1st outer placeholder.
            placeholder_traces[0].debug_tensor_value,
            [placeholder_op_digests[0].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        self.assertAllClose(  # 2nd outer placeholder.
            placeholder_traces[1].debug_tensor_value,
            [placeholder_op_digests[1].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        self.assertAllClose(  # 1st inner placeholder.
            placeholder_traces[2].debug_tensor_value,
            [placeholder_op_digests[2].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        self.assertAllClose(  # 2nd outer placeholder.
            placeholder_traces[3].debug_tensor_value,
            [placeholder_op_digests[3].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        # 1st AddV2 op.
        self.assertAllClose(
            non_placeholder_traces[0].debug_tensor_value,
            [add_op_digests[0].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        # Log op.
        self.assertAllClose(
            non_placeholder_traces[1].debug_tensor_value,
            [log_op_digests[0].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        # 2nd AddV2 op.
        self.assertAllClose(
            non_placeholder_traces[2].debug_tensor_value,
            [add_op_digests[1].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        # Sin op.
        self.assertAllClose(
            non_placeholder_traces[3].debug_tensor_value,
            [sin_op_digests[0].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
      elif tensor_debug_mode == "SHAPE":
        # 1st element: tensor_id.
        # 2nd element: dtype (float32).
        # 3rd element: rank (scalar).
        # 4th element: element count (1).
        # Remaining elements: shape padded to fixed length (6).
        self.assertAllClose(  # 1st outer placeholder.
            placeholder_traces[0].debug_tensor_value,
            [placeholder_op_digests[0].output_tensor_ids[0],
             1, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertAllClose(  # 2nd outer placeholder.
            placeholder_traces[1].debug_tensor_value,
            [placeholder_op_digests[1].output_tensor_ids[0],
             1, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertAllClose(  # 1st inner placeholder.
            placeholder_traces[2].debug_tensor_value,
            [placeholder_op_digests[2].output_tensor_ids[0],
             1, 0, 1, 0, 0, 0, 0, 0, 0])
        self.assertAllClose(  # 2nd outer placeholder.
            placeholder_traces[3].debug_tensor_value,
            [placeholder_op_digests[3].output_tensor_ids[0],
             1, 0, 1, 0, 0, 0, 0, 0, 0])
        # 1st AddV2 op.
        self.assertAllClose(
            non_placeholder_traces[0].debug_tensor_value,
            [add_op_digests[0].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
        # Log op.
        self.assertAllClose(
            non_placeholder_traces[1].debug_tensor_value,
            [log_op_digests[0].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
        # 2nd AddV2 op.
        self.assertAllClose(
            non_placeholder_traces[2].debug_tensor_value,
            [add_op_digests[1].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
        # Sin op.
        self.assertAllClose(
            non_placeholder_traces[3].debug_tensor_value,
            [sin_op_digests[0].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
      else:  # FULL_TENSOR.
        placeholder_full_tensor_values = [
            reader.graph_execution_trace_to_tensor_value(trace)
            for trace in placeholder_traces]
        self.assertAllClose(placeholder_full_tensor_values[0], x)  # Input x.
        self.assertAllClose(placeholder_full_tensor_values[1], y)  # Input y.
        self.assertAllClose(placeholder_full_tensor_values[2], x)  # Input x.
        self.assertAllClose(placeholder_full_tensor_values[3], y)  # Input y.
        non_placeholder_full_tensor_values = [
            reader.graph_execution_trace_to_tensor_value(trace)
            for trace in non_placeholder_traces]
        self.assertAllClose(
            non_placeholder_full_tensor_values[0], 5.0)  # 1st AddV2 op.
        self.assertAllClose(
            non_placeholder_full_tensor_values[1], np.log(5.0))  # Log op.
        self.assertAllClose(
            non_placeholder_full_tensor_values[2],
            np.log(5.0) + 1.0)  # 2nd AddV2 op.
        self.assertAllClose(
            non_placeholder_full_tensor_values[3],
            np.sin(np.log(5.0) + 1.0))  # Sin op.

  def testCapturingExecutedGraphIdsOfTwoCompilationsOfSameFunction(self):
    """Test correct executed IDs of two FuncGraphs from the same Py function."""
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="NO_TENSOR")

    @def_function.function
    def ceil_times_two(x):
      return math_ops.ceil(x) * 2.0

    x_float32 = np.array(3.5, dtype=np.float32)
    x_float64 = np.array(4.5, dtype=np.float64)
    # Four executions, with two different FuncGraphs, which should lead
    # to two unique executed graph IDs (see assertion below).
    self.assertAllClose(ceil_times_two(x_float32), 8.0)
    self.assertAllClose(ceil_times_two(x_float64), 10.0)
    self.assertAllClose(ceil_times_two(x_float32), 8.0)
    self.assertAllClose(ceil_times_two(x_float64), 10.0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()

      executions = reader.executions()
      self.assertLen(executions, 4)
      for execution in executions:
        self.assertStartsWith(execution.op_type, "__inference_ceil_times_two_")
      executed_graph_ids = [execution.graph_id for execution in executions]
      self.assertEqual(executed_graph_ids[0], executed_graph_ids[2])
      self.assertEqual(executed_graph_ids[1], executed_graph_ids[3])
      self.assertNotEqual(executed_graph_ids[0], executed_graph_ids[1])
      self.assertNotEqual(executed_graph_ids[2], executed_graph_ids[3])
      for executed_graph_id in executed_graph_ids:
        self.assertEqual(
            reader.graph_by_id(executed_graph_id).name, "ceil_times_two")

  def testCapturingExecutedGraphIdsOfDuplicateFunctionNames(self):
    """Two FuncGraphs compiled from Python functions with identical names."""
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="NO_TENSOR")

    class TestClass(object):

      @def_function.function
      def ceil_times_two(self, x):
        return math_ops.ceil(x) * 2.0

    # The `ceil_times_two` method of the two objects will be compiled
    # into separate FuncGraphs.
    test_object_1 = TestClass()
    test_object_2 = TestClass()

    x = np.array(3.5, dtype=np.float32)
    # Four executions, with two different FuncGraphs, which should lead
    # to two unique executed graph IDs (see assertion below).
    self.assertAllClose(test_object_1.ceil_times_two(x), 8.0)
    self.assertAllClose(test_object_2.ceil_times_two(x), 8.0)
    self.assertAllClose(test_object_1.ceil_times_two(x), 8.0)
    self.assertAllClose(test_object_2.ceil_times_two(x), 8.0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      executions = reader.executions()
      self.assertLen(executions, 4)
      for execution in executions:
        self.assertStartsWith(execution.op_type, "__inference_ceil_times_two_")
      executed_graph_ids = [execution.graph_id for execution in executions]
      self.assertEqual(executed_graph_ids[0], executed_graph_ids[2])
      self.assertEqual(executed_graph_ids[1], executed_graph_ids[3])
      self.assertNotEqual(executed_graph_ids[0], executed_graph_ids[1])
      self.assertNotEqual(executed_graph_ids[2], executed_graph_ids[3])
      for executed_graph_id in executed_graph_ids:
        self.assertEqual(
            reader.graph_by_id(executed_graph_id).name, "ceil_times_two")

  @parameterized.named_parameters(
      ("AddV2", "AddV2"),
      ("Log", "Log"),
      ("AddV2AndLog", "(AddV2|Log)"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testOpRegex(self, op_regex):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="FULL_TENSOR",
        op_regex=op_regex)

    @def_function.function
    def log_sum(x, y):
      return math_ops.log(x + y)

    @def_function.function
    def sin1p_log_sum(x, y):
      return math_ops.sin(1.0 + log_sum(x, y))

    x = constant_op.constant(2.0)
    y = constant_op.constant(3.0)
    self.assertAllClose(
        self.evaluate(sin1p_log_sum(x, y)), np.sin(1.0 + np.log(5.0)))
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_op_digests = reader.graph_op_digests()
      op_types = [digest.op_type for digest in graph_op_digests]
      self.assertIn("AddV2", op_types)
      self.assertIn("Log", op_types)
      self.assertIn("Sin", op_types)

      graph_exec_digests = reader.graph_execution_traces(digest=True)
      executed_op_types = [digest.op_type for digest in graph_exec_digests]
      tensor_values = [reader.graph_execution_trace_to_tensor_value(digest)
                       for digest in graph_exec_digests]
      if op_regex == "AddV2":
        self.assertEqual(executed_op_types, ["AddV2", "AddV2"])
        self.assertLen(tensor_values, 2)
        self.assertAllClose(tensor_values[0], 5.0)  # 1st AddV2 op.
        self.assertAllClose(
            tensor_values[1], np.log(5.0) + 1.0)  # 2nd AddV2 op.
      elif op_regex == "Log":
        self.assertEqual(executed_op_types, ["Log"])
        self.assertLen(tensor_values, 1)
        self.assertAllClose(tensor_values[0], np.log(5.0))  # Log op.
      else:  # "(AddV2|Log)"
        self.assertEqual(executed_op_types, ["AddV2", "Log", "AddV2"])
        self.assertLen(tensor_values, 3)
        self.assertAllClose(tensor_values[0], 5.0)  # 1st AddV2 op.
        self.assertAllClose(tensor_values[1], np.log(5.0))  # Log op.
        self.assertAllClose(
            tensor_values[2], np.log(5.0) + 1.0)  # 2nd AddV2 op.

  def testIncorrectTensorDTypeArgFormatLeadsToError(self):
    with self.assertRaisesRegexp(
        ValueError,
        r".*expected.*list.*tuple.*callable.*but received.*\{\}"):
      dumping_callback.enable_dump_debug_info(self.dump_root,
                                              tensor_dtypes=dict())
    with self.assertRaisesRegexp(
        ValueError,
        r".*expected.*list.*tuple.*callable.*but received.*"):
      dumping_callback.enable_dump_debug_info(self.dump_root,
                                              tensor_dtypes="float32")
    with self.assertRaisesRegexp(
        ValueError,
        r".*expected.*list.*tuple.*callable.*but received.*"):
      dumping_callback.enable_dump_debug_info(
          self.dump_root, tensor_dtypes=dtypes.float32)
    with self.assertRaises(TypeError):
      dumping_callback.enable_dump_debug_info(self.dump_root, tensor_dtypes=[
          lambda dtype: dtype.is_floating, lambda dtype: dtype.is_integer])

  @parameterized.named_parameters(
      ("float", [dtypes.float32], None),
      ("float_only_sum", ["float32"], "Sum"),
      ("float_no_sum", (dtypes.float32,), "(?!Sum)"),
      ("int", [dtypes.int32], None),
      ("int_via_lambda", lambda dtype: dtype.is_integer, None),
      ("exclude_Sum", None, "(?!Sum)"),
      ("All", None, None),
  )
  @test_util.run_in_graph_and_eager_modes
  def testTensorDTypesAndOpRegexFilters(self,
                                        tensor_dtypes,
                                        op_regex):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="FULL_TENSOR",
        tensor_dtypes=tensor_dtypes,
        op_regex=op_regex)

    @def_function.function
    def unique_sum(xs):
      """Sum over the unique values, for testing."""
      unique_xs, indices = array_ops.unique(xs)
      return math_ops.reduce_sum(unique_xs), indices

    xs = constant_op.constant([2., 6., 8., 1., 2.], dtype=dtypes.float32)
    y, indices = self.evaluate(unique_sum(xs))
    self.assertAllClose(y, 17.)
    self.assertAllEqual(indices, [0, 1, 2, 3, 0])

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_exec_digests = reader.graph_execution_traces(digest=True)
      executed_op_types = [digest.op_type for digest in graph_exec_digests
                           if digest.op_type != "Placeholder"]
      tensor_values = [reader.graph_execution_trace_to_tensor_value(digest)
                       for digest in graph_exec_digests
                       if digest.op_type != "Placeholder"]

      if tensor_dtypes == [dtypes.float32] and not op_regex:
        self.assertEqual(executed_op_types, ["Unique", "Sum"])
        self.assertLen(tensor_values, 2)
        self.assertAllClose(tensor_values[0], [2, 6, 8, 1])  # Unique values.
        self.assertAllClose(tensor_values[1], 17.)  # Sum.
      elif tensor_dtypes == ["float32"] and op_regex == "Sum":
        self.assertEqual(executed_op_types, ["Sum"])
        self.assertLen(tensor_values, 1)
        self.assertAllClose(tensor_values[0], 17.)  # Sum.
      elif tensor_dtypes == (dtypes.float32,) and op_regex == "(?!Sum)":
        self.assertEqual(executed_op_types, ["Unique"])
        self.assertLen(tensor_values, 1)
        self.assertAllClose(tensor_values[0], [2, 6, 8, 1])  # Unique values.
      elif tensor_dtypes == [dtypes.int32] and not op_regex:
        self.assertEqual(executed_op_types, ["Unique"])
        self.assertLen(tensor_values, 1)
        self.assertAllEqual(
            tensor_values[0], [0, 1, 2, 3, 0])  # Unique indices.
      elif callable(tensor_dtypes) and not op_regex:
        self.assertEqual(executed_op_types, ["Unique"])
        self.assertLen(tensor_values, 1)
        self.assertAllEqual(
            tensor_values[0], [0, 1, 2, 3, 0])  # Unique indices.
      elif not tensor_dtypes and op_regex == "(?!Sum)":
        self.assertEqual(executed_op_types, ["Unique", "Unique"])
        self.assertLen(tensor_values, 2)
        self.assertAllClose(tensor_values[0], [2, 6, 8, 1])  # Unique values.
        self.assertAllEqual(
            tensor_values[1], [0, 1, 2, 3, 0])  # Unique indices.
      else:  # "All".
        self.assertEqual(executed_op_types, ["Unique", "Unique", "Sum"])
        self.assertLen(tensor_values, 3)
        self.assertAllClose(tensor_values[0], [2, 6, 8, 1])  # Unique values.
        self.assertAllEqual(
            tensor_values[1], [0, 1, 2, 3, 0])  # Unique indices.
        self.assertAllClose(tensor_values[2], 17)  # Sum.

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("CurtHealth", "CURT_HEALTH"),
      ("FullTensor", "FULL_TENSOR"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testFunctionExecutionWithControlFlow(self, tensor_debug_mode):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

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
    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_op_digests = reader.graph_op_digests()
      op_types = [digest.op_type for digest in graph_op_digests]
      self.assertIn("Less", op_types)
      self.assertIn("Mul", op_types)
      self.assertIn("AddV2", op_types)

      # Before FlushExecutionFiles() is called, the .execution and
      # .graph_execution_traces files should be both empty.
      self.assertEqual(reader.num_executions(), 0)
      self.assertEqual(reader.num_graph_execution_traces(), 0)

      # TODO(cais): Backport execution instrumentation to tf.Session.
      writer.FlushExecutionFiles()
      # After the flushing, the .execution file should hold the appropriate
      # contents.
      reader.update()
      if context.executing_eagerly():
        # NOTE(b/142486213): Execution of the TF function happens with
        # Session.run() in v1 graph mode, hence it doesn't get logged to the
        executions = reader.executions()
        self.assertLen(executions, 1)
        executed_op_types = [execution.op_type for execution in executions]
        self.assertIn("iterative_doubling", executions[0].op_type)
        execution = executions[0]
        self.assertLen(execution.input_tensor_ids, 2)
        self.assertLen(execution.output_tensor_ids, 1)
        self.assertEqual(
            debug_event_pb2.TensorDebugMode.keys()[execution.tensor_debug_mode],
            tensor_debug_mode)
        if tensor_debug_mode == "FULL_TENSOR":
          tensor_values = reader.execution_to_tensor_values(execution)
          self.assertAllClose(tensor_values, [8.0])

      graph_exec_traces = reader.graph_execution_traces()
      executed_op_types = [trace.op_type for trace in graph_exec_traces]
      if tensor_debug_mode != "CURT_HEALTH":
        # Less outputs a boolean tensor, which is not tracked under CURT_HEALTH.
        # The Less op should have been executed 5 times.
        self.assertEqual(executed_op_types.count("Less"), 5)
        # The last executed op should be Less.
        self.assertEqual(executed_op_types[-1], "Less")
        # AddV2 produces an int tensor, which is not tracked under CURT_HEALTH.
        # The AddV2 op should have been run, but we refrain from asserting on
        # how many times it's executed.
        self.assertIn("AddV2", executed_op_types)
        for trace in graph_exec_traces:
          self.assertEqual(trace.output_slot, 0)
      # The Mul op should have been executed 4 times.
      self.assertEqual(executed_op_types.count("Mul"), 4)

      tensor_values = [reader.graph_execution_trace_to_tensor_value(trace)
                       for trace in graph_exec_traces]
      if tensor_debug_mode == "NO_TENSOR":
        # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought
        # to be an empty float32 tensor.
        for tensor_value in tensor_values:
          self.assertAllEqual(tensor_value, [])
      elif tensor_debug_mode == "CURT_HEALTH":
        for trace in graph_exec_traces:
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          # 1st element: tensor_id; 2nd element: 0 indicating no inf or nan.
          self.assertAllClose(trace.debug_tensor_value, [tensor_id, 0.0])
      elif tensor_debug_mode == "FULL_TENSOR":
        less_values = [
            reader.graph_execution_trace_to_tensor_value(trace)
            for trace in graph_exec_traces if trace.op_type == "Less"]
        self.assertAllEqual(less_values, [True, True, True, True, False])
        mul_values = [
            reader.graph_execution_trace_to_tensor_value(trace)
            for trace in graph_exec_traces if trace.op_type == "Mul"]
        self.assertAllClose(mul_values, [1.0, 2.0, 4.0, 8.0])

  def testCallingEnableTracingTwiceWithTheSameDumpRootIsIdempotent(self):
    dumping_callback.enable_dump_debug_info(self.dump_root)
    writer = dumping_callback.enable_dump_debug_info(self.dump_root)

    x = constant_op.constant([10.0, 12.0, 10.0])
    for _ in range(2):
      array_ops.unique(x)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      executions = reader.executions()
      self.assertLen(executions, 2)
      for execution in executions:
        self.assertGreater(execution.wall_time, 0)
        self.assertEqual(execution.op_type, "Unique")
        self.assertEqual(execution.num_outputs, 2)
        _, stack_frames = reader.read_execution_stack_trace(execution)
        self._verifyStackFrames(stack_frames)

  def testCallingEnableTracingTwiceWithDifferentDumpRootsOverwrites(self):
    dumping_callback.enable_dump_debug_info(self.dump_root)
    new_dump_root = self.dump_root + "_new_dump_root"
    writer = dumping_callback.enable_dump_debug_info(new_dump_root)

    x = constant_op.constant([10.0, 12.0, 10.0])
    for _ in range(2):
      array_ops.unique(x)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(new_dump_root) as reader:
      reader.update()
      executions = reader.executions()
      self.assertLen(executions, 2)
      for execution in executions:
        self.assertGreater(execution.wall_time, 0)
        self.assertEqual(execution.op_type, "Unique")
        self.assertEqual(execution.num_outputs, 2)
        _, stack_frames = reader.read_execution_stack_trace(execution)
        self._verifyStackFrames(stack_frames)

    with debug_events_reader.DebugDataReader(
        self.dump_root) as old_dump_root_reader:
      old_dump_root_reader.update()
      # The old dump root shouldn't have been written to.
      self.assertEqual(old_dump_root_reader.num_executions(), 0)
      self.assertFalse(old_dump_root_reader.outermost_graphs())

  def testCallingEnableRepeatedlyWithDifferentTensorDebugMode(self):
    """Assert calling enable_dump_debug_info() with two tensor-debug modes.

    It should lead to overwriting of the previously-configured mode.
    """
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="NO_TENSOR")

    @def_function.function
    def add_1_divide_by_2(x):
      return (x + 1.0) / 2.0

    self.assertAllClose(add_1_divide_by_2(constant_op.constant(4.0)), 2.5)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_exec_digests = reader.graph_execution_traces(digest=True)
      tensor_values = [reader.graph_execution_trace_to_tensor_value(digest)
                       for digest in graph_exec_digests]
      for tensor_value in tensor_values:
        # Under NO_TENSOR mode, each tensor is summarized as an empty float32
        # array.
        self.assertAllEqual(tensor_value, [])

    with self.assertRaisesRegexp(
        ValueError, r"already.*NO_TENSOR.*FULL_TENSOR.*not be honored"):
      dumping_callback.enable_dump_debug_info(
          self.dump_root, tensor_debug_mode="FULL_TENSOR")

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("FullTensor", "FULL_TENSOR"),
  )
  def testDisableTracingWorks(self, tensor_debug_mode):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)
    dumping_callback.disable_dump_debug_info()

    x = constant_op.constant([10.0, 12.0, 10.0])
    for _ in range(2):
      array_ops.unique(x)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      self.assertEqual(reader.num_executions(), 0)
      self.assertEqual(reader.num_graph_execution_traces(), 0)
      self.assertFalse(reader.outermost_graphs())

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("CurtHealth", "CURT_HEALTH"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("FullHealth", "FULL_HEALTH"),
      ("Shape", "SHAPE"),
      ("FullTensor", "FULL_TENSOR"),
  )
  def testMultiThreadedExecutionWithSameSetting(self, tensor_debug_mode):
    """Dumping from multiple threads using the same setting."""
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)
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

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      exec_digests = reader.executions(digest=True)
      prev_wall_time = 1
      for exec_digest in exec_digests:
        self.assertGreaterEqual(exec_digest.wall_time, prev_wall_time)
        prev_wall_time = exec_digest.wall_time

      graph_exec_traces = reader.graph_execution_traces()
      executed_op_types = [trace.op_type for trace in graph_exec_traces]
      self.assertEqual(executed_op_types.count("Mul"), 1 + num_threads)
      self.assertEqual(
          executed_op_types.count("ReadVariableOp"), 2 * (1 + num_threads))
      for trace in graph_exec_traces:
        # These are all single-output tensors.
        self.assertEqual(trace.output_slot, 0)

    tensor_values = [reader.graph_execution_trace_to_tensor_value(trace)
                     for trace in graph_exec_traces]
    if tensor_debug_mode == "NO_TENSOR":
      for tensor_value in tensor_values:
        self.assertAllEqual(tensor_value, [])
    elif tensor_debug_mode == "CURT_HEALTH":
      for trace in graph_exec_traces:
        tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
        # 1st element: tensor ID; 2nd element: 0 indicating no inf or nan.
        self.assertAllClose(trace.debug_tensor_value, [tensor_id, 0])
    elif tensor_debug_mode == "CONCISE_HEALTH":
      for trace in graph_exec_traces:
        tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
        # 1st element: tensor ID.
        # 2nd element: element count. Remaining elements: all zero because there
        # is no -inf, inf or nan.
        self.assertAllClose(trace.debug_tensor_value, [tensor_id, 1, 0, 0, 0])
    elif tensor_debug_mode == "FULL_HEALTH":
      for trace in graph_exec_traces:
        tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
        # Elements: [
        #   -1 is the unset tensor_id for eager op execution,
        #   device ID (set to -1 for now),
        #   dtype, rank, element_count,
        #   neg_inf_count, pos_inf_count, nan_count
        #   neg_finite_count, zero_count, pos_finite_count]
        self.assertAllClose(
            trace.debug_tensor_value,
            [tensor_id, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
    elif tensor_debug_mode == "SHAPE":
      for trace in graph_exec_traces:
        if trace.op_type == "Mul":
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          mul_value = reader.graph_execution_trace_to_tensor_value(trace)
          # 1st element: tensor_id, should be >= 0.
          # 2nd element: dtype enum value (float32).
          # 3rd element: rank.
          # 4th element: element count.
          self.assertAllClose(mul_value, [tensor_id, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    elif tensor_debug_mode == "FULL_TENSOR":
      mul_values = [
          reader.graph_execution_trace_to_tensor_value(trace)
          for trace in graph_exec_traces if trace.op_type == "Mul"]
      self.assertAllClose(mul_values, [6.0, 6.0, 6.0, 6.0])

  def testMultiThreadedDumpingWithDifferentSettings(self):
    dump_root_1 = os.path.join(self.dump_root, "dump_root_1")
    dump_root_2 = os.path.join(self.dump_root, "dump_root_2")
    v1 = variables.Variable(10.0, dtype=dtypes.float32)
    v2 = variables.Variable(3.0, dtype=dtypes.float32)

    def add_negative_v1_squared_to_itself():
      writer = dumping_callback.enable_dump_debug_info(
          dump_root_1, tensor_debug_mode="FULL_TENSOR")
      # Run in a loop to facilitate interleaving between threads.
      for _ in range(3):
        v1.assign_add(-(v1 ** 2.0))
      writer.FlushNonExecutionFiles()
      writer.FlushExecutionFiles()

    def add_negative_v2_squared_to_itself():
      writer = dumping_callback.enable_dump_debug_info(
          dump_root_2, tensor_debug_mode="FULL_TENSOR")
      v2_squared = v2 ** 2.0
      # Since dumping is disabled before the Neg op is called, no tensor data
      # should be dumped from the op, but this shouldn't affect the dumping of
      # the tensor data from the Neg op in `add_negative_v1_squared_to_itself`.
      # Both behavior is checked below.
      dumping_callback.disable_dump_debug_info()
      negative_v2_squared = -v2_squared
      v2.assign_add(negative_v2_squared)
      writer.FlushNonExecutionFiles()
      writer.FlushExecutionFiles()

    # v2 is mutated on a sub-thread.
    sub_thread = threading.Thread(target=add_negative_v2_squared_to_itself)
    sub_thread.start()
    add_negative_v1_squared_to_itself()  # v1 is mutated on the main thread.
    sub_thread.join()
    # 10 - 10 * 10 = -90.
    # -90 - (-90 * -90) = -8190.
    # -8190 - (-8190 * -8190) = -67084290.
    self.assertAllClose(v1.read_value(), -67084290.0)
    self.assertAllClose(v2.read_value(), -6.0)

    with debug_events_reader.DebugDataReader(dump_root_1) as reader:
      reader.update()
      exec_digests = reader.executions(digest=True)
      v1_squared_values = [
          reader.execution_to_tensor_values(digest)
          for digest in exec_digests if digest.op_type == "Pow"]
      negative_v1_squared_values = [
          reader.execution_to_tensor_values(digest)
          for digest in exec_digests if digest.op_type == "Neg"]
      self.assertAllClose(v1_squared_values, [[100.0], [8100.0], [67076100.0]])
      self.assertAllClose(
          negative_v1_squared_values, [[-100.0], [-8100.0], [-67076100.0]])

    with debug_events_reader.DebugDataReader(dump_root_2) as reader:
      reader.update()
      exec_digests = reader.executions(digest=True)
      executed_op_types = [digest.op_type for digest in exec_digests]
      self.assertNotIn("Neg", executed_op_types)
      v2_squared_values = [
          reader.execution_to_tensor_values(digest)
          for digest in exec_digests if digest.op_type == "Pow"]
      self.assertAllClose(v2_squared_values, [[9.0]])

  @test_util.run_in_graph_and_eager_modes
  def testNestedContextIsCapturedByGraphOpCreationHistory(self):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="NO_TENSOR")

    @def_function.function
    def iterative_doubling(x, times):
      i = constant_op.constant(0, dtype=dtypes.int32)
      while i < times:
        x = x * 2.0 - 1.0
        i += 1
      return x

    x = constant_op.constant(2.0, dtype=dtypes.float32)
    times = constant_op.constant(4, dtype=dtypes.int32)
    # 2 * 2 - 1 = 3; 3 * 2 - 1 = 5; 5 * 2 - 1 = 9; 9 * 2 - 1 = 17.
    self.assertAllClose(self.evaluate(iterative_doubling(x, times)), 17.0)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()
    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      less_op_digest = reader.graph_op_digests(op_type="Less")[-1]
      mul_op_digest = reader.graph_op_digests(op_type="Mul")[-1]
      sub_op_digest = reader.graph_op_digests(op_type="Sub")[-1]
      # The Less op is from the while-loop cond context and hence should have
      # a different innermost context ID from the mul and sub ops, which are
      # both from the while-loop body context.
      self.assertNotEqual(less_op_digest.graph_id, mul_op_digest.graph_id)
      self.assertNotEqual(less_op_digest.graph_id, sub_op_digest.graph_id)
      # The Mul and Sub ops are from the same innermost context.
      self.assertEqual(mul_op_digest.graph_id, sub_op_digest.graph_id)

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("FullTensor", "FULL_TENSOR"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testSimpleKerasRecurrentModelPredict(self, tensor_debug_mode):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)
    model = _create_simple_recurrent_keras_model([3, 4])
    batch_size = 5
    xs = np.ones([batch_size, 3, 4])
    self.assertAllClose(model.predict(xs), np.zeros([batch_size, 1]))

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      if context.executing_eagerly():
        # NOTE(b/142486213): Execution of the TF function happens with
        # Session.run() in v1 graph mode, hence it doesn't get logged to the
        # .execution file.
        self.assertTrue(reader.executions(digest=True))

      graph_exec_digests = reader.graph_execution_traces(digest=True)
      executed_op_types = [digest.op_type for digest in graph_exec_digests]
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
      tensor_values = [reader.graph_execution_trace_to_tensor_value(digest)
                       for digest in graph_exec_digests]
      if tensor_debug_mode == "NO_TENSOR":
        for tensor_value in tensor_values:
          self.assertAllEqual(tensor_value, [])
      else:
        # Refrain from asserting the internal implementation details of the LSTM
        # layer.
        self.assertTrue(any(
            bool(tensor_value.size) for tensor_value in tensor_values))

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("FullTensor", "FULL_TENSOR"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testSimpleKerasRecurrentModelFit(self, tensor_debug_mode):
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)
    model = _create_simple_recurrent_keras_model([3, 4])
    xs = np.ones([5, 3, 4])
    ys = np.ones([5, 1])

    history = model.fit(xs, ys, epochs=3, verbose=0)
    self.assertAllClose(
        history.history["loss"], [1.0, 0.9603999853134155, 0.9223681688308716])

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      if context.executing_eagerly():
        exec_digests = reader.executions(digest=True)
        self.assertTrue(exec_digests)
        if tensor_debug_mode == "NO_TENSOR":
          for digest in exec_digests:
            tensor_values = reader.execution_to_tensor_values(digest)
            for tensor_value in tensor_values:
              self.assertEqual(tensor_value, [])

      graph_exec_digests = reader.graph_execution_traces(digest=True)
      executed_op_types = [digest.op_type for digest in graph_exec_digests]
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
      if tensor_debug_mode == "NO_TENSOR":
        for digest in graph_exec_digests:
          tensor_values = reader.graph_execution_trace_to_tensor_value(digest)
          for tensor_value in tensor_values:
            self.assertEqual(tensor_value, [])

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("FullTensor", "FULL_TENSOR"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testMobileNetV2Fit(self, tensor_debug_mode):
    """Test training Keras MobileNetV2 works with dumping."""
    # Use a large circular-buffer to make sure we capture all the executed ops.
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root,
        tensor_debug_mode=tensor_debug_mode,
        circular_buffer_size=100000)
    model = mobilenet_v2.MobileNetV2(
        input_shape=(32, 32, 3), alpha=0.1, weights=None)
    y = model.layers[22].output
    y = core.Flatten()(y)
    y = core.Dense(1)(y)
    model = models.Model(inputs=model.inputs, outputs=y)

    batch_size = 2
    xs = np.zeros([batch_size] + list(model.input_shape[1:]))
    ys = np.zeros([batch_size] + list(model.output_shape[1:]))
    model.compile(optimizer="sgd", loss="mse")
    epochs = 1
    history = model.fit(xs, ys, epochs=epochs, verbose=0)
    self.assertLen(history.history["loss"], epochs)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      if context.executing_eagerly():
        # NOTE(b/142486213): Execution of the TF function happens with
        # Session.run() in v1 graph mode, hence it doesn't get logged to the
        # .execution file.
        exec_digests = reader.executions(digest=True)
        self.assertTrue(exec_digests)

      graph_exec_digests = reader.graph_execution_traces()
      executed_op_types = [digest.op_type for digest in graph_exec_digests]
      # These are the ops that we can safely assume to have been executed during
      # the model's fit() call.
      self.assertIn("Conv2D", executed_op_types)
      self.assertIn("Relu6", executed_op_types)
      self.assertIn("Conv2DBackpropFilter", executed_op_types)
      self.assertIn("Relu6Grad", executed_op_types)

      if tensor_debug_mode == "NO_TENSOR":
        # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought
        # to be an empty float32 tensor.
        tensor_values = [
            reader.graph_execution_trace_to_tensor_value(digest)
            for digest in graph_exec_digests]
        for tensor_value in tensor_values:
          self.assertAllEqual(tensor_value, [])
      elif tensor_debug_mode == "FULL_TENSOR":
        conv2d_values = [
            reader.graph_execution_trace_to_tensor_value(digest)
            for digest in graph_exec_digests if digest.op_type == "Conv2D"]
        self.assertTrue(conv2d_values)
        for conv2d_value in conv2d_values:
          self.assertGreater(len(conv2d_value.shape), 1)
          self.assertEqual(conv2d_value.shape[0], batch_size)
        relu6_values = [
            reader.graph_execution_trace_to_tensor_value(digest)
            for digest in graph_exec_digests if digest.op_type == "Relu6"]
        self.assertTrue(relu6_values)
        for relu6_value in relu6_values:
          self.assertGreater(len(relu6_value.shape), 1)
          self.assertEqual(relu6_value.shape[0], batch_size)
        conv2d_bp_filter_values = [
            reader.graph_execution_trace_to_tensor_value(digest)
            for digest in graph_exec_digests
            if digest.op_type == "Conv2DBackpropFilter"]
        self.assertTrue(conv2d_bp_filter_values)
        for conv2d_bp_filter_value in conv2d_bp_filter_values:
          self.assertGreater(len(conv2d_bp_filter_value.shape), 1)
        relu6_grad_values = [
            reader.graph_execution_trace_to_tensor_value(digest)
            for digest in graph_exec_digests if digest.op_type == "Relu6Grad"]
        self.assertTrue(relu6_grad_values)
        for relu6_grad_value in relu6_grad_values:
          self.assertGreater(len(relu6_grad_value.shape), 1)


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
