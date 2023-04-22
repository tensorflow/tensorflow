# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized

import numpy as np

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_monitors
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.debug.lib import dumping_callback_test_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class TestMonitor(debug_events_monitors.BaseMonitor):

  def __init__(self, debug_data_reader):
    super(TestMonitor, self).__init__(debug_data_reader)
    # Mapping execution index to Execution data objects.
    self.executions = dict()
    # Mapping graph execution trace index to GraphExecutionTrace data objects.
    self.graph_execution_traces = dict()

  def on_execution(self, execution_index, execution):
    if execution_index in self.executions:
      raise ValueError("Duplicate execution index: %d" % execution_index)
    self.executions[execution_index] = execution

  def on_graph_execution_trace(self, graph_execution_trace_index,
                               graph_execution_trace):
    if graph_execution_trace_index in self.graph_execution_traces:
      raise ValueError("Duplicate graph-execution-trace index: %d" %
                       graph_execution_trace_index)
    self.graph_execution_traces[
        graph_execution_trace_index] = graph_execution_trace


class DebugEventsMonitorTest(dumping_callback_test_lib.DumpingCallbackTestBase,
                             parameterized.TestCase):

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("FullHealth", "FULL_HEALTH"),
      ("FullTensor", "FULL_TENSOR"),
  )
  def testOnExecutionIsCalled(self, tensor_debug_mode):
    x = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float32)
    y = constant_op.constant([[-1], [1]], dtype=dtypes.float32)
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)
    math_ops.matmul(x, y)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      test_monitor = TestMonitor(reader)
      reader.update()
      self.assertLen(test_monitor.executions, 1)
      self.assertEmpty(test_monitor.graph_execution_traces)
      execution = test_monitor.executions[0]
      self.assertTrue(execution.wall_time)
      self.assertEqual(execution.op_type, "MatMul")
      self.assertLen(execution.output_tensor_device_ids, 1)
      self.assertLen(execution.input_tensor_ids, 2)
      self.assertLen(execution.output_tensor_ids, 1)
      self.assertEqual(execution.num_outputs, 1)
      self.assertEqual(execution.graph_id, "")
      if tensor_debug_mode == "NO_TENSOR":
        self.assertIsNone(execution.debug_tensor_values)
      elif tensor_debug_mode == "CONCISE_HEALTH":
        self.assertLen(execution.debug_tensor_values, 1)
        # [tensor_id, element_count, neg_inf_count, pos_inf_count, nan_count].
        self.assertLen(execution.debug_tensor_values[0], 5)
      elif tensor_debug_mode == "FULL_HEALTH":
        self.assertLen(execution.debug_tensor_values, 1)
        # [tensor_id, device_id, dtype, rank, element_count,
        #  neg_inf_count, pos_inf_count, nan_count,
        #  neg_finite_count, zero_count, pos_finite_count].
        self.assertLen(execution.debug_tensor_values[0], 11)
      elif tensor_debug_mode == "FULL_TENSOR":
        # Full tensor values are not stored in the debug_tensor_values field.
        self.assertIsNone(execution.debug_tensor_values)
        self.assertAllClose(
            reader.execution_to_tensor_values(execution), [[[1.], [1.]]])

  @parameterized.named_parameters(
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("FullHealth", "FULL_HEALTH"),
      ("FullTensor", "FULL_TENSOR"),
  )
  def testOnGraphExecutionTraceIsCalled(self, tensor_debug_mode):
    xs = constant_op.constant([2., 6., 8., 1., 2.], dtype=dtypes.float32)
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

    @def_function.function
    def unique_sum(xs):
      """Sum over the unique values, for testing."""
      unique_xs, indices = array_ops.unique(xs)
      return math_ops.reduce_sum(unique_xs), indices

    unique_sum(xs)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      test_monitor = TestMonitor(reader)
      reader.update()
      self.assertLen(test_monitor.executions, 1)

      execution = test_monitor.executions[0]
      self.assertTrue(execution.wall_time)
      self.assertStartsWith(execution.op_type, "__inference_unique_sum")
      self.assertLen(execution.output_tensor_device_ids, 2)
      self.assertLen(execution.input_tensor_ids, 1)
      self.assertLen(execution.output_tensor_ids, 2)
      self.assertEqual(execution.num_outputs, 2)
      self.assertTrue(execution.graph_id)

      traces = test_monitor.graph_execution_traces
      if tensor_debug_mode == "CONCISE_HEALTH":
        self.assertLen(traces, 3)  # [Placeholder:0, Unique:0 , Sum:0].
        self.assertEqual(traces[0].op_type, "Placeholder")
        self.assertEqual(traces[0].output_slot, 0)
        self.assertEqual(traces[1].op_type, "Unique")
        self.assertEqual(traces[1].output_slot, 0)
        # Unique:1 is not traced under CONCISE_HEALTH mode, as it's int-dtype.
        self.assertEqual(traces[2].op_type, "Sum")
        self.assertEqual(traces[2].output_slot, 0)
        # [tensor_id, element_count, neg_inf_count, pos_inf_count, nan_count].
        self.assertLen(traces[0].debug_tensor_value, 5)
        self.assertLen(traces[1].debug_tensor_value, 5)
        self.assertLen(traces[2].debug_tensor_value, 5)
      elif tensor_debug_mode == "FULL_HEALTH":
        self.assertLen(traces, 3)  # [Placeholder:0, Unique:0 , Sum:0].
        self.assertEqual(traces[0].op_type, "Placeholder")
        self.assertEqual(traces[0].output_slot, 0)
        self.assertEqual(traces[1].op_type, "Unique")
        self.assertEqual(traces[1].output_slot, 0)
        # Unique:1 is not traced under FULL_HEALTH mode, as it's int-dtype.
        self.assertEqual(traces[2].op_type, "Sum")
        self.assertEqual(traces[2].output_slot, 0)
        # [tensor_id, device_id, dtype, rank, element_count,
        #  neg_inf_count, pos_inf_count, nan_count,
        #  neg_finite_count, zero_count, pos_finite_count].
        self.assertLen(traces[0].debug_tensor_value, 11)
        self.assertLen(traces[1].debug_tensor_value, 11)
        self.assertLen(traces[2].debug_tensor_value, 11)
      elif tensor_debug_mode == "FULL_TENSOR":
        # [Placeholder:0, Unique:0, Unique:1, Const:0, Sum:0].
        self.assertLen(traces, 5)
        self.assertEqual(traces[0].op_type, "Placeholder")
        self.assertEqual(traces[0].output_slot, 0)
        self.assertIsNone(traces[0].debug_tensor_value)
        self.assertAllEqual(
            reader.graph_execution_trace_to_tensor_value(traces[0]),
            [2., 6., 8., 1., 2.])
        self.assertEqual(traces[1].op_type, "Unique")
        self.assertEqual(traces[1].output_slot, 0)
        self.assertIsNone(traces[1].debug_tensor_value)
        self.assertAllEqual(
            reader.graph_execution_trace_to_tensor_value(traces[1]),
            [2., 6., 8., 1.])
        self.assertEqual(traces[2].op_type, "Unique")
        self.assertEqual(traces[2].output_slot, 1)
        self.assertIsNone(traces[2].debug_tensor_value)
        self.assertAllEqual(
            reader.graph_execution_trace_to_tensor_value(traces[2]),
            [0, 1, 2, 3, 0])
        self.assertEqual(traces[3].op_type, "Const")
        self.assertEqual(traces[3].output_slot, 0)
        self.assertIsNone(traces[3].debug_tensor_value)
        self.assertAllClose(
            reader.graph_execution_trace_to_tensor_value(traces[3]), [0])
        self.assertEqual(traces[4].op_type, "Sum")
        self.assertEqual(traces[4].output_slot, 0)
        self.assertIsNone(traces[4].debug_tensor_value)
        self.assertAllClose(
            reader.graph_execution_trace_to_tensor_value(traces[4]), 17.)


class AlertDataObjectsTest(test_util.TensorFlowTestCase):
  """Unit tests for alert-class objects."""

  def testInfNanMonitor(self):
    alert = debug_events_monitors.InfNanAlert(
        1234,
        "FooOp",
        1,
        size=1000,
        num_neg_inf=5,
        num_pos_inf=10,
        num_nan=20,
        execution_index=777,
        graph_execution_trace_index=888)
    self.assertEqual(alert.wall_time, 1234)
    self.assertEqual(alert.op_type, "FooOp")
    self.assertEqual(alert.output_slot, 1)
    self.assertEqual(alert.size, 1000)
    self.assertEqual(alert.num_neg_inf, 5)
    self.assertEqual(alert.num_pos_inf, 10)
    self.assertEqual(alert.num_nan, 20)
    self.assertEqual(alert.execution_index, 777)
    self.assertEqual(alert.graph_execution_trace_index, 888)


class InfNanMonitorTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testInfNanMonitorStartsWithEmptyAlerts(self):
    mock_reader = test.mock.MagicMock()
    monitor = debug_events_monitors.InfNanMonitor(mock_reader)
    self.assertEmpty(monitor.alerts())

  def testInfNanMonitorOnExecutionUnderCurtHealthMode(self):
    mock_reader = test.mock.MagicMock()
    monitor = debug_events_monitors.InfNanMonitor(mock_reader)
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 1, "FooOp", output_tensor_device_ids=[0, 1])
    execution = debug_events_reader.Execution(
        execution_digest,
        "worker01", ["a1", "b2", "e3"],
        debug_event_pb2.TensorDebugMode.CURT_HEALTH,
        graph_id=None,
        input_tensor_ids=[12, 34],
        output_tensor_ids=[56, 78],
        debug_tensor_values=[[-1, 0], [-1, 1]])  # [tensor_id, any_inf_nan].
    monitor.on_execution(50, execution)

    self.assertLen(monitor.alerts(), 1)
    alert = monitor.alerts()[0]
    self.assertEqual(alert.wall_time, 1234)
    self.assertEqual(alert.op_type, "FooOp")
    self.assertEqual(alert.output_slot, 1)
    # The four fields below are unavailable under CURT_HEALTH mode by design.
    self.assertIsNone(alert.size)
    self.assertIsNone(alert.num_neg_inf)
    self.assertIsNone(alert.num_pos_inf)
    self.assertIsNone(alert.num_nan)
    self.assertEqual(alert.execution_index, 50)
    self.assertIsNone(alert.graph_execution_trace_index)

  @parameterized.named_parameters(
      ("ConciseHealth",
       debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
       # [tensor_id, size, num_neg_inf, num_pos_inf, num_nan].
       [[-1, 10, 1, 2, 3],
        [-1, 100, 0, 0, 0]]),
      ("FullHealth",
       debug_event_pb2.TensorDebugMode.FULL_HEALTH,
       # [tensor_id, device_id, dtype, rank, element_count,
       #  neg_inf_count, pos_inf_count, nan_count,
       #  neg_finite_count, zero_count, pos_finite_count].
       [[-1, -1, 1, 1, 10, 1, 2, 3, 0, 0, 0],
        [-1, -1, 1, 1, 100, 0, 0, 0, 10, 30, 60]]),
  )
  def testInfNanMonitorOnExecutionUnderHealthMode(self,
                                                  tensor_debug_mode,
                                                  debug_tensor_values):
    mock_reader = test.mock.MagicMock()
    monitor = debug_events_monitors.InfNanMonitor(mock_reader)
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 1, "BarOp", output_tensor_device_ids=[0, 1])

    execution = debug_events_reader.Execution(
        execution_digest,
        "worker01",
        ["a1", "b2", "e3"],
        tensor_debug_mode,
        graph_id=None,
        input_tensor_ids=[12, 34],
        output_tensor_ids=[56, 78],
        debug_tensor_values=debug_tensor_values)
    monitor.on_execution(60, execution)

    self.assertLen(monitor.alerts(), 1)
    alert = monitor.alerts()[0]
    self.assertEqual(alert.wall_time, 1234)
    self.assertEqual(alert.op_type, "BarOp")
    self.assertEqual(alert.output_slot, 0)
    self.assertEqual(alert.size, 10)
    self.assertEqual(alert.num_neg_inf, 1)
    self.assertEqual(alert.num_pos_inf, 2)
    self.assertEqual(alert.num_nan, 3)
    self.assertEqual(alert.execution_index, 60)
    self.assertIsNone(alert.graph_execution_trace_index)

  @parameterized.named_parameters(
      ("Shape",
       debug_event_pb2.TensorDebugMode.SHAPE,
       # [tensor_id, dtype, rank, element_cont, ...shape_truncate_6]
       [[-1, 1, 2, 6, 3, 2, 0, 0, 0, 0],
        [-1, 10, 1, 7, 7, 0, 0, 0, 0, 0]]),
  )
  def testInfNanMonitorOnExecutionUnderModeWithNoInfNanInfo(
      self,
      tensor_debug_mode,
      debug_tensor_values):
    mock_reader = test.mock.MagicMock()
    monitor = debug_events_monitors.InfNanMonitor(mock_reader)
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 1, "BarOp", output_tensor_device_ids=[0, 1])

    execution = debug_events_reader.Execution(
        execution_digest,
        "worker01",
        ["a1", "b2", "e3"],
        tensor_debug_mode,
        graph_id=None,
        input_tensor_ids=[12, 34],
        output_tensor_ids=[56, 78],
        debug_tensor_values=debug_tensor_values)
    monitor.on_execution(60, execution)

    self.assertEmpty(monitor.alerts())

  @parameterized.named_parameters(
      ("FloatsScalarWithInfAndNan", np.inf, np.float32, 1, 0, 1, 0),
      ("Floats2DWithInfAndNan", [[0, np.nan, np.nan, -np.inf]
                                ], np.float32, 4, 1, 0, 2),
      ("Floats1DWithoutInfOrNan", [0, -1e6, 1e6, 9e5], np.float32, 4, 0, 0, 0),
      ("Integers", [[0, 1000, -200, -300]], np.int32, 4, 0, 0, 0),
      ("Booleans", [False, True, False, False], np.int32, 4, 0, 0, 0),
  )
  def testInfNanMonitorOnExecutionUnderFullTensorModeWorks(
      self, tensor_value, dtype, expected_size, expected_num_neg_inf,
      expected_num_pos_inf, expected_num_nan):
    mock_reader = test.mock.MagicMock()
    mock_reader.execution_to_tensor_values.return_value = [
        np.array([[0.0, -1.0, 1.0]]),
        np.array(tensor_value, dtype=dtype)
    ]
    monitor = debug_events_monitors.InfNanMonitor(mock_reader)
    execution_digest = debug_events_reader.ExecutionDigest(
        1234,
        1,
        "__inference_bar_function_1234",
        output_tensor_device_ids=[0, 1])
    execution = debug_events_reader.Execution(
        execution_digest,
        "worker01", ["a1", "b2", "e3"],
        debug_event_pb2.TensorDebugMode.FULL_TENSOR,
        graph_id=None,
        input_tensor_ids=[12, 34],
        output_tensor_ids=[56, 78])
    monitor.on_execution(70, execution)

    if expected_num_neg_inf or expected_num_pos_inf or expected_num_nan:
      self.assertLen(monitor.alerts(), 1)
      alert = monitor.alerts()[0]
      self.assertEqual(alert.wall_time, 1234)
      self.assertEqual(alert.op_type, "__inference_bar_function_1234")
      self.assertEqual(alert.output_slot, 1)
      self.assertEqual(alert.size, expected_size)
      self.assertEqual(alert.num_neg_inf, expected_num_neg_inf)
      self.assertEqual(alert.num_pos_inf, expected_num_pos_inf)
      self.assertEqual(alert.num_nan, expected_num_nan)
      self.assertEqual(alert.execution_index, 70)
      self.assertIsNone(alert.graph_execution_trace_index, 70)
    else:
      self.assertEmpty(monitor.alerts())

  def testInfNaNMonitorOnGraphExecutionTraceCurtHealthMode(self):
    mock_reader = test.mock.MagicMock()
    monitor = debug_events_monitors.InfNanMonitor(mock_reader)
    trace_digest = debug_events_reader.GraphExecutionTraceDigest(
        1234, 1, "FooOp", "FooOp_1", 2, "g1")
    trace = debug_events_reader.GraphExecutionTrace(
        trace_digest, ["g0", "g1"],
        debug_event_pb2.TensorDebugMode.CURT_HEALTH,
        debug_tensor_value=[9, 1])  # [tensor_id, any_inf_nan].
    monitor.on_graph_execution_trace(55, trace)
    self.assertLen(monitor.alerts(), 1)
    alert = monitor.alerts()[0]
    self.assertEqual(alert.wall_time, 1234)
    self.assertEqual(alert.op_type, "FooOp")
    self.assertEqual(alert.output_slot, 2)
    # The four fields below are unavailable under CURT_HEALTH mode by design.
    self.assertIsNone(alert.size)
    self.assertIsNone(alert.num_neg_inf)
    self.assertIsNone(alert.num_pos_inf)
    self.assertIsNone(alert.num_nan)
    self.assertIsNone(alert.execution_index)
    self.assertEqual(alert.graph_execution_trace_index, 55)

  def testInfNaNMonitorOnGraphExecutionTraceConciseHealthMode(self):
    mock_reader = test.mock.MagicMock()
    monitor = debug_events_monitors.InfNanMonitor(mock_reader)
    trace_digest = debug_events_reader.GraphExecutionTraceDigest(
        1234, 1, "FooOp", "FooOp_1", 2, "g1")
    trace = debug_events_reader.GraphExecutionTrace(
        trace_digest,
        ["g0", "g1"],
        debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
        # [tensor_id, size, num_neg_inf, num_pos_inf, num_nan].
        debug_tensor_value=[9, 100, 3, 2, 1])
    monitor.on_graph_execution_trace(55, trace)

    self.assertLen(monitor.alerts(), 1)
    alert = monitor.alerts()[0]
    self.assertEqual(alert.wall_time, 1234)
    self.assertEqual(alert.op_type, "FooOp")
    self.assertEqual(alert.output_slot, 2)
    self.assertEqual(alert.size, 100)
    self.assertEqual(alert.num_neg_inf, 3)
    self.assertEqual(alert.num_pos_inf, 2)
    self.assertEqual(alert.num_nan, 1)
    self.assertEqual(alert.graph_execution_trace_index, 55)

  @parameterized.named_parameters(
      ("FloatsScalarWithInfAndNan", np.inf, np.float32, 1, 0, 1, 0),
      ("Floats2DWithInfAndNan", [[0, np.nan, np.nan, -np.inf]
                                ], np.float32, 4, 1, 0, 2),
      ("Floats1DWithoutInfOrNan", [0, -1e6, 1e6, 9e5], np.float32, 4, 0, 0, 0),
      ("Integers", [[0, 1000, -200, -300]], np.int32, 4, 0, 0, 0),
      ("Booleans", [False, True, False, False], np.int32, 4, 0, 0, 0),
  )
  def testInfNanMonitorOnGraphExecutionTraceUnderFullTensorModeWorks(
      self, tensor_value, dtype, expected_size, expected_num_neg_inf,
      expected_num_pos_inf, expected_num_nan):
    mock_reader = test.mock.MagicMock()
    mock_reader.graph_execution_trace_to_tensor_value.return_value = np.array(
        tensor_value, dtype=dtype)
    monitor = debug_events_monitors.InfNanMonitor(mock_reader)
    trace_digest = debug_events_reader.GraphExecutionTraceDigest(
        1234, 1, "BazOp", "name_scope_3/BazOp_1", 2, "g1")
    trace = debug_events_reader.GraphExecutionTrace(
        trace_digest, ["g0", "g1"], debug_event_pb2.TensorDebugMode.FULL_TENSOR)
    monitor.on_graph_execution_trace(80, trace)

    if expected_num_neg_inf or expected_num_pos_inf or expected_num_nan:
      self.assertLen(monitor.alerts(), 1)
      alert = monitor.alerts()[0]
      self.assertEqual(alert.wall_time, 1234)
      self.assertEqual(alert.op_type, "BazOp")
      self.assertEqual(alert.output_slot, 2)
      self.assertEqual(alert.size, expected_size)
      self.assertEqual(alert.num_neg_inf, expected_num_neg_inf)
      self.assertEqual(alert.num_pos_inf, expected_num_pos_inf)
      self.assertEqual(alert.num_nan, expected_num_nan)
      self.assertIsNone(alert.execution_index)
      self.assertEqual(alert.graph_execution_trace_index, 80)
    else:
      self.assertEmpty(monitor.alerts())

  def testLimitingInfNanMonitorAlertCountWorks(self):
    mock_reader = test.mock.MagicMock()
    monitor = debug_events_monitors.InfNanMonitor(mock_reader, limit=3)
    for i in range(10):
      execution_digest = debug_events_reader.ExecutionDigest(
          i * 1000, 1, "FooOp", output_tensor_device_ids=[0, 1])
      execution = debug_events_reader.Execution(
          execution_digest,
          "worker01", ["a1", "b2", "e3"],
          debug_event_pb2.TensorDebugMode.CURT_HEALTH,
          graph_id=None,
          input_tensor_ids=[12, 34],
          output_tensor_ids=[56, 78],
          debug_tensor_values=[[-1, 0], [-1, 1]])  # [tensor_id, any_inf_nan].
      monitor.on_execution(i, execution)

    alerts = monitor.alerts()
    self.assertLen(alerts, 3)
    for i, alert in enumerate(alerts):
      self.assertEqual(alert.wall_time, i * 1000)
      self.assertEqual(alert.op_type, "FooOp")
      self.assertEqual(alert.output_slot, 1)
      # The four fields below are unavailable under CURT_HEALTH mode by design.
      self.assertIsNone(alert.size)
      self.assertIsNone(alert.num_neg_inf)
      self.assertIsNone(alert.num_pos_inf)
      self.assertIsNone(alert.num_nan)
      self.assertEqual(alert.execution_index, i)
      self.assertIsNone(alert.graph_execution_trace_index)


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
