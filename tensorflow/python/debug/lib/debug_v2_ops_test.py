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
"""Test for the internal ops used by tfdbg v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.compat import compat
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import dumping_callback_test_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class DebugIdentityV2OpTest(dumping_callback_test_lib.DumpingCallbackTestBase):
  """Tests for DebugIdentityV2Op: when DebugEventsWriter is initialized.

  DebugEventsWriter being initialized prior to DebugIdentityV2 ops being invoked
  for the first time is the typical case (e.g., tfdbg2 running on a local
  machine with only local devices.)
  """

  def setUp(self):
    super(DebugIdentityV2OpTest, self).setUp()
    # Testing using a small circular-buffer size.
    self.circular_buffer_size = 4
    self.tfdbg_run_id = "test_tfdbg_run"
    self.writer = debug_events_writer.DebugEventsWriter(
        self.dump_root, self.tfdbg_run_id, self.circular_buffer_size)

  def tearDown(self):
    self.writer.Close()
    super(DebugIdentityV2OpTest, self).tearDown()

  @test_util.run_in_graph_and_eager_modes
  def testSingleTensorFullTensorDebugModeWithCircularBufferBehavior(self):

    @def_function.function
    def write_debug_trace(x):
      square = math_ops.square(x)
      gen_debug_ops.debug_identity_v2(
          square,
          tfdbg_context_id="deadbeaf",
          op_name="Square",
          output_slot=0,
          tensor_debug_mode=debug_event_pb2.TensorDebugMode.FULL_TENSOR,
          debug_urls=["file://%s" % self.dump_root])

      sqrt = math_ops.sqrt(x)
      gen_debug_ops.debug_identity_v2(
          sqrt,
          tfdbg_context_id="beafdead",
          op_name="Sqrt",
          output_slot=0,
          tensor_debug_mode=debug_event_pb2.TensorDebugMode.FULL_TENSOR,
          debug_urls=["file://%s" % self.dump_root])
      return square + sqrt

    x = np.array([3.0, 4.0])
    # Only the graph-execution trace of the last iteration should be written
    # to self.dump_root.
    for _ in range(self.circular_buffer_size // 2 + 1):
      self.assertAllClose(
          write_debug_trace(x), [9.0 + np.sqrt(3.0), 16.0 + 2.0])

    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      # Check that the .metadata DebugEvents data file has been created, even
      # before FlushExecutionFiles() is called.
      self.assertGreater(reader.starting_wall_time(), 0)
      self.assertTrue(reader.tensorflow_version())
      self.assertTrue(reader.tfdbg_file_version().startswith("debug.Event"))

      graph_trace_iter = reader.graph_execution_traces_iterators()[0]
      # Before FlushExecutionFiles() is called, the .graph_execution_traces file
      # ought to be empty.
      with self.assertRaises(StopIteration):
        next(graph_trace_iter)

      # Flush the circular buffer.
      self.writer.FlushExecutionFiles()
      graph_trace_iter = reader.graph_execution_traces_iterators()[0]

      # The circular buffer has a size of 4. So only the data from the
      # last two iterations should have been written to self.dump_root.
      for _ in range(2):
        debug_event = next(graph_trace_iter).debug_event
        self.assertGreater(debug_event.wall_time, 0)
        trace = debug_event.graph_execution_trace
        self.assertEqual(trace.tfdbg_context_id, "deadbeaf")
        self.assertEqual(trace.op_name, "Square")
        self.assertEqual(trace.output_slot, 0)
        self.assertEqual(trace.tensor_debug_mode,
                         debug_event_pb2.TensorDebugMode.FULL_TENSOR)
        tensor_value = tensor_util.MakeNdarray(trace.tensor_proto)
        self.assertAllClose(tensor_value, [9.0, 16.0])

        debug_event = next(graph_trace_iter).debug_event
        self.assertGreater(debug_event.wall_time, 0)
        trace = debug_event.graph_execution_trace
        self.assertEqual(trace.tfdbg_context_id, "beafdead")
        self.assertEqual(trace.op_name, "Sqrt")
        self.assertEqual(trace.output_slot, 0)
        self.assertEqual(trace.tensor_debug_mode,
                         debug_event_pb2.TensorDebugMode.FULL_TENSOR)
        tensor_value = tensor_util.MakeNdarray(trace.tensor_proto)
        self.assertAllClose(tensor_value, [np.sqrt(3.0), 2.0])

      # Only the graph-execution trace of the last iteration should be written
      # to self.dump_root.
      with self.assertRaises(StopIteration):
        next(graph_trace_iter)

  @test_util.run_in_graph_and_eager_modes
  def testControlFlow(self):

    @def_function.function
    def collatz(x):
      counter = constant_op.constant(0, dtype=dtypes.int32)
      while math_ops.greater(x, 1):
        counter = counter + 1
        gen_debug_ops.debug_identity_v2(
            x,
            tfdbg_context_id="deadbeaf",
            op_name="x",
            output_slot=0,
            tensor_debug_mode=debug_event_pb2.TensorDebugMode.FULL_TENSOR,
            debug_urls=["file://%s" % self.dump_root])
        if math_ops.equal(x % 2, 0):
          x = math_ops.div(x, 2)
        else:
          x = x * 3 + 1
      return counter

    x = constant_op.constant(10, dtype=dtypes.int32)
    self.evaluate(collatz(x))

    self.writer.FlushExecutionFiles()
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      graph_trace_iter = reader.graph_execution_traces_iterators()[0]
      try:
        x_values = []
        timestamp = 0
        while True:
          debug_event = next(graph_trace_iter).debug_event
          self.assertGreater(debug_event.wall_time, timestamp)
          timestamp = debug_event.wall_time
          trace = debug_event.graph_execution_trace
          self.assertEqual(trace.tfdbg_context_id, "deadbeaf")
          self.assertEqual(trace.op_name, "x")
          self.assertEqual(trace.output_slot, 0)
          self.assertEqual(trace.tensor_debug_mode,
                           debug_event_pb2.TensorDebugMode.FULL_TENSOR)
          x_values.append(int(tensor_util.MakeNdarray(trace.tensor_proto)))
      except StopIteration:
        pass

      # Due to the circular buffer, only the last 4 iterations of
      # [10, 5, 16, 8, 4, 2] should have been written.
      self.assertAllEqual(x_values, [16, 8, 4, 2])

  @test_util.run_in_graph_and_eager_modes
  def testTwoDumpRoots(self):
    another_dump_root = os.path.join(self.dump_root, "another")
    another_debug_url = "file://%s" % another_dump_root
    another_writer = debug_events_writer.DebugEventsWriter(
        another_dump_root, "test_tfdbg_run")

    @def_function.function
    def write_debug_trace(x):
      # DebugIdentityV2 is a stateful op. It ought to be included by auto
      # control dependency.
      square = math_ops.square(x)
      gen_debug_ops.debug_identity_v2(
          square,
          tfdbg_context_id="deadbeaf",
          tensor_debug_mode=debug_event_pb2.TensorDebugMode.FULL_TENSOR,
          debug_urls=["file://%s" % self.dump_root, another_debug_url])
      return square + 1.0

    x = np.array([3.0, 4.0])
    self.assertAllClose(write_debug_trace(x), np.array([10.0, 17.0]))
    self.writer.FlushExecutionFiles()
    another_writer.FlushExecutionFiles()
    another_writer.Close()

    for debug_root in (self.dump_root, another_dump_root):
      with debug_events_reader.DebugEventsReader(debug_root) as reader:
        graph_trace_iter = reader.graph_execution_traces_iterators()[0]

        debug_event = next(graph_trace_iter).debug_event
        trace = debug_event.graph_execution_trace
        self.assertEqual(trace.tfdbg_context_id, "deadbeaf")
        self.assertEqual(trace.op_name, "")
        self.assertEqual(trace.tensor_debug_mode,
                         debug_event_pb2.TensorDebugMode.FULL_TENSOR)
        tensor_value = tensor_util.MakeNdarray(trace.tensor_proto)
        self.assertAllClose(tensor_value, [9.0, 16.0])

        with self.assertRaises(StopIteration):
          next(graph_trace_iter)


class DebugIdentityV2OpUninitializedWriterTest(
    dumping_callback_test_lib.DumpingCallbackTestBase):
  """Tests for DebugIdentityV2Op: when DebugEventsWriter is not initialized.

  This case can occur when DebugIdentityV2Ops are running on a remote
  TensorFlow server (e.g., a TPU worker).
  """

  @test_util.run_in_graph_and_eager_modes
  def testInvokingDebugIdentityV2OpBeforeCreatingDebugEventsWriterWorks(self):
    if not compat.forward_compatible(2020, 6, 24):
      self.skipTest("Functionality currently not supported.")
    circular_buffer_size = 3

    @def_function.function
    def write_debug_trace(x):
      # DebugIdentityV2 is a stateful op. It ought to be included by auto
      # control dependency.
      square = math_ops.square(x)
      gen_debug_ops.debug_identity_v2(
          square,
          tfdbg_context_id="deadbeaf",
          op_name="Square",
          output_slot=0,
          tensor_debug_mode=debug_event_pb2.TensorDebugMode.FULL_TENSOR,
          debug_urls=["file://%s" % self.dump_root],
          circular_buffer_size=circular_buffer_size)
      return square

    # The DebugIdentityV2 ops are invokes *before* a DebugEventsWriter at the
    # same dump root is created.
    for i in range(circular_buffer_size * 2):
      self.assertAllClose(
          write_debug_trace(np.array([i]).astype(np.float32)), [i**2.0])
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   "test_tfdbg_run",
                                                   circular_buffer_size)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      graph_trace_iter = reader.graph_execution_traces_iterators()[0]
      graph_execution_traces = []
      while True:
        try:
          graph_execution_traces.append(
              next(graph_trace_iter).debug_event.graph_execution_trace)
        except StopIteration:
          break
      self.assertLen(graph_execution_traces, circular_buffer_size)
      for i in range(circular_buffer_size):
        self.assertAllClose(
            tensor_util.MakeNdarray(graph_execution_traces[i].tensor_proto),
            [(i + circular_buffer_size)**2.0])


class DebugNumericSummaryV2Test(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpReduceInfNanThreeSlots(self):

    def debug_summary(x):
      return self.evaluate(gen_debug_ops.debug_numeric_summary_v2(
          x, tensor_debug_mode=(
              debug_event_pb2.TensorDebugMode.REDUCE_INF_NAN_THREE_SLOTS)))

    self.assertAllEqual(
        debug_summary(constant_op.constant([])), [0.0, 0.0, 0.0])
    self.assertAllEqual(
        debug_summary(constant_op.constant(42.0)), [0.0, 0.0, 0.0])
    self.assertAllEqual(
        debug_summary(constant_op.constant([3.0, 4.0])), [0.0, 0.0, 0.0])
    self.assertAllEqual(
        debug_summary(constant_op.constant(np.array([3.0, -np.inf]))),
        [-np.inf, 0.0, 0.0])
    self.assertAllEqual(
        debug_summary(constant_op.constant(np.array([[0, 0], [np.nan, 0]]))),
        [0.0, 0.0, np.nan])
    self.assertAllEqual(
        debug_summary(
            constant_op.constant(np.array([[0, 0], [np.nan, np.inf]]))),
        [0.0, np.inf, np.nan])
    self.assertAllEqual(
        debug_summary(
            constant_op.constant(np.array([[0, np.inf], [np.nan, -np.inf]]))),
        [-np.inf, np.inf, np.nan])

    x = np.zeros([100, 100], dtype=np.float16)
    x[32, 47] = np.nan
    self.assertAllEqual(
        debug_summary(constant_op.constant(x)), [0.0, 0.0, np.nan])
    x = np.zeros([97, 97], dtype=np.float32)
    x[50, 83] = -np.inf
    self.assertAllEqual(
        debug_summary(constant_op.constant(x)), [-np.inf, 0.0, 0.0])
    x[1, 41] = np.nan
    self.assertAllEqual(
        debug_summary(constant_op.constant(x)), [-np.inf, 0.0, np.nan])
    x = np.zeros([9701], dtype=np.float64)
    x[9700] = np.nan
    self.assertAllEqual(
        debug_summary(constant_op.constant(x)), [0.0, 0.0, np.nan])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpLargeTensorIDError(self):
    modes = [
        debug_event_pb2.TensorDebugMode.CURT_HEALTH,
        debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
        debug_event_pb2.TensorDebugMode.SHAPE,
    ]
    # Maximum allowed tensor_id
    tensor_id = np.power(2, 53, dtype=np.int64)
    for mode in modes:
      self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              constant_op.constant(42.0),
              tensor_debug_mode=mode,
              tensor_id=tensor_id,
              output_dtype=dtypes.float64))
    # Incrementing by one should error
    tensor_id += 1
    for mode in modes:
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(
            gen_debug_ops.debug_numeric_summary_v2(
                constant_op.constant(42.0),
                tensor_debug_mode=mode,
                tensor_id=tensor_id,
                output_dtype=dtypes.float64))

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpCurtHealthValuesSmall(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.CURT_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    tensor, tensor_id = debug_summary(constant_op.constant([]))
    self.assertAllEqual(tensor, [tensor_id, 0.0])

    tensor, tensor_id = debug_summary(constant_op.constant(42.0))
    self.assertAllEqual(tensor, [tensor_id, 0.0])

    tensor, tensor_id = debug_summary(constant_op.constant([3.0, 4.0]))
    self.assertAllEqual(tensor, [tensor_id, 0.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([3.0, -np.inf])))
    self.assertAllEqual(tensor, [tensor_id, 1.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, 0], [np.nan, 0]])))
    self.assertAllEqual(tensor, [tensor_id, 1.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, 0], [np.nan, np.inf]])))
    self.assertAllEqual(tensor, [tensor_id, 1.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, np.inf], [np.nan, -np.inf]])))
    self.assertAllEqual(tensor, [tensor_id, 1.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpCurtHealthValuesLarge(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.CURT_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    x = np.zeros([100, 100], dtype=np.float16)
    x[32, 47] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 1.0])
    x = np.zeros([97, 97], dtype=np.float32)
    x[50, 83] = -np.inf
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 1.0])
    x[1, 41] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 1.0])
    x = np.zeros([9701], dtype=np.float64)
    x[9700] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 1.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpCurtHealthConsistency(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.CURT_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    x = np.zeros([100, 100], dtype=np.float16)
    x[43, 99] = np.nan
    c = constant_op.constant(x)
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

    x = np.zeros([100, 100, 50], dtype=np.float64)
    x[0, 0, 1] = np.inf
    c = constant_op.constant(x)
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

    c = constant_op.constant(np.ones((100, 200), np.double))
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpConciseHealthSmall(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(
                  debug_event_pb2.TensorDebugMode.CONCISE_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    tensor, tensor_id = debug_summary(constant_op.constant([]))
    self.assertAllEqual(tensor, [tensor_id, 0.0, 0.0, 0.0, 0.0])

    tensor, tensor_id = debug_summary(constant_op.constant(42.0))
    self.assertAllEqual(tensor, [tensor_id, 1.0, 0.0, 0.0, 0.0])

    tensor, tensor_id = debug_summary(constant_op.constant([3.0, 4.0]))
    self.assertAllEqual(tensor, [tensor_id, 2.0, 0.0, 0.0, 0.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([3.0, -np.inf])))
    self.assertAllEqual(tensor, [tensor_id, 2.0, 1.0, 0.0, 0.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, 0], [np.nan, 0]])))
    self.assertAllEqual(tensor, [tensor_id, 4.0, 0.0, 0.0, 1.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, 0], [np.nan, np.inf]])))
    self.assertAllEqual(tensor, [tensor_id, 4.0, 0.0, 1.0, 1.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, np.inf], [np.nan, -np.inf]])))
    self.assertAllEqual(tensor, [tensor_id, 4.0, 1.0, 1.0, 1.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpConciseHealthLarge(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(
                  debug_event_pb2.TensorDebugMode.CONCISE_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    x = np.zeros([100, 100], dtype=np.float16)
    x[32, :] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 10000.0, 0.0, 0.0, 100.0])
    x = np.zeros([97, 97], dtype=np.float32)
    x[50, 83:85] = -np.inf
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 97 * 97, 2.0, 0.0, 0.0])
    x[1:9, 41] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 97 * 97, 2.0, 0.0, 8.0])
    x = np.zeros([9701], dtype=np.float64)
    x[9700] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 9701, 0.0, 0.0, 1.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpConciseHealthConsistency(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(
                  debug_event_pb2.TensorDebugMode.CONCISE_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    # Assert the same op is returns a consistent value
    x = np.zeros([100, 100], dtype=np.float16)
    x[3, 4] = -np.inf
    c = constant_op.constant(x)
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

    c = constant_op.constant(np.ones((100, 200), np.double))
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpShapeEmpty(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.SHAPE),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    tensor, tensor_id = debug_summary(constant_op.constant(0.0))
    self.assertAllEqual(
        tensor, [tensor_id, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpShapeSmall(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.SHAPE),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    x = np.zeros([3, 4], dtype=np.float32)
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(
        tensor, [tensor_id, 1.0, 2.0, 12.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])

    x = np.ones([1, 2, 3, 4, 5, 6], dtype=np.float16)
    x[0, 1, 2, 2, 2, 2] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(
        tensor,
        [tensor_id, 19, 6.0, 2 * 3 * 4 * 5 * 6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    x = np.zeros([2], dtype=np.float32)
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(
        tensor, [tensor_id, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    tensor, tensor_id = debug_summary(constant_op.constant([]))
    self.assertAllEqual(
        tensor, [tensor_id, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpShapeLarge(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.SHAPE),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    x = np.ones([1, 2, 3, 4, 5, 6, 7], dtype=np.double)
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [
        tensor_id, 2.0, 7.0, 2 * 3 * 4 * 5 * 6 * 7, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
    ])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpFullHealthSmall(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.FULL_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    tensor, tensor_id = debug_summary(constant_op.constant([]))
    expected = [tensor_id, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(constant_op.constant(42.0))
    expected = [tensor_id, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(constant_op.constant([3.0, 4.0]))
    expected = [tensor_id, -1, 1, 1, 2, 0, 0, 0, 0, 0, 2]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([3, -np.inf], dtype=np.float32)))
    expected = [tensor_id, -1, 1, 1, 2, 1, 0, 0, 0, 0, 1]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, 0], [np.nan, 0]], dtype=np.float64)))
    expected = [tensor_id, -1, 2, 2, 4, 0, 0, 1, 0, 3, 0]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(
        constant_op.constant(
            np.array([[0, 0], [np.nan, np.inf]], dtype=np.float16)))
    expected = [tensor_id, -1, 19, 2, 4, 0, 1, 1, 0, 2, 0]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(
        constant_op.constant(
            np.array([[0, np.inf], [np.nan, -np.inf]], dtype=np.float32)))
    expected = [tensor_id, -1, 1, 2, 4, 1, 1, 1, 0, 1, 0]
    self.assertAllEqual(tensor, expected)

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpFullHealthLarge(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.FULL_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    def tensor_counts(arr):
      counts = [len(np.shape(arr)), np.size(arr), 0, 0, 0, 0, 0, 0]
      for n in np.ravel(arr):
        if np.isneginf(n):
          counts[2] += 1
        elif np.isposinf(n):
          counts[3] += 1
        elif np.isnan(n):
          counts[4] += 1
        elif n < 0.:
          counts[5] += 1
        elif n == 0.:
          counts[6] += 1
        else:
          counts[7] += 1
      return counts

    x = np.zeros([50, 50], dtype=np.float16)
    x[32, 47] = np.nan
    x[0:4, 3] = np.inf
    x[40:50, 40:50] = 10
    x[3, 20] = -10
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    expected = [tensor_id, -1, 19] + tensor_counts(x)
    self.assertAllEqual(tensor, expected)

    x = np.ones([25, 25, 50], dtype=np.float32) * np.inf
    x[:, :, 1] = np.nan
    x[:, :, 2] = -np.inf
    x[:, :, 3] = -1
    x[:, :, 4] = 0
    x[:, :, 5] = 1
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    expected = [tensor_id, -1, 1] + tensor_counts(x)
    self.assertAllEqual(tensor, expected)
    x[0, 0, 0] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    expected = [tensor_id, -1, 1,] + tensor_counts(x)
    self.assertAllEqual(tensor, expected)
    x = np.zeros([9701], dtype=np.float64)
    x[9700] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    expected = [tensor_id, -1, 2] + tensor_counts(x)
    self.assertAllEqual(tensor, expected)

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpFullHealthConsistency(self):

    def debug_summary(x):
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.FULL_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    # Assert the same op is returns a consistent value
    x = np.zeros([100, 100], dtype=np.float16)
    x[32, 47] = np.nan
    x[0:4, 3] = np.inf
    x[90:100, 90:100] = 10
    x[3, 20] = -10
    c = constant_op.constant(x)
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

    x = np.ones((100, 200, 3, 10), np.double)
    x[1, 30, 2] = 10
    x[5, :, 0, 1] = np.nan
    x[90:100, 150, :, :] = np.inf
    c = constant_op.constant(x)
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

  def testCheckNumericsV2OpNegativeAndPositiveInf(self):
    """Test that CheckNumericsV2 op distinguishes negative and positive infs."""
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([-1.0, 1.0])
      t2 = constant_op.constant([0.0, 0.0])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"pass through test.*had -Inf and \+Inf values"):
        self.evaluate(
            array_ops.check_numerics_v2(t1 / t2, message="pass through test"))

  def testCheckNumericsV2OpNegativeAndPositiveInfAndNaN(self):
    """CheckNumericsV2 op distinguishes - & + infs when nan is present."""
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([-1.0, 1.0, 0.0])
      t2 = constant_op.constant([0.0, 0.0, 0.0])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"pass through test.*had -Inf, \+Inf, and NaN values"):
        self.evaluate(
            array_ops.check_numerics_v2(t1 / t2, message="pass through test"))

  def testCheckNumericsV2PositiveInfAndNaN(self):
    """Test that CheckNumericsV2 op shows sign of inf when nan is present."""
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([0.0, 1.0])
      t2 = constant_op.constant([0.0, 0.0])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"pass through test.*had \+Inf and NaN values"):
        self.evaluate(
            array_ops.check_numerics_v2(t1 / t2, message="pass through test"))


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
