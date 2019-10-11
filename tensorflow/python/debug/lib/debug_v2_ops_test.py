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

import glob
import os
import tempfile

import numpy as np

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


# TODO(cais): Refactor into own module when necessary.
class DebugEventsDir(object):

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
    self._prefix = metadata_paths[0][:-len(".metadata")]

    self._graph_execution_traces_path = ("%s.graph_execution_traces" %
                                         self._prefix)

  def metadata_iterator(self):
    for r in tf_record.tf_record_iterator(self._metadata_path):
      yield debug_event_pb2.DebugEvent.FromString(r)

  # TODO(cais): Add source_files_iterator()
  # TODO(cais): Add stack_frames_iterator()
  # TODO(cais): Add graphs_iterator()
  # TODO(cais): Add execution_iterator()

  def graph_execution_traces_iterator(self):
    if not os.path.isfile(self._graph_execution_traces_path):
      raise ValueError("DebugEvent data file does not exist: %s" %
                       self._graph_execution_traces_path)
    for r in tf_record.tf_record_iterator(self._graph_execution_traces_path):
      yield debug_event_pb2.DebugEvent.FromString(r)


class DebugIdentityV2OpTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(DebugIdentityV2OpTest, self).setUp()
    self.dump_root = tempfile.mkdtemp()
    # Testing using a small cyclic-buffer size.
    self.cyclic_buffer_size = 4
    self.writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                        self.cyclic_buffer_size)

  def tearDown(self):
    self.writer.Close()
    if os.path.isdir(self.dump_root):
      file_io.delete_recursively(self.dump_root)
    super(DebugIdentityV2OpTest, self).tearDown()

  @test_util.run_in_graph_and_eager_modes
  def testSingleTensorFullTensorDebugModeWithCyclicBufferBehavior(self):

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
    for _ in range(self.cyclic_buffer_size // 2 + 1):
      self.assertAllClose(
          write_debug_trace(x), [9.0 + np.sqrt(3.0), 16.0 + 2.0])

    debug_events_dir = DebugEventsDir(self.dump_root)
    metadata_iter = debug_events_dir.metadata_iterator()
    # Check that the .metadata DebugEvents data file has been created, even
    # before FlushExecutionFiles() is called.
    debug_event = next(metadata_iter)
    self.assertGreater(debug_event.wall_time, 0)
    self.assertTrue(debug_event.debug_metadata.tensorflow_version)
    self.assertTrue(
        debug_event.debug_metadata.file_version.startswith("debug.Event:"))

    graph_trace_iter = debug_events_dir.graph_execution_traces_iterator()
    # Before FlushExecutionFiles() is called, the .graph_execution_traces file
    # ought to be empty.
    with self.assertRaises(StopIteration):
      next(graph_trace_iter)

    # Flush the cyclic buffer.
    self.writer.FlushExecutionFiles()
    graph_trace_iter = debug_events_dir.graph_execution_traces_iterator()

    # The cyclic buffer has a size of 4. So only the data from the
    # last two iterations should have been written to self.dump_root.
    for _ in range(2):
      debug_event = next(graph_trace_iter)
      self.assertGreater(debug_event.wall_time, 0)
      trace = debug_event.graph_execution_trace
      self.assertEqual(trace.tfdbg_context_id, "deadbeaf")
      self.assertEqual(trace.op_name, "Square")
      self.assertEqual(trace.output_slot, 0)
      self.assertEqual(trace.tensor_debug_mode,
                       debug_event_pb2.TensorDebugMode.FULL_TENSOR)
      tensor_value = tensor_util.MakeNdarray(trace.tensor_proto)
      self.assertAllClose(tensor_value, [9.0, 16.0])

      debug_event = next(graph_trace_iter)
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
    debug_events_dir = DebugEventsDir(self.dump_root)
    graph_trace_iter = debug_events_dir.graph_execution_traces_iterator()
    try:
      x_values = []
      timestamp = 0
      while True:
        debug_event = next(graph_trace_iter)
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

    # Due to the cyclic buffer, only the last 4 iterations of
    # [10, 5, 16, 8, 4, 2] should have been written.
    self.assertAllEqual(x_values, [16, 8, 4, 2])

  @test_util.run_in_graph_and_eager_modes
  def testTwoDumpRoots(self):
    another_dump_root = os.path.join(self.dump_root, "another")
    another_debug_url = "file://%s" % another_dump_root
    another_writer = debug_events_writer.DebugEventsWriter(another_dump_root)

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
      debug_events_dir = DebugEventsDir(debug_root)
      graph_trace_iter = debug_events_dir.graph_execution_traces_iterator()

      debug_event = next(graph_trace_iter)
      trace = debug_event.graph_execution_trace
      self.assertEqual(trace.tfdbg_context_id, "deadbeaf")
      self.assertEqual(trace.op_name, "")
      self.assertEqual(trace.tensor_debug_mode,
                       debug_event_pb2.TensorDebugMode.FULL_TENSOR)
      tensor_value = tensor_util.MakeNdarray(trace.tensor_proto)
      self.assertAllClose(tensor_value, [9.0, 16.0])

      with self.assertRaises(StopIteration):
        next(graph_trace_iter)


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
