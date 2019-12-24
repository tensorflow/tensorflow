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
"""Shared library for testing tfdbg v2 dumping callback."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import shutil
import socket
import tempfile

from tensorflow.core.framework import types_pb2
from tensorflow.python.debug.lib import check_numerics_callback
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions


class DumpingCallbackTestBase(test_util.TensorFlowTestCase):
  """Base test-case class for tfdbg v2 callbacks."""

  def setUp(self):
    super(DumpingCallbackTestBase, self).setUp()
    self.dump_root = tempfile.mkdtemp()

  def tearDown(self):
    if os.path.isdir(self.dump_root):
      shutil.rmtree(self.dump_root, ignore_errors=True)
    check_numerics_callback.disable_check_numerics()
    dumping_callback.disable_dump_debug_info()
    super(DumpingCallbackTestBase, self).tearDown()

  def _readAndCheckMetadataFile(self):
    """Read and check the .metadata debug-events file."""
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      metadata_iter = reader.metadata_iterator()
      metadata = next(metadata_iter).debug_event.debug_metadata
      self.assertEqual(metadata.tensorflow_version, versions.__version__)
      self.assertTrue(metadata.file_version.startswith("debug.Event"))

  def _readAndCheckSourceFilesAndStackFrames(self):
    """Read and verify the .source_files & .stack_frames debug-event files.

    Returns:
      An OrderedDict mapping stack frame IDs to stack frames (FileLineCol).
    """
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      # Check the content of the .source_files file.
      source_files_iter = reader.source_files_iterator()
      source_file_paths = []
      prev_wall_time = 1
      for debug_event, _ in source_files_iter:
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
      # A map from ID to stack frame.
      stack_frame_by_id = collections.OrderedDict()
      stack_frames_iter = reader.stack_frames_iterator()
      prev_wall_time = 0
      for debug_event, _ in stack_frames_iter:
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
      op_name_to_op_type: An `OrderedDict` mapping op name to op type.
      op_name_to_context_id: A `dict` mapping op name to the ID of the innermost
        containing graph (context).
    """
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      graphs_iter = reader.graphs_iterator()
      prev_wall_time = 0
      op_types = []
      op_name_to_op_type = collections.OrderedDict()
      op_name_to_context_id = dict()  # Maps op name to ID of innermost context.
      context_ids = set()
      symbolic_tensor_ids = set()
      # Maps context ID to ID of directly enclosing context (`None` for
      # outermost contexts).
      context_id_to_outer_id = dict()

      for debug_event, _ in graphs_iter:
        self.assertGreaterEqual(debug_event.wall_time, prev_wall_time)
        prev_wall_time = debug_event.wall_time
        # A DebugEvent in the .graphs file contains either of the two fields:
        # - graph_op_creation for creation of a symbolic op in a graph context.
        # - debugged_graph for information regarding the graph (context).
        if debug_event.graph_op_creation.ByteSize():
          graph_op_creation = debug_event.graph_op_creation
          self.assertTrue(graph_op_creation.op_type)
          op_types.append(graph_op_creation.op_type)
          self.assertTrue(graph_op_creation.op_name)
          op_name_to_op_type[
              graph_op_creation.op_name] = graph_op_creation.op_type
          op_name_to_context_id[
              graph_op_creation.op_name] = graph_op_creation.graph_id
          self.assertTrue(graph_op_creation.graph_id)
          context_ids.add(graph_op_creation.graph_id)
          self.assertTrue(graph_op_creation.code_location)
          if graph_op_creation.num_outputs:
            self.assertLen(graph_op_creation.output_tensor_ids,
                           graph_op_creation.num_outputs)
            # Check that all symblic tensor IDs are unique.
            for tensor_id in graph_op_creation.output_tensor_ids:
              self.assertNotIn(tensor_id, symbolic_tensor_ids)
              symbolic_tensor_ids.add(tensor_id)
          for stack_frame_id in graph_op_creation.code_location.stack_frame_ids:
            self.assertIn(stack_frame_id, stack_frame_by_id)
        else:
          debugged_graph = debug_event.debugged_graph
          if debugged_graph.outer_context_id:
            inner_id = debugged_graph.graph_id
            outer_id = debugged_graph.outer_context_id
            if inner_id in context_id_to_outer_id:
              # The outer context of a context must be always the same.
              self.assertEqual(context_id_to_outer_id[inner_id], outer_id)
            else:
              context_id_to_outer_id[inner_id] = outer_id
          else:
            # This is an outermost context.
            if debugged_graph.graph_id in context_id_to_outer_id:
              self.assertIsNone(context_id_to_outer_id[debugged_graph.graph_id])
            else:
              context_id_to_outer_id[debugged_graph.graph_id] = None

      # If any graph is created, the graph context hierarchy must be populated.
      # In addition, the context of each graph op must be locatable within the
      # graph context hierarchy.
      for context_id in op_name_to_context_id.values():
        self.assertIn(context_id, context_id_to_outer_id)
      return context_ids, op_types, op_name_to_op_type, op_name_to_context_id

  def _readAndCheckExecutionFile(self, dump_root=None):
    """Read and verify the content of the .execution debug-event file.

    Args:
      dump_root: Optional argument that can be used to override the default
        dump root to read the data from.

    Returns:
      executed_op_types: Types of ops that are created, as a `list` of `str`.
      executed_graph_ids: A `list` of the same length as `executed_op_types`.
        If the executed op is a FuncGraph, the corresponding element of the
        `list` will be the ID of the FuncGraph. Else, the corresponding element
        will be an empty string. This allows establishing connection between
        eagerly executed FuncGraphs and their prior graph building.
      input_tensor_ids: Input tensor IDs for each of the ops executed, as a
        `list` of `list` of `int`s, with the same length as `executed_op_types`.
      output_tensor_ids: Output tensor IDs for each of the ops executed, as a
        `list` of `list` of `int`s, with the same length as `executed_op_types`.
      tensor_debug_modes: Tensor debug modes used to instrument each of ops
        executed.
      tensor_values: A `list` of `list` of `np.ndarray`s, representing the
        tensor values. Each item of the outer `list` corresponds to one
        execution event. Each item of the inner `list` corresponds to one
        output tensor slot of the executed op or Function.
    """
    dump_root = self.dump_root if dump_root is None else dump_root
    with debug_events_reader.DebugEventsReader(dump_root) as reader:
      execution_iter = reader.execution_iterator()
      prev_wall_time = 1
      executed_op_types = []
      executed_graph_ids = []  # Empty string for execution of inidividual ops.
      input_tensor_ids = []
      output_tensor_ids = []
      tensor_debug_modes = []
      tensor_values = []
      for debug_event, _ in execution_iter:
        self.assertGreaterEqual(debug_event.wall_time, prev_wall_time)
        prev_wall_time = debug_event.wall_time
        execution = debug_event.execution
        executed_op_types.append(execution.op_type)
        executed_graph_ids.append(execution.graph_id)
        input_tensor_ids.append(execution.input_tensor_ids)
        output_tensor_ids.append(execution.output_tensor_ids)
        tensor_debug_modes.append(execution.tensor_debug_mode)
        tensor_values.append([
            tensor_util.MakeNdarray(tensor_proto)
            for tensor_proto in execution.tensor_protos
        ])
      # TODO(cais): When tensor debug modes other than NO_TENSOR is supported,
      # return tensor_values as well.
      return (executed_op_types, executed_graph_ids, input_tensor_ids,
              output_tensor_ids, tensor_debug_modes, tensor_values)

  def _readAndCheckGraphExecutionTracesFile(self, context_ids):
    """Read & verify the content of the .graph_execution_trace debug-event file.

    Args:
      context_ids: Op-creation context IDs from _readAndCheckGraphsFile().

    Returns:
      op_names: Names of the ops that are executed, as a `list` of `str`s.
      device_names: Names of the devices that the ops belong to, respectively.
        A `list` of `str`s of the same length as `op_name`s.
      output_slots: Output slots, as a `list` of `int`s, with the same length as
        `op_names`. In other words, for an executed op with N output tensors,
        there will be N entries in this `list` and in `op_names`, at
        corresponding indices.
      tensor_values: Tensor values or their concise summaries, depending on
        TensorDebugMode.
    """
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      graph_execution_traces_iter = reader.graph_execution_traces_iterator()
      op_names = []
      device_names = []
      output_slots = []
      tensor_values = []
      for debug_event, _ in graph_execution_traces_iter:
        self.assertGreaterEqual(debug_event.wall_time, 0)
        graph_execution_trace = debug_event.graph_execution_trace
        op_names.append(graph_execution_trace.op_name)
        self.assertTrue(graph_execution_trace.device_name)
        device_names.append(graph_execution_trace.device_name)
        # All the ops in the graph have only one output.
        self.assertTrue(graph_execution_trace.tfdbg_context_id)
        self.assertIn(graph_execution_trace.tfdbg_context_id, context_ids)
        output_slots.append(graph_execution_trace.output_slot)
        dtype = dtypes.DType(graph_execution_trace.tensor_proto.dtype)
        if (dtype.is_numpy_compatible and
            dtype._type_enum != types_pb2.DT_STRING):  # pylint:disable=protected-access
          # TODO(cais): Figure out how to properly convert string tensor proto
          # to numpy representation.
          tensor_values.append(
              tensor_util.MakeNdarray(graph_execution_trace.tensor_proto))
        else:
          tensor_values.append(None)
      return op_names, device_names, output_slots, tensor_values
