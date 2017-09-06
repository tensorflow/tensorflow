# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests of the Analyzer CLI Backend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import tempfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.cli import cli_shared
from tensorflow.python.debug.cli import command_parser
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


def no_rewrite_session_config():
  rewriter_config = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=True)
  graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
  return config_pb2.ConfigProto(graph_options=graph_options)


def line_number_above():
  return tf_inspect.stack()[1][2] - 1


def parse_op_and_node(line):
  """Parse a line containing an op node followed by a node name.

  For example, if the line is
    "  [Variable] hidden/weights",
  this function will return ("Variable", "hidden/weights")

  Args:
    line: The line to be parsed, as a str.

  Returns:
    Name of the parsed op type.
    Name of the parsed node.
  """

  op_type = line.strip().split(" ")[0].replace("[", "").replace("]", "")

  # Not using [-1], to tolerate any other items that might be present behind
  # the node name.
  node_name = line.strip().split(" ")[1]

  return op_type, node_name


def assert_column_header_command_shortcut(tst,
                                          command,
                                          reverse,
                                          node_name_regex,
                                          op_type_regex,
                                          tensor_filter_name):
  tst.assertFalse(reverse and "-r" in command)
  tst.assertFalse(not(op_type_regex) and ("-t %s" % op_type_regex) in command)
  tst.assertFalse(
      not(node_name_regex) and ("-t %s" % node_name_regex) in command)
  tst.assertFalse(
      not(tensor_filter_name) and ("-t %s" % tensor_filter_name) in command)


def assert_listed_tensors(tst,
                          out,
                          expected_tensor_names,
                          expected_op_types,
                          node_name_regex=None,
                          op_type_regex=None,
                          tensor_filter_name=None,
                          sort_by="timestamp",
                          reverse=False):
  """Check RichTextLines output for list_tensors commands.

  Args:
    tst: A test_util.TensorFlowTestCase instance.
    out: The RichTextLines object to be checked.
    expected_tensor_names: (list of str) Expected tensor names in the list.
    expected_op_types: (list of str) Expected op types of the tensors, in the
      same order as the expected_tensor_names.
    node_name_regex: Optional: node name regex filter.
    op_type_regex: Optional: op type regex filter.
    tensor_filter_name: Optional: name of the tensor filter.
    sort_by: (str) (timestamp | op_type | tensor_name) the field by which the
      tensors in the list are sorted.
    reverse: (bool) whether the sorting is in reverse (i.e., descending) order.
  """

  line_iter = iter(out.lines)
  attr_segs = out.font_attr_segs
  line_counter = 0

  num_tensors = len(expected_tensor_names)

  if tensor_filter_name is None:
    tst.assertEqual("%d dumped tensor(s):" % num_tensors, next(line_iter))
  else:
    tst.assertEqual("%d dumped tensor(s) passing filter \"%s\":" %
                    (num_tensors, tensor_filter_name), next(line_iter))
  line_counter += 1

  if op_type_regex is not None:
    tst.assertEqual("Op type regex filter: \"%s\"" % op_type_regex,
                    next(line_iter))
    line_counter += 1

  if node_name_regex is not None:
    tst.assertEqual("Node name regex filter: \"%s\"" % node_name_regex,
                    next(line_iter))
    line_counter += 1

  tst.assertEqual("", next(line_iter))
  line_counter += 1

  # Verify the column heads "t (ms)", "Op type" and "Tensor name" are present.
  line = next(line_iter)
  tst.assertIn("t (ms)", line)
  tst.assertIn("Op type", line)
  tst.assertIn("Tensor name", line)

  # Verify the command shortcuts in the top row.
  attr_segs = out.font_attr_segs[line_counter]
  attr_seg = attr_segs[0]
  tst.assertEqual(0, attr_seg[0])
  tst.assertEqual(len("t (ms)"), attr_seg[1])
  command = attr_seg[2][0].content
  tst.assertIn("-s timestamp", command)
  assert_column_header_command_shortcut(
      tst, command, reverse, node_name_regex, op_type_regex,
      tensor_filter_name)
  tst.assertEqual("bold", attr_seg[2][1])

  idx0 = line.index("Size")
  attr_seg = attr_segs[1]
  tst.assertEqual(idx0, attr_seg[0])
  tst.assertEqual(idx0 + len("Size (B)"), attr_seg[1])
  command = attr_seg[2][0].content
  tst.assertIn("-s dump_size", command)
  assert_column_header_command_shortcut(tst, command, reverse, node_name_regex,
                                        op_type_regex, tensor_filter_name)
  tst.assertEqual("bold", attr_seg[2][1])

  idx0 = line.index("Op type")
  attr_seg = attr_segs[2]
  tst.assertEqual(idx0, attr_seg[0])
  tst.assertEqual(idx0 + len("Op type"), attr_seg[1])
  command = attr_seg[2][0].content
  tst.assertIn("-s op_type", command)
  assert_column_header_command_shortcut(
      tst, command, reverse, node_name_regex, op_type_regex,
      tensor_filter_name)
  tst.assertEqual("bold", attr_seg[2][1])

  idx0 = line.index("Tensor name")
  attr_seg = attr_segs[3]
  tst.assertEqual(idx0, attr_seg[0])
  tst.assertEqual(idx0 + len("Tensor name"), attr_seg[1])
  command = attr_seg[2][0].content
  tst.assertIn("-s tensor_name", command)
  assert_column_header_command_shortcut(
      tst, command, reverse, node_name_regex, op_type_regex,
      tensor_filter_name)
  tst.assertEqual("bold", attr_seg[2][1])

  # Verify the listed tensors and their timestamps.
  tensor_timestamps = []
  dump_sizes_bytes = []
  op_types = []
  tensor_names = []
  for line in line_iter:
    items = line.split(" ")
    items = [item for item in items if item]

    rel_time = float(items[0][1:-1])
    tst.assertGreaterEqual(rel_time, 0.0)

    tensor_timestamps.append(rel_time)
    dump_sizes_bytes.append(command_parser.parse_readable_size_str(items[1]))
    op_types.append(items[2])
    tensor_names.append(items[3])

  # Verify that the tensors should be listed in ascending order of their
  # timestamps.
  if sort_by == "timestamp":
    sorted_timestamps = sorted(tensor_timestamps)
    if reverse:
      sorted_timestamps.reverse()
    tst.assertEqual(sorted_timestamps, tensor_timestamps)
  elif sort_by == "dump_size":
    sorted_dump_sizes_bytes = sorted(dump_sizes_bytes)
    if reverse:
      sorted_dump_sizes_bytes.reverse()
    tst.assertEqual(sorted_dump_sizes_bytes, dump_sizes_bytes)
  elif sort_by == "op_type":
    sorted_op_types = sorted(op_types)
    if reverse:
      sorted_op_types.reverse()
    tst.assertEqual(sorted_op_types, op_types)
  elif sort_by == "tensor_name":
    sorted_tensor_names = sorted(tensor_names)
    if reverse:
      sorted_tensor_names.reverse()
    tst.assertEqual(sorted_tensor_names, tensor_names)
  else:
    tst.fail("Invalid value in sort_by: %s" % sort_by)

  # Verify that the tensors are all listed.
  for tensor_name, op_type in zip(expected_tensor_names, expected_op_types):
    tst.assertIn(tensor_name, tensor_names)
    index = tensor_names.index(tensor_name)
    tst.assertEqual(op_type, op_types[index])


def assert_node_attribute_lines(tst,
                                out,
                                node_name,
                                op_type,
                                device,
                                input_op_type_node_name_pairs,
                                ctrl_input_op_type_node_name_pairs,
                                recipient_op_type_node_name_pairs,
                                ctrl_recipient_op_type_node_name_pairs,
                                attr_key_val_pairs=None,
                                num_dumped_tensors=None,
                                show_stack_trace=False,
                                stack_trace_available=False):
  """Check RichTextLines output for node_info commands.

  Args:
    tst: A test_util.TensorFlowTestCase instance.
    out: The RichTextLines object to be checked.
    node_name: Name of the node.
    op_type: Op type of the node, as a str.
    device: Name of the device on which the node resides.
    input_op_type_node_name_pairs: A list of 2-tuples of op type and node name,
      for the (non-control) inputs to the node.
    ctrl_input_op_type_node_name_pairs: A list of 2-tuples of op type and node
      name, for the control inputs to the node.
    recipient_op_type_node_name_pairs: A list of 2-tuples of op type and node
      name, for the (non-control) output recipients to the node.
    ctrl_recipient_op_type_node_name_pairs: A list of 2-tuples of op type and
      node name, for the control output recipients to the node.
    attr_key_val_pairs: Optional: attribute key-value pairs of the node, as a
      list of 2-tuples.
    num_dumped_tensors: Optional: number of tensor dumps from the node.
    show_stack_trace: (bool) whether the stack trace of the node's
      construction is asserted to be present.
    stack_trace_available: (bool) whether Python stack trace is available.
  """

  line_iter = iter(out.lines)

  tst.assertEqual("Node %s" % node_name, next(line_iter))
  tst.assertEqual("", next(line_iter))
  tst.assertEqual("  Op: %s" % op_type, next(line_iter))
  tst.assertEqual("  Device: %s" % device, next(line_iter))
  tst.assertEqual("", next(line_iter))
  tst.assertEqual("  %d input(s) + %d control input(s):" %
                  (len(input_op_type_node_name_pairs),
                   len(ctrl_input_op_type_node_name_pairs)), next(line_iter))

  # Check inputs.
  tst.assertEqual("    %d input(s):" % len(input_op_type_node_name_pairs),
                  next(line_iter))
  for op_type, node_name in input_op_type_node_name_pairs:
    tst.assertEqual("      [%s] %s" % (op_type, node_name), next(line_iter))

  tst.assertEqual("", next(line_iter))

  # Check control inputs.
  if ctrl_input_op_type_node_name_pairs:
    tst.assertEqual("    %d control input(s):" %
                    len(ctrl_input_op_type_node_name_pairs), next(line_iter))
    for op_type, node_name in ctrl_input_op_type_node_name_pairs:
      tst.assertEqual("      [%s] %s" % (op_type, node_name), next(line_iter))

    tst.assertEqual("", next(line_iter))

  tst.assertEqual("  %d recipient(s) + %d control recipient(s):" %
                  (len(recipient_op_type_node_name_pairs),
                   len(ctrl_recipient_op_type_node_name_pairs)),
                  next(line_iter))

  # Check recipients, the order of which is not deterministic.
  tst.assertEqual("    %d recipient(s):" %
                  len(recipient_op_type_node_name_pairs), next(line_iter))

  t_recs = []
  for _ in recipient_op_type_node_name_pairs:
    line = next(line_iter)

    op_type, node_name = parse_op_and_node(line)
    t_recs.append((op_type, node_name))

  tst.assertItemsEqual(recipient_op_type_node_name_pairs, t_recs)

  # Check control recipients, the order of which is not deterministic.
  if ctrl_recipient_op_type_node_name_pairs:
    tst.assertEqual("", next(line_iter))

    tst.assertEqual("    %d control recipient(s):" %
                    len(ctrl_recipient_op_type_node_name_pairs),
                    next(line_iter))

    t_ctrl_recs = []
    for _ in ctrl_recipient_op_type_node_name_pairs:
      line = next(line_iter)

      op_type, node_name = parse_op_and_node(line)
      t_ctrl_recs.append((op_type, node_name))

    tst.assertItemsEqual(ctrl_recipient_op_type_node_name_pairs, t_ctrl_recs)

  # The order of multiple attributes can be non-deterministic.
  if attr_key_val_pairs:
    tst.assertEqual("", next(line_iter))

    tst.assertEqual("Node attributes:", next(line_iter))

    kv_pairs = []
    for key, val in attr_key_val_pairs:
      key = next(line_iter).strip().replace(":", "")

      val = next(line_iter).strip()

      kv_pairs.append((key, val))

      tst.assertEqual("", next(line_iter))

    tst.assertItemsEqual(attr_key_val_pairs, kv_pairs)

  if num_dumped_tensors is not None:
    tst.assertEqual("%d dumped tensor(s):" % num_dumped_tensors,
                    next(line_iter))
    tst.assertEqual("", next(line_iter))

    dump_timestamps_ms = []
    for _ in xrange(num_dumped_tensors):
      line = next(line_iter)

      tst.assertStartsWith(line.strip(), "Slot 0 @ DebugIdentity @")
      tst.assertTrue(line.strip().endswith(" ms"))

      dump_timestamp_ms = float(line.strip().split(" @ ")[-1].replace("ms", ""))
      tst.assertGreaterEqual(dump_timestamp_ms, 0.0)

      dump_timestamps_ms.append(dump_timestamp_ms)

    tst.assertEqual(sorted(dump_timestamps_ms), dump_timestamps_ms)

  if show_stack_trace:
    tst.assertEqual("", next(line_iter))
    tst.assertEqual("", next(line_iter))
    tst.assertEqual("Traceback of node construction:", next(line_iter))
    if stack_trace_available:
      try:
        depth_counter = 0
        while True:
          for i in range(5):
            line = next(line_iter)
            if i == 0:
              tst.assertEqual(depth_counter, int(line.split(":")[0]))
            elif i == 1:
              tst.assertStartsWith(line, "  Line:")
            elif i == 2:
              tst.assertStartsWith(line, "  Function:")
            elif i == 3:
              tst.assertStartsWith(line, "  Text:")
            elif i == 4:
              tst.assertEqual("", line)

          depth_counter += 1
      except StopIteration:
        tst.assertEqual(0, i)
    else:
      tst.assertEqual("(Unavailable because no Python graph has been loaded)",
                      next(line_iter))


def check_syntax_error_output(tst, out, command_prefix):
  """Check RichTextLines output for valid command prefix but invalid syntax."""

  tst.assertEqual([
      "Syntax error for command: %s" % command_prefix,
      "For help, do \"help %s\"" % command_prefix
  ], out.lines)


def check_error_output(tst, out, command_prefix, args):
  """Check RichTextLines output from invalid/erroneous commands.

  Args:
    tst: A test_util.TensorFlowTestCase instance.
    out: The RichTextLines object to be checked.
    command_prefix: The command prefix of the command that caused the error.
    args: The arguments (excluding prefix) of the command that caused the error.
  """

  tst.assertGreater(len(out.lines), 2)
  tst.assertStartsWith(out.lines[0],
                       "Error occurred during handling of command: %s %s" %
                       (command_prefix, " ".join(args)))


def check_main_menu(tst,
                    out,
                    list_tensors_enabled=False,
                    node_info_node_name=None,
                    print_tensor_node_name=None,
                    list_inputs_node_name=None,
                    list_outputs_node_name=None):
  """Check the main menu annotation of an output."""

  tst.assertIn(debugger_cli_common.MAIN_MENU_KEY, out.annotations)

  menu = out.annotations[debugger_cli_common.MAIN_MENU_KEY]
  tst.assertEqual(list_tensors_enabled,
                  menu.caption_to_item("list_tensors").is_enabled())

  menu_item = menu.caption_to_item("node_info")
  if node_info_node_name:
    tst.assertTrue(menu_item.is_enabled())
    tst.assertTrue(menu_item.content.endswith(node_info_node_name))
  else:
    tst.assertFalse(menu_item.is_enabled())

  menu_item = menu.caption_to_item("print_tensor")
  if print_tensor_node_name:
    tst.assertTrue(menu_item.is_enabled())
    tst.assertTrue(menu_item.content.endswith(print_tensor_node_name))
  else:
    tst.assertFalse(menu_item.is_enabled())

  menu_item = menu.caption_to_item("list_inputs")
  if list_inputs_node_name:
    tst.assertTrue(menu_item.is_enabled())
    tst.assertTrue(menu_item.content.endswith(list_inputs_node_name))
  else:
    tst.assertFalse(menu_item.is_enabled())

  menu_item = menu.caption_to_item("list_outputs")
  if list_outputs_node_name:
    tst.assertTrue(menu_item.is_enabled())
    tst.assertTrue(menu_item.content.endswith(list_outputs_node_name))
  else:
    tst.assertFalse(menu_item.is_enabled())

  tst.assertTrue(menu.caption_to_item("run_info").is_enabled())
  tst.assertTrue(menu.caption_to_item("help").is_enabled())


def check_menu_item(tst, out, line_index, expected_begin, expected_end,
                    expected_command):
  attr_segs = out.font_attr_segs[line_index]
  found_menu_item = False
  for begin, end, attribute in attr_segs:
    attributes = [attribute] if not isinstance(attribute, list) else attribute
    menu_item = [attribute for attribute in attributes if
                 isinstance(attribute, debugger_cli_common.MenuItem)]
    if menu_item:
      tst.assertEqual(expected_begin, begin)
      tst.assertEqual(expected_end, end)
      tst.assertEqual(expected_command, menu_item[0].content)
      found_menu_item = True
      break
  tst.assertTrue(found_menu_item)


class AnalyzerCLISimpleMulAddTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    cls._dump_root = tempfile.mkdtemp()

    cls._is_gpu_available = test.is_gpu_available()
    if cls._is_gpu_available:
      gpu_name = test_util.gpu_device_name()
      cls._main_device = "/job:localhost/replica:0/task:0" + gpu_name
    else:
      cls._main_device = "/job:localhost/replica:0/task:0/cpu:0"

    cls._curr_file_path = os.path.abspath(
        tf_inspect.getfile(tf_inspect.currentframe()))

    cls._sess = session.Session(config=no_rewrite_session_config())
    with cls._sess as sess:
      u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      v_init_val = np.array([[2.0], [-1.0]])

      u_name = "simple_mul_add/u"
      v_name = "simple_mul_add/v"

      u_init = constant_op.constant(u_init_val, shape=[2, 2], name="u_init")
      u = variables.Variable(u_init, name=u_name)
      cls._u_line_number = line_number_above()

      v_init = constant_op.constant(v_init_val, shape=[2, 1], name="v_init")
      v = variables.Variable(v_init, name=v_name)
      cls._v_line_number = line_number_above()

      w = math_ops.matmul(u, v, name="simple_mul_add/matmul")
      cls._w_line_number = line_number_above()

      x = math_ops.add(w, w, name="simple_mul_add/add")
      cls._x_line_number = line_number_above()

      u.initializer.run()
      v.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls="file://%s" % cls._dump_root)

      # Invoke Session.run().
      run_metadata = config_pb2.RunMetadata()
      sess.run(x, options=run_options, run_metadata=run_metadata)

    cls._debug_dump = debug_data.DebugDumpDir(
        cls._dump_root, partition_graphs=run_metadata.partition_graphs)

    # Construct the analyzer.
    cls._analyzer = analyzer_cli.DebugAnalyzer(cls._debug_dump)

    # Construct the handler registry.
    cls._registry = debugger_cli_common.CommandHandlerRegistry()

    # Register command handlers.
    cls._registry.register_command_handler(
        "list_tensors",
        cls._analyzer.list_tensors,
        cls._analyzer.get_help("list_tensors"),
        prefix_aliases=["lt"])
    cls._registry.register_command_handler(
        "node_info",
        cls._analyzer.node_info,
        cls._analyzer.get_help("node_info"),
        prefix_aliases=["ni"])
    cls._registry.register_command_handler(
        "print_tensor",
        cls._analyzer.print_tensor,
        cls._analyzer.get_help("print_tensor"),
        prefix_aliases=["pt"])
    cls._registry.register_command_handler(
        "print_source",
        cls._analyzer.print_source,
        cls._analyzer.get_help("print_source"),
        prefix_aliases=["ps"])
    cls._registry.register_command_handler(
        "list_source",
        cls._analyzer.list_source,
        cls._analyzer.get_help("list_source"),
        prefix_aliases=["ls"])
    cls._registry.register_command_handler(
        "eval",
        cls._analyzer.evaluate_expression,
        cls._analyzer.get_help("eval"),
        prefix_aliases=["ev"])

  @classmethod
  def tearDownClass(cls):
    # Tear down temporary dump directory.
    shutil.rmtree(cls._dump_root)

  def testMeasureTensorListColumnWidthsGivesRightAnswerForEmptyData(self):
    timestamp_col_width, dump_size_col_width, op_type_col_width = (
        self._analyzer._measure_tensor_list_column_widths([]))
    self.assertEqual(len("t (ms)") + 1, timestamp_col_width)
    self.assertEqual(len("Size (B)") + 1, dump_size_col_width)
    self.assertEqual(len("Op type") + 1, op_type_col_width)

  def testMeasureTensorListColumnWidthsGivesRightAnswerForData(self):
    dump = self._debug_dump.dumped_tensor_data[0]
    self.assertLess(dump.dump_size_bytes, 1000)
    self.assertEqual(
        "VariableV2", self._debug_dump.node_op_type(dump.node_name))
    _, dump_size_col_width, op_type_col_width = (
        self._analyzer._measure_tensor_list_column_widths([dump]))
    # The length of str(dump.dump_size_bytes) is less than the length of
    # "Size (B)" (8). So the column width should be determined by the length of
    # "Size (B)".
    self.assertEqual(len("Size (B)") + 1, dump_size_col_width)
    # The length of "VariableV2" is greater than the length of "Op type". So the
    # column should be determined by the length of "VariableV2".
    self.assertEqual(len("VariableV2") + 1, op_type_col_width)

  def testListTensors(self):
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", [])

    assert_listed_tensors(self, out, [
        "simple_mul_add/u:0", "simple_mul_add/v:0", "simple_mul_add/u/read:0",
        "simple_mul_add/v/read:0", "simple_mul_add/matmul:0",
        "simple_mul_add/add:0"
    ], ["VariableV2", "VariableV2", "Identity", "Identity", "MatMul", "Add"])

    # Check the main menu.
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInReverseTimeOrderWorks(self):
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-s", "timestamp", "-r"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ],
        ["VariableV2", "VariableV2", "Identity", "Identity", "MatMul", "Add"],
        sort_by="timestamp",
        reverse=True)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInDumpSizeOrderWorks(self):
    out = self._registry.dispatch_command("lt", ["-s", "dump_size"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ],
        ["VariableV2", "VariableV2", "Identity", "Identity", "MatMul", "Add"],
        sort_by="dump_size")
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInReverseDumpSizeOrderWorks(self):
    out = self._registry.dispatch_command("lt", ["-s", "dump_size", "-r"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ],
        ["VariableV2", "VariableV2", "Identity", "Identity", "MatMul", "Add"],
        sort_by="dump_size",
        reverse=True)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsWithInvalidSortByFieldGivesError(self):
    out = self._registry.dispatch_command("lt", ["-s", "foobar"])
    self.assertIn("ValueError: Unsupported key to sort tensors by: foobar",
                  out.lines)

  def testListTensorsInOpTypeOrderWorks(self):
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-s", "op_type"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ],
        ["VariableV2", "VariableV2", "Identity", "Identity", "MatMul", "Add"],
        sort_by="op_type",
        reverse=False)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInReverseOpTypeOrderWorks(self):
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-s", "op_type", "-r"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ],
        ["VariableV2", "VariableV2", "Identity", "Identity", "MatMul", "Add"],
        sort_by="op_type",
        reverse=True)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInTensorNameOrderWorks(self):
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-s", "tensor_name"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ],
        ["VariableV2", "VariableV2", "Identity", "Identity", "MatMul", "Add"],
        sort_by="tensor_name",
        reverse=False)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInReverseTensorNameOrderWorks(self):
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-s", "tensor_name", "-r"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u:0", "simple_mul_add/v:0",
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0",
            "simple_mul_add/matmul:0", "simple_mul_add/add:0"
        ],
        ["VariableV2", "VariableV2", "Identity", "Identity", "MatMul", "Add"],
        sort_by="tensor_name",
        reverse=True)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsFilterByNodeNameRegex(self):
    out = self._registry.dispatch_command("list_tensors",
                                          ["--node_name_filter", ".*read.*"])
    assert_listed_tensors(
        self,
        out, ["simple_mul_add/u/read:0", "simple_mul_add/v/read:0"],
        ["Identity", "Identity"],
        node_name_regex=".*read.*")

    out = self._registry.dispatch_command("list_tensors", ["-n", "^read"])
    assert_listed_tensors(self, out, [], [], node_name_regex="^read")
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorFilterByOpTypeRegex(self):
    out = self._registry.dispatch_command("list_tensors",
                                          ["--op_type_filter", "Identity"])
    assert_listed_tensors(
        self,
        out, ["simple_mul_add/u/read:0", "simple_mul_add/v/read:0"],
        ["Identity", "Identity"],
        op_type_regex="Identity")

    out = self._registry.dispatch_command("list_tensors",
                                          ["-t", "(Add|MatMul)"])
    assert_listed_tensors(
        self,
        out, ["simple_mul_add/add:0", "simple_mul_add/matmul:0"],
        ["Add", "MatMul"],
        op_type_regex="(Add|MatMul)")
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorFilterByNodeNameRegexAndOpTypeRegex(self):
    out = self._registry.dispatch_command(
        "list_tensors", ["-t", "(Add|MatMul)", "-n", ".*add$"])
    assert_listed_tensors(
        self,
        out, ["simple_mul_add/add:0"], ["Add"],
        node_name_regex=".*add$",
        op_type_regex="(Add|MatMul)")
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsFilterNanOrInf(self):
    """Test register and invoke a tensor filter."""

    # First, register the filter.
    self._analyzer.add_tensor_filter("has_inf_or_nan",
                                     debug_data.has_inf_or_nan)

    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-f", "has_inf_or_nan"])

    # This TF graph run did not generate any bad numerical values.
    assert_listed_tensors(
        self, out, [], [], tensor_filter_name="has_inf_or_nan")
    # TODO(cais): A test with some actual bad numerical values.

    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorNonexistentFilter(self):
    """Test attempt to use a nonexistent tensor filter."""

    out = self._registry.dispatch_command("lt", ["-f", "foo_filter"])

    self.assertEqual(["ERROR: There is no tensor filter named \"foo_filter\"."],
                     out.lines)
    check_main_menu(self, out, list_tensors_enabled=False)

  def testListTensorsInvalidOptions(self):
    out = self._registry.dispatch_command("list_tensors", ["--bar"])
    check_syntax_error_output(self, out, "list_tensors")

  def testNodeInfoByNodeName(self):
    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("node_info", [node_name])

    recipients = [("Add", "simple_mul_add/add"), ("Add", "simple_mul_add/add")]

    assert_node_attribute_lines(self, out, node_name, "MatMul",
                                self._main_device,
                                [("Identity", "simple_mul_add/u/read"),
                                 ("Identity", "simple_mul_add/v/read")], [],
                                recipients, [])
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

    # Verify that the node name is bold in the first line.
    self.assertEqual(
        [(len(out.lines[0]) - len(node_name), len(out.lines[0]), "bold")],
        out.font_attr_segs[0])

  def testNodeInfoShowAttributes(self):
    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("node_info", ["-a", node_name])

    assert_node_attribute_lines(
        self,
        out,
        node_name,
        "MatMul",
        self._main_device, [("Identity", "simple_mul_add/u/read"),
                            ("Identity", "simple_mul_add/v/read")], [],
        [("Add", "simple_mul_add/add"), ("Add", "simple_mul_add/add")], [],
        attr_key_val_pairs=[("transpose_a", "b: false"),
                            ("transpose_b", "b: false"),
                            ("T", "type: DT_DOUBLE")])
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

  def testNodeInfoShowDumps(self):
    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("node_info", ["-d", node_name])

    assert_node_attribute_lines(
        self,
        out,
        node_name,
        "MatMul",
        self._main_device, [("Identity", "simple_mul_add/u/read"),
                            ("Identity", "simple_mul_add/v/read")], [],
        [("Add", "simple_mul_add/add"), ("Add", "simple_mul_add/add")], [],
        num_dumped_tensors=1)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)
    check_menu_item(self, out, 16,
                    len(out.lines[16]) - len(out.lines[16].strip()),
                    len(out.lines[16]), "pt %s:0 -n 0" % node_name)

  def testNodeInfoShowStackTraceUnavailableIsIndicated(self):
    self._debug_dump.set_python_graph(None)

    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("node_info", ["-t", node_name])

    assert_node_attribute_lines(
        self,
        out,
        node_name,
        "MatMul",
        self._main_device, [("Identity", "simple_mul_add/u/read"),
                            ("Identity", "simple_mul_add/v/read")], [],
        [("Add", "simple_mul_add/add"), ("Add", "simple_mul_add/add")], [],
        show_stack_trace=True, stack_trace_available=False)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

  def testNodeInfoShowStackTraceAvailableWorks(self):
    self._debug_dump.set_python_graph(self._sess.graph)

    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("node_info", ["-t", node_name])

    assert_node_attribute_lines(
        self,
        out,
        node_name,
        "MatMul",
        self._main_device, [("Identity", "simple_mul_add/u/read"),
                            ("Identity", "simple_mul_add/v/read")], [],
        [("Add", "simple_mul_add/add"), ("Add", "simple_mul_add/add")], [],
        show_stack_trace=True, stack_trace_available=True)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

  def testNodeInfoByTensorName(self):
    node_name = "simple_mul_add/u/read"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command("node_info", [tensor_name])

    assert_node_attribute_lines(self, out, node_name, "Identity",
                                self._main_device,
                                [("VariableV2", "simple_mul_add/u")], [],
                                [("MatMul", "simple_mul_add/matmul")], [])
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        list_inputs_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

  def testNodeInfoNonexistentNodeName(self):
    out = self._registry.dispatch_command("node_info", ["bar"])
    self.assertEqual(
        ["ERROR: There is no node named \"bar\" in the partition graphs"],
        out.lines)
    # Check color indicating error.
    self.assertEqual({0: [(0, 59, cli_shared.COLOR_RED)]}, out.font_attr_segs)
    check_main_menu(self, out, list_tensors_enabled=True)

  def testPrintTensor(self):
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name], screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\":" % tensor_name,
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    self.assertIn(5, out.annotations)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorHighlightingRanges(self):
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "--ranges", "[-inf, 0.0]"],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\": " % tensor_name +
        "Highlighted([-inf, 0.0]): 1 of 2 element(s) (50.00%)",
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    self.assertIn(5, out.annotations)
    self.assertEqual([(8, 11, "bold")], out.font_attr_segs[5])

    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "--ranges", "[[-inf, -5.5], [5.5, inf]]"],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\": " % tensor_name +
        "Highlighted([[-inf, -5.5], [5.5, inf]]): "
        "1 of 2 element(s) (50.00%)",
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    self.assertIn(5, out.annotations)
    self.assertEqual([(9, 11, "bold")], out.font_attr_segs[4])
    self.assertNotIn(5, out.font_attr_segs)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorHighlightingRangesAndIncludingNumericSummary(self):
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "--ranges", "[-inf, 0.0]", "-s"],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\": " % tensor_name +
        "Highlighted([-inf, 0.0]): 1 of 2 element(s) (50.00%)",
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "Numeric summary:",
        "| - + | total |",
        "| 1 1 |     2 |",
        "|  min  max mean  std |",
        "| -2.0  7.0  2.5  4.5 |",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(10, out.annotations)
    self.assertIn(11, out.annotations)
    self.assertEqual([(8, 11, "bold")], out.font_attr_segs[11])

  def testPrintTensorWithSlicing(self):
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name + "[1, :]"], screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity[1, :]\":" % tensor_name, "  dtype: float64",
        "  shape: (1,)", "", "array([-2.])"
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorInvalidSlicingString(self):
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name + "[1, foo()]"], screen_info={"cols": 80})

    self.assertEqual("Error occurred during handling of command: print_tensor "
                     + tensor_name + "[1, foo()]:", out.lines[0])
    self.assertEqual("ValueError: Invalid tensor-slicing string.",
                     out.lines[-2])

  def testPrintTensorValidExplicitNumber(self):
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "-n", "0"], screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"%s:DebugIdentity\":" % tensor_name,
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    self.assertIn(5, out.annotations)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorInvalidExplicitNumber(self):
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "print_tensor", [tensor_name, "-n", "1"], screen_info={"cols": 80})

    self.assertEqual([
        "ERROR: Invalid number (1) for tensor simple_mul_add/matmul:0, "
        "which generated one dump."
    ], out.lines)

    self.assertNotIn("tensor_metadata", out.annotations)

    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorMissingOutputSlotLeadsToOnlyDumpedTensorPrinted(self):
    node_name = "simple_mul_add/matmul"
    out = self._registry.dispatch_command("print_tensor", [node_name])

    self.assertEqual([
        "Tensor \"%s:0:DebugIdentity\":" % node_name, "  dtype: float64",
        "  shape: (2, 1)", "", "array([[ 7.],", "       [-2.]])"
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        list_inputs_node_name=node_name,
        list_outputs_node_name=node_name)

  def testPrintTensorNonexistentNodeName(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["simple_mul_add/matmul/foo:0"])

    self.assertEqual([
        "ERROR: Node \"simple_mul_add/matmul/foo\" does not exist in partition "
        "graphs"
    ], out.lines)
    check_main_menu(self, out, list_tensors_enabled=True)

  def testEvalExpression(self):
    node_name = "simple_mul_add/matmul"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command(
        "eval", ["np.matmul(`%s`, `%s`.T)" % (tensor_name, tensor_name)],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"from eval of expression "
        "'np.matmul(`simple_mul_add/matmul:0`, "
        "`simple_mul_add/matmul:0`.T)'\":",
        "  dtype: float64",
        "  shape: (2, 2)",
        "",
        "Numeric summary:",
        "| - + | total |",
        "| 2 2 |     4 |",
        "|           min           max          mean           std |",
        "|         -14.0          49.0          6.25 25.7524270701 |",
        "",
        "array([[ 49., -14.],",
        "       [-14.,   4.]])"], out.lines)

  def testAddGetTensorFilterLambda(self):
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump)
    analyzer.add_tensor_filter("foo_filter", lambda x, y: True)
    self.assertTrue(analyzer.get_tensor_filter("foo_filter")(None, None))

  def testAddGetTensorFilterNestedFunction(self):
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump)

    def foo_filter(unused_arg_0, unused_arg_1):
      return True

    analyzer.add_tensor_filter("foo_filter", foo_filter)
    self.assertTrue(analyzer.get_tensor_filter("foo_filter")(None, None))

  def testAddTensorFilterEmptyName(self):
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump)

    with self.assertRaisesRegexp(ValueError,
                                 "Input argument filter_name cannot be empty."):
      analyzer.add_tensor_filter("", lambda datum, tensor: True)

  def testAddTensorFilterNonStrName(self):
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump)

    with self.assertRaisesRegexp(
        TypeError,
        "Input argument filter_name is expected to be str, ""but is not"):
      analyzer.add_tensor_filter(1, lambda datum, tensor: True)

  def testAddGetTensorFilterNonCallable(self):
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump)

    with self.assertRaisesRegexp(
        TypeError, "Input argument filter_callable is expected to be callable, "
        "but is not."):
      analyzer.add_tensor_filter("foo_filter", "bar")

  def testGetNonexistentTensorFilter(self):
    analyzer = analyzer_cli.DebugAnalyzer(self._debug_dump)

    analyzer.add_tensor_filter("foo_filter", lambda datum, tensor: True)
    with self.assertRaisesRegexp(ValueError,
                                 "There is no tensor filter named \"bar\""):
      analyzer.get_tensor_filter("bar")

  def _findSourceLine(self, annotated_source, line_number):
    """Find line of given line number in annotated source.

    Args:
      annotated_source: (debugger_cli_common.RichTextLines) the annotated source
      line_number: (int) 1-based line number

    Returns:
      (int) If line_number is found, 0-based line index in
        annotated_source.lines. Otherwise, None.
    """

    index = None
    for i, line in enumerate(annotated_source.lines):
      if line.startswith("L%d " % line_number):
        index = i
        break
    return index

  def testPrintSourceForOpNamesWholeFileWorks(self):
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command(
        "print_source", [self._curr_file_path], screen_info={"cols": 80})

    # Verify the annotation of the line that creates u.
    index = self._findSourceLine(out, self._u_line_number)
    self.assertEqual(
        ["L%d         u = variables.Variable(u_init, name=u_name)" %
         self._u_line_number,
         "    simple_mul_add/u",
         "    simple_mul_add/u/Assign",
         "    simple_mul_add/u/read"],
        out.lines[index : index + 4])
    self.assertEqual("pt simple_mul_add/u",
                     out.font_attr_segs[index + 1][0][2].content)
    # simple_mul_add/u/Assign is not used in this run because the Variable has
    # already been initialized.
    self.assertEqual(cli_shared.COLOR_BLUE, out.font_attr_segs[index + 2][0][2])
    self.assertEqual("pt simple_mul_add/u/read",
                     out.font_attr_segs[index + 3][0][2].content)

    # Verify the annotation of the line that creates v.
    index = self._findSourceLine(out, self._v_line_number)
    self.assertEqual(
        ["L%d         v = variables.Variable(v_init, name=v_name)" %
         self._v_line_number,
         "    simple_mul_add/v"],
        out.lines[index : index + 2])
    self.assertEqual("pt simple_mul_add/v",
                     out.font_attr_segs[index + 1][0][2].content)

    # Verify the annotation of the line that creates w.
    index = self._findSourceLine(out, self._w_line_number)
    self.assertEqual(
        ["L%d         " % self._w_line_number +
         "w = math_ops.matmul(u, v, name=\"simple_mul_add/matmul\")",
         "    simple_mul_add/matmul"],
        out.lines[index : index + 2])
    self.assertEqual("pt simple_mul_add/matmul",
                     out.font_attr_segs[index + 1][0][2].content)

    # Verify the annotation of the line that creates x.
    index = self._findSourceLine(out, self._x_line_number)
    self.assertEqual(
        ["L%d         " % self._x_line_number +
         "x = math_ops.add(w, w, name=\"simple_mul_add/add\")",
         "    simple_mul_add/add"],
        out.lines[index : index + 2])
    self.assertEqual("pt simple_mul_add/add",
                     out.font_attr_segs[index + 1][0][2].content)

  def testPrintSourceForTensorNamesWholeFileWorks(self):
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command(
        "print_source",
        [self._curr_file_path, "--tensors"],
        screen_info={"cols": 80})

    # Verify the annotation of the line that creates u.
    index = self._findSourceLine(out, self._u_line_number)
    self.assertEqual(
        ["L%d         u = variables.Variable(u_init, name=u_name)" %
         self._u_line_number,
         "    simple_mul_add/u/read:0",
         "    simple_mul_add/u:0"],
        out.lines[index : index + 3])
    self.assertEqual("pt simple_mul_add/u/read:0",
                     out.font_attr_segs[index + 1][0][2].content)
    self.assertEqual("pt simple_mul_add/u:0",
                     out.font_attr_segs[index + 2][0][2].content)

  def testPrintSourceForOpNamesStartingAtSpecifiedLineWorks(self):
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command(
        "print_source",
        [self._curr_file_path, "-b", "3"],
        screen_info={"cols": 80})

    self.assertEqual(
        2, out.annotations[debugger_cli_common.INIT_SCROLL_POS_KEY])

    index = self._findSourceLine(out, self._u_line_number)
    self.assertEqual(
        ["L%d         u = variables.Variable(u_init, name=u_name)" %
         self._u_line_number,
         "    simple_mul_add/u",
         "    simple_mul_add/u/Assign",
         "    simple_mul_add/u/read"],
        out.lines[index : index + 4])
    self.assertEqual("pt simple_mul_add/u",
                     out.font_attr_segs[index + 1][0][2].content)
    # simple_mul_add/u/Assign is not used in this run because the Variable has
    # already been initialized.
    self.assertEqual(cli_shared.COLOR_BLUE, out.font_attr_segs[index + 2][0][2])
    self.assertEqual("pt simple_mul_add/u/read",
                     out.font_attr_segs[index + 3][0][2].content)

  def testPrintSourceForOpNameSettingMaximumElementCountWorks(self):
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command(
        "print_source",
        [self._curr_file_path, "-m", "1"],
        screen_info={"cols": 80})

    index = self._findSourceLine(out, self._u_line_number)
    self.assertEqual(
        ["L%d         u = variables.Variable(u_init, name=u_name)" %
         self._u_line_number,
         "    simple_mul_add/u",
         "    (... Omitted 2 of 3 op(s) ...) +5"],
        out.lines[index : index + 3])
    self.assertEqual("pt simple_mul_add/u",
                     out.font_attr_segs[index + 1][0][2].content)
    more_elements_command = out.font_attr_segs[index + 2][-1][2].content
    self.assertStartsWith(more_elements_command,
                          "ps %s " % self._curr_file_path)
    self.assertIn(" -m 6", more_elements_command)

  def testListSourceWorks(self):
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command("list_source", [])

    non_tf_lib_files_start = [
        i for i in xrange(len(out.lines))
        if out.lines[i].startswith("Source file path")][0] + 1
    non_tf_lib_files_end = [
        i for i in xrange(len(out.lines))
        if out.lines[i].startswith("TensorFlow Python library file(s):")][0] - 1
    non_tf_lib_files = [
        line.split(" ")[0] for line
        in out.lines[non_tf_lib_files_start : non_tf_lib_files_end]]
    self.assertIn(self._curr_file_path, non_tf_lib_files)

    # Check that the TF library files are marked with special color attribute.
    for i in xrange(non_tf_lib_files_end + 1, len(out.lines)):
      if not out.lines[i]:
        continue
      for attr_seg in  out.font_attr_segs[i]:
        self.assertTrue(cli_shared.COLOR_GRAY in attr_seg[2] or
                        attr_seg[2] == cli_shared.COLOR_GRAY)

  def testListSourceWithNodeNameFilterWithMatchesWorks(self):
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command("list_source", ["-n", ".*/read"])

    self.assertStartsWith(out.lines[1], "Node name regex filter: \".*/read\"")

    non_tf_lib_files_start = [
        i for i in xrange(len(out.lines))
        if out.lines[i].startswith("Source file path")][0] + 1
    non_tf_lib_files_end = [
        i for i in xrange(len(out.lines))
        if out.lines[i].startswith("TensorFlow Python library file(s):")][0] - 1
    non_tf_lib_files = [
        line.split(" ")[0] for line
        in out.lines[non_tf_lib_files_start : non_tf_lib_files_end]]
    self.assertIn(self._curr_file_path, non_tf_lib_files)

    # Check that the TF library files are marked with special color attribute.
    for i in xrange(non_tf_lib_files_end + 1, len(out.lines)):
      if not out.lines[i]:
        continue
      for attr_seg in  out.font_attr_segs[i]:
        self.assertTrue(cli_shared.COLOR_GRAY in attr_seg[2] or
                        attr_seg[2] == cli_shared.COLOR_GRAY)

  def testListSourceWithNodeNameFilterWithNoMatchesWorks(self):
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command("list_source", ["-n", "^$"])

    self.assertEqual([
        "List of source files that created nodes in this run",
        "Node name regex filter: \"^$\"", "",
        "[No source file information.]"], out.lines)

  def testListSourceWithPathAndNodeNameFiltersWorks(self):
    self._debug_dump.set_python_graph(self._sess.graph)
    out = self._registry.dispatch_command(
        "list_source", ["-p", self._curr_file_path, "-n", ".*read"])

    self.assertEqual([
        "List of source files that created nodes in this run",
        "File path regex filter: \"%s\"" % self._curr_file_path,
        "Node name regex filter: \".*read\"", ""], out.lines[:4])

  def testListSourceWithCompiledPythonSourceWorks(self):
    def fake_list_source_files_against_dump(dump,
                                            path_regex_whitelist=None,
                                            node_name_regex_whitelist=None):
      del dump, path_regex_whitelist, node_name_regex_whitelist
      return [("compiled_1.pyc", False, 10, 20, 30, 4),
              ("compiled_2.pyo", False, 10, 20, 30, 5),
              ("uncompiled.py", False, 10, 20, 30, 6)]

    with test.mock.patch.object(
        source_utils, "list_source_files_against_dump",
        side_effect=fake_list_source_files_against_dump):
      out = self._registry.dispatch_command("list_source", [])

      self.assertStartsWith(out.lines[4], "compiled_1.pyc")
      self.assertEqual((0, 14, [cli_shared.COLOR_WHITE]),
                       out.font_attr_segs[4][0])
      self.assertStartsWith(out.lines[5], "compiled_2.pyo")
      self.assertEqual((0, 14, [cli_shared.COLOR_WHITE]),
                       out.font_attr_segs[5][0])
      self.assertStartsWith(out.lines[6], "uncompiled.py")
      self.assertEqual(0, out.font_attr_segs[6][0][0])
      self.assertEqual(13, out.font_attr_segs[6][0][1])
      self.assertEqual(cli_shared.COLOR_WHITE, out.font_attr_segs[6][0][2][0])
      self.assertEqual("ps uncompiled.py -b 6",
                       out.font_attr_segs[6][0][2][1].content)


class AnalyzerCLIPrintLargeTensorTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    cls._dump_root = tempfile.mkdtemp()

    with session.Session(config=no_rewrite_session_config()) as sess:
      # 2400 elements should exceed the default threshold (2000).
      x = constant_op.constant(np.zeros([300, 8]), name="large_tensors/x")

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls="file://%s" % cls._dump_root)

      # Invoke Session.run().
      run_metadata = config_pb2.RunMetadata()
      sess.run(x, options=run_options, run_metadata=run_metadata)

    cls._debug_dump = debug_data.DebugDumpDir(
        cls._dump_root, partition_graphs=run_metadata.partition_graphs)

    # Construct the analyzer.
    cls._analyzer = analyzer_cli.DebugAnalyzer(cls._debug_dump)

    # Construct the handler registry.
    cls._registry = debugger_cli_common.CommandHandlerRegistry()

    # Register command handler.
    cls._registry.register_command_handler(
        "print_tensor",
        cls._analyzer.print_tensor,
        cls._analyzer.get_help("print_tensor"),
        prefix_aliases=["pt"])

  @classmethod
  def tearDownClass(cls):
    # Tear down temporary dump directory.
    shutil.rmtree(cls._dump_root)

  def testPrintLargeTensorWithoutAllOption(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["large_tensors/x:0"], screen_info={"cols": 80})

    # Assert that ellipses are present in the tensor value printout.
    self.assertIn("...,", out.lines[4])

    # 2100 still exceeds 2000.
    out = self._registry.dispatch_command(
        "print_tensor", ["large_tensors/x:0[:, 0:7]"],
        screen_info={"cols": 80})

    self.assertIn("...,", out.lines[4])

  def testPrintLargeTensorWithAllOption(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["large_tensors/x:0", "-a"],
        screen_info={"cols": 80})

    # Assert that ellipses are not present in the tensor value printout.
    self.assertNotIn("...,", out.lines[4])

    out = self._registry.dispatch_command(
        "print_tensor", ["large_tensors/x:0[:, 0:7]", "--all"],
        screen_info={"cols": 80})
    self.assertNotIn("...,", out.lines[4])


class AnalyzerCLIControlDepTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    cls._dump_root = tempfile.mkdtemp()

    cls._is_gpu_available = test.is_gpu_available()
    if cls._is_gpu_available:
      gpu_name = test_util.gpu_device_name()
      cls._main_device = "/job:localhost/replica:0/task:0" + gpu_name
    else:
      cls._main_device = "/job:localhost/replica:0/task:0/cpu:0"

    with session.Session(config=no_rewrite_session_config()) as sess:
      x_init_val = np.array([5.0, 3.0])
      x_init = constant_op.constant(x_init_val, shape=[2])
      x = variables.Variable(x_init, name="control_deps/x")

      y = math_ops.add(x, x, name="control_deps/y")
      y = control_flow_ops.with_dependencies(
          [x], y, name="control_deps/ctrl_dep_y")

      z = math_ops.multiply(x, y, name="control_deps/z")

      z = control_flow_ops.with_dependencies(
          [x, y], z, name="control_deps/ctrl_dep_z")

      x.initializer.run()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls="file://%s" % cls._dump_root)

      # Invoke Session.run().
      run_metadata = config_pb2.RunMetadata()
      sess.run(z, options=run_options, run_metadata=run_metadata)

    debug_dump = debug_data.DebugDumpDir(
        cls._dump_root, partition_graphs=run_metadata.partition_graphs)

    # Construct the analyzer.
    analyzer = analyzer_cli.DebugAnalyzer(debug_dump)

    # Construct the handler registry.
    cls._registry = debugger_cli_common.CommandHandlerRegistry()

    # Register command handlers.
    cls._registry.register_command_handler(
        "node_info",
        analyzer.node_info,
        analyzer.get_help("node_info"),
        prefix_aliases=["ni"])
    cls._registry.register_command_handler(
        "list_inputs",
        analyzer.list_inputs,
        analyzer.get_help("list_inputs"),
        prefix_aliases=["li"])
    cls._registry.register_command_handler(
        "list_outputs",
        analyzer.list_outputs,
        analyzer.get_help("list_outputs"),
        prefix_aliases=["lo"])

  @classmethod
  def tearDownClass(cls):
    # Tear down temporary dump directory.
    shutil.rmtree(cls._dump_root)

  def testNodeInfoWithControlDependencies(self):
    # Call node_info on a node with control inputs.
    out = self._registry.dispatch_command("node_info",
                                          ["control_deps/ctrl_dep_y"])

    assert_node_attribute_lines(
        self, out, "control_deps/ctrl_dep_y", "Identity",
        self._main_device, [("Add", "control_deps/y")],
        [("VariableV2", "control_deps/x")],
        [("Mul", "control_deps/z")],
        [("Identity", "control_deps/ctrl_dep_z")])

    # Call node info on a node with control recipients.
    out = self._registry.dispatch_command("ni", ["control_deps/x"])

    assert_node_attribute_lines(self, out, "control_deps/x", "VariableV2",
                                self._main_device, [], [],
                                [("Identity", "control_deps/x/read")],
                                [("Identity", "control_deps/ctrl_dep_y"),
                                 ("Identity", "control_deps/ctrl_dep_z")])

    # Verify the menu items (command shortcuts) in the output.
    check_menu_item(self, out, 10,
                    len(out.lines[10]) - len("control_deps/x/read"),
                    len(out.lines[10]), "ni -a -d -t control_deps/x/read")
    if out.lines[13].endswith("control_deps/ctrl_dep_y"):
      y_line = 13
      z_line = 14
    else:
      y_line = 14
      z_line = 13
    check_menu_item(self, out, y_line,
                    len(out.lines[y_line]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[y_line]),
                    "ni -a -d -t control_deps/ctrl_dep_y")
    check_menu_item(self, out, z_line,
                    len(out.lines[z_line]) - len("control_deps/ctrl_dep_z"),
                    len(out.lines[z_line]),
                    "ni -a -d -t control_deps/ctrl_dep_z")

  def testListInputsNonRecursiveNoControl(self):
    """List inputs non-recursively, without any control inputs."""

    # Do not include node op types.
    node_name = "control_deps/z"
    out = self._registry.dispatch_command("list_inputs", [node_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 1):" % node_name,
        "|- (1) control_deps/x/read", "|  |- ...",
        "|- (1) control_deps/ctrl_dep_y", "   |- ...", "", "Legend:",
        "  (d): recursion depth = d."
    ], out.lines)

    # Include node op types.
    out = self._registry.dispatch_command("li", ["-t", node_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 1):" % node_name,
        "|- (1) [Identity] control_deps/x/read", "|  |- ...",
        "|- (1) [Identity] control_deps/ctrl_dep_y", "   |- ...", "", "Legend:",
        "  (d): recursion depth = d.", "  [Op]: Input node has op type Op."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

    # Verify that the node name has bold attribute.
    self.assertEqual([(16, 16 + len(node_name), "bold")], out.font_attr_segs[0])

    # Verify the menu items (command shortcuts) in the output.
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/x/read"),
                    len(out.lines[1]), "li -c -r control_deps/x/read")
    check_menu_item(self, out, 3,
                    len(out.lines[3]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[3]), "li -c -r control_deps/ctrl_dep_y")

  def testListInputsNonRecursiveNoControlUsingTensorName(self):
    """List inputs using the name of an output tensor of the node."""

    # Do not include node op types.
    node_name = "control_deps/z"
    tensor_name = node_name + ":0"
    out = self._registry.dispatch_command("list_inputs", [tensor_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 1):" % node_name,
        "|- (1) control_deps/x/read", "|  |- ...",
        "|- (1) control_deps/ctrl_dep_y", "   |- ...", "", "Legend:",
        "  (d): recursion depth = d."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/x/read"),
                    len(out.lines[1]), "li -c -r control_deps/x/read")
    check_menu_item(self, out, 3,
                    len(out.lines[3]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[3]), "li -c -r control_deps/ctrl_dep_y")

  def testListInputsNonRecursiveWithControls(self):
    """List inputs non-recursively, with control inputs."""
    node_name = "control_deps/ctrl_dep_z"
    out = self._registry.dispatch_command("li", ["-t", node_name, "-c"])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 1, " % node_name +
        "control inputs included):", "|- (1) [Mul] control_deps/z", "|  |- ...",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y", "|  |- ...",
        "|- (1) (Ctrl) [VariableV2] control_deps/x", "", "Legend:",
        "  (d): recursion depth = d.", "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/z"),
                    len(out.lines[1]), "li -c -r control_deps/z")
    check_menu_item(self, out, 3,
                    len(out.lines[3]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[3]), "li -c -r control_deps/ctrl_dep_y")
    check_menu_item(self, out, 5,
                    len(out.lines[5]) - len("control_deps/x"),
                    len(out.lines[5]), "li -c -r control_deps/x")

  def testListInputsRecursiveWithControls(self):
    """List inputs recursively, with control inputs."""
    node_name = "control_deps/ctrl_dep_z"
    out = self._registry.dispatch_command("li", ["-c", "-r", "-t", node_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 20, " % node_name +
        "control inputs included):", "|- (1) [Mul] control_deps/z",
        "|  |- (2) [Identity] control_deps/x/read",
        "|  |  |- (3) [VariableV2] control_deps/x",
        "|  |- (2) [Identity] control_deps/ctrl_dep_y",
        "|     |- (3) [Add] control_deps/y",
        "|     |  |- (4) [Identity] control_deps/x/read",
        "|     |  |  |- (5) [VariableV2] control_deps/x",
        "|     |  |- (4) [Identity] control_deps/x/read",
        "|     |     |- (5) [VariableV2] control_deps/x",
        "|     |- (3) (Ctrl) [VariableV2] control_deps/x",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y",
        "|  |- (2) [Add] control_deps/y",
        "|  |  |- (3) [Identity] control_deps/x/read",
        "|  |  |  |- (4) [VariableV2] control_deps/x",
        "|  |  |- (3) [Identity] control_deps/x/read",
        "|  |     |- (4) [VariableV2] control_deps/x",
        "|  |- (2) (Ctrl) [VariableV2] control_deps/x",
        "|- (1) (Ctrl) [VariableV2] control_deps/x", "", "Legend:",
        "  (d): recursion depth = d.", "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/z"),
                    len(out.lines[1]), "li -c -r control_deps/z")
    check_menu_item(self, out, 11,
                    len(out.lines[11]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[11]), "li -c -r control_deps/ctrl_dep_y")
    check_menu_item(self, out, 18,
                    len(out.lines[18]) - len("control_deps/x"),
                    len(out.lines[18]), "li -c -r control_deps/x")

  def testListInputsRecursiveWithControlsWithDepthLimit(self):
    """List inputs recursively, with control inputs and a depth limit."""
    node_name = "control_deps/ctrl_dep_z"
    out = self._registry.dispatch_command(
        "li", ["-c", "-r", "-t", "-d", "2", node_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 2, " % node_name +
        "control inputs included):", "|- (1) [Mul] control_deps/z",
        "|  |- (2) [Identity] control_deps/x/read", "|  |  |- ...",
        "|  |- (2) [Identity] control_deps/ctrl_dep_y", "|     |- ...",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y",
        "|  |- (2) [Add] control_deps/y", "|  |  |- ...",
        "|  |- (2) (Ctrl) [VariableV2] control_deps/x",
        "|- (1) (Ctrl) [VariableV2] control_deps/x", "", "Legend:",
        "  (d): recursion depth = d.", "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/z"),
                    len(out.lines[1]), "li -c -r control_deps/z")
    check_menu_item(self, out, 10,
                    len(out.lines[10]) - len("control_deps/x"),
                    len(out.lines[10]), "li -c -r control_deps/x")

  def testListInputsNodeWithoutInputs(self):
    """List the inputs to a node without any input."""
    node_name = "control_deps/x"
    out = self._registry.dispatch_command("li", ["-c", "-r", "-t", node_name])

    self.assertEqual([
        "Inputs to node \"%s\" (Depth limit = 20, control " % node_name +
        "inputs included):", "  [None]", "", "Legend:",
        "  (d): recursion depth = d.", "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."
    ], out.lines)
    check_main_menu(
        self,
        out,
        list_tensors_enabled=True,
        node_info_node_name=node_name,
        print_tensor_node_name=node_name,
        list_outputs_node_name=node_name)

  def testListInputsNonexistentNode(self):
    out = self._registry.dispatch_command(
        "list_inputs", ["control_deps/z/foo"])

    self.assertEqual([
        "ERROR: There is no node named \"control_deps/z/foo\" in the "
        "partition graphs"], out.lines)

  def testListRecipientsRecursiveWithControlsWithDepthLimit(self):
    """List recipients recursively, with control inputs and a depth limit."""

    out = self._registry.dispatch_command(
        "lo", ["-c", "-r", "-t", "-d", "1", "control_deps/x"])

    self.assertEqual([
        "Recipients of node \"control_deps/x\" (Depth limit = 1, control "
        "recipients included):",
        "|- (1) [Identity] control_deps/x/read",
        "|  |- ...",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y",
        "|  |- ...",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_z",
        "", "Legend:", "  (d): recursion depth = d.",
        "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."], out.lines)
    check_menu_item(self, out, 1,
                    len(out.lines[1]) - len("control_deps/x/read"),
                    len(out.lines[1]), "lo -c -r control_deps/x/read")
    check_menu_item(self, out, 3,
                    len(out.lines[3]) - len("control_deps/ctrl_dep_y"),
                    len(out.lines[3]), "lo -c -r control_deps/ctrl_dep_y")
    check_menu_item(self, out, 5,
                    len(out.lines[5]) - len("control_deps/ctrl_dep_z"),
                    len(out.lines[5]), "lo -c -r control_deps/ctrl_dep_z")

    # Verify the bold attribute of the node name.
    self.assertEqual([(20, 20 + len("control_deps/x"), "bold")],
                     out.font_attr_segs[0])


class AnalyzerCLIWhileLoopTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    cls._dump_root = tempfile.mkdtemp()

    with session.Session(config=no_rewrite_session_config()) as sess:
      loop_var = constant_op.constant(0, name="while_loop_test/loop_var")
      cond = lambda loop_var: math_ops.less(loop_var, 10)
      body = lambda loop_var: math_ops.add(loop_var, 1)
      while_loop = control_flow_ops.while_loop(
          cond, body, [loop_var], parallel_iterations=1)

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_url = "file://%s" % cls._dump_root

      watch_opts = run_options.debug_options.debug_tensor_watch_opts

      # Add debug tensor watch for "while/Identity".
      watch = watch_opts.add()
      watch.node_name = "while/Identity"
      watch.output_slot = 0
      watch.debug_ops.append("DebugIdentity")
      watch.debug_urls.append(debug_url)

      # Invoke Session.run().
      run_metadata = config_pb2.RunMetadata()
      sess.run(while_loop, options=run_options, run_metadata=run_metadata)

    cls._debug_dump = debug_data.DebugDumpDir(
        cls._dump_root, partition_graphs=run_metadata.partition_graphs)

    cls._analyzer = analyzer_cli.DebugAnalyzer(cls._debug_dump)
    cls._registry = debugger_cli_common.CommandHandlerRegistry()
    cls._registry.register_command_handler(
        "list_tensors",
        cls._analyzer.list_tensors,
        cls._analyzer.get_help("list_tensors"),
        prefix_aliases=["lt"])
    cls._registry.register_command_handler(
        "print_tensor",
        cls._analyzer.print_tensor,
        cls._analyzer.get_help("print_tensor"),
        prefix_aliases=["pt"])

  @classmethod
  def tearDownClass(cls):
    # Tear down temporary dump directory.
    shutil.rmtree(cls._dump_root)

  def testMultipleDumpsPrintTensorNoNumber(self):
    output = self._registry.dispatch_command("pt", ["while/Identity:0"])

    self.assertEqual("Tensor \"while/Identity:0\" generated 10 dumps:",
                     output.lines[0])

    for i in xrange(10):
      self.assertTrue(output.lines[i + 1].startswith("#%d" % i))
      self.assertTrue(output.lines[i + 1].endswith(
          " ms] while/Identity:0:DebugIdentity"))

    self.assertEqual(
        "You can use the -n (--number) flag to specify which dump to print.",
        output.lines[-3])
    self.assertEqual("For example:", output.lines[-2])
    self.assertEqual("  print_tensor while/Identity:0 -n 0", output.lines[-1])

  def testMultipleDumpsPrintTensorWithNumber(self):
    for i in xrange(5):
      output = self._registry.dispatch_command(
          "pt", ["while/Identity:0", "-n", "%d" % i])

      self.assertEqual("Tensor \"while/Identity:0:DebugIdentity (dump #%d)\":" %
                       i, output.lines[0])
      self.assertEqual("  dtype: int32", output.lines[1])
      self.assertEqual("  shape: ()", output.lines[2])
      self.assertEqual("", output.lines[3])
      self.assertTrue(output.lines[4].startswith("array(%d" % i))
      self.assertTrue(output.lines[4].endswith(")"))

  def testMultipleDumpsPrintTensorInvalidNumber(self):
    output = self._registry.dispatch_command("pt",
                                             ["while/Identity:0", "-n", "10"])

    self.assertEqual([
        "ERROR: Specified number (10) exceeds the number of available dumps "
        "(10) for tensor while/Identity:0"
    ], output.lines)


if __name__ == "__main__":
  googletest.main()
