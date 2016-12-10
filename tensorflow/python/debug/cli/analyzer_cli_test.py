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

import shutil
import tempfile

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug import debug_data
from tensorflow.python.debug import debug_utils
from tensorflow.python.debug.cli import analyzer_cli
from tensorflow.python.debug.cli import debugger_cli_common
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


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


def assert_listed_tensors(tst,
                          out,
                          expected_tensor_names,
                          node_name_regex=None,
                          op_type_regex=None,
                          tensor_filter_name=None):
  """Check RichTextLines output for list_tensors commands.

  Args:
    tst: A test_util.TensorFlowTestCase instance.
    out: The RichTextLines object to be checked.
    expected_tensor_names: Expected tensor names in the list.
    node_name_regex: Optional: node name regex filter.
    op_type_regex: Optional: op type regex filter.
    tensor_filter_name: Optional: name of the tensor filter.
  """

  line_iter = iter(out.lines)

  num_tensors = len(expected_tensor_names)

  if tensor_filter_name is None:
    tst.assertEqual("%d dumped tensor(s):" % num_tensors, next(line_iter))
  else:
    tst.assertEqual("%d dumped tensor(s) passing filter \"%s\":" %
                    (num_tensors, tensor_filter_name), next(line_iter))

  if op_type_regex is not None:
    tst.assertEqual("Op type regex filter: \"%s\"" % op_type_regex,
                    next(line_iter))

  if node_name_regex is not None:
    tst.assertEqual("Node name regex filter: \"%s\"" % node_name_regex,
                    next(line_iter))

  tst.assertEqual("", next(line_iter))

  # Verify the listed tensors and their timestamps.
  tensor_timestamps = []
  tensor_names = []
  for line in line_iter:
    rel_time = float(line.split("ms] ")[0].replace("[", ""))
    tst.assertGreaterEqual(rel_time, 0.0)

    tensor_timestamps.append(rel_time)
    tensor_names.append(line.split("ms] ")[1])

  # Verify that the tensors should be listed in ascending order of their
  # timestamps.
  tst.assertEqual(sorted(tensor_timestamps), tensor_timestamps)

  # Verify that the tensors are all listed.
  for tensor_name in expected_tensor_names:
    tst.assertIn(tensor_name, tensor_names)


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
                                num_dumped_tensors=None):
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
    tst.assertEqual("", next(line_iter))

    tst.assertEqual("%d dumped tensor(s):" % num_dumped_tensors,
                    next(line_iter))

    dump_timestamps_ms = []
    for _ in xrange(num_dumped_tensors):
      line = next(line_iter)

      tst.assertStartsWith(line.strip(), "Slot 0 @ DebugIdentity @")
      tst.assertTrue(line.strip().endswith(" ms"))

      dump_timestamp_ms = float(line.strip().split(" @ ")[-1].replace("ms", ""))
      tst.assertGreaterEqual(dump_timestamp_ms, 0.0)

      dump_timestamps_ms.append(dump_timestamp_ms)

    tst.assertEqual(sorted(dump_timestamps_ms), dump_timestamps_ms)


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


class AnalyzerCLISimpleMulAddTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    cls._dump_root = tempfile.mkdtemp()

    cls._is_gpu_available = test.is_gpu_available()
    if cls._is_gpu_available:
      cls._main_device = "/job:localhost/replica:0/task:0/gpu:0"
    else:
      cls._main_device = "/job:localhost/replica:0/task:0/cpu:0"

    with session.Session() as sess:
      u_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      v_init_val = np.array([[2.0], [-1.0]])

      u_name = "simple_mul_add/u"
      v_name = "simple_mul_add/v"

      u_init = constant_op.constant(u_init_val, shape=[2, 2])
      u = variables.Variable(u_init, name=u_name)
      v_init = constant_op.constant(v_init_val, shape=[2, 1])
      v = variables.Variable(v_init, name=v_name)

      w = math_ops.matmul(u, v, name="simple_mul_add/matmul")

      x = math_ops.add(w, w, name="simple_mul_add/add")

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

  @classmethod
  def tearDownClass(cls):
    # Tear down temporary dump directory.
    shutil.rmtree(cls._dump_root)

  def testListTensors(self):
    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", [])

    assert_listed_tensors(self, out, [
        "simple_mul_add/u:0", "simple_mul_add/v:0", "simple_mul_add/u/read:0",
        "simple_mul_add/v/read:0", "simple_mul_add/matmul:0",
        "simple_mul_add/add:0"
    ])

  def testListTensorsFilterByNodeNameRegex(self):
    out = self._registry.dispatch_command("list_tensors",
                                          ["--node_name_filter", ".*read.*"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0"
        ],
        node_name_regex=".*read.*")

    out = self._registry.dispatch_command("list_tensors", ["-n", "^read"])
    assert_listed_tensors(self, out, [], node_name_regex="^read")

  def testListTensorFilterByOpTypeRegex(self):
    out = self._registry.dispatch_command("list_tensors",
                                          ["--op_type_filter", "Identity"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/u/read:0", "simple_mul_add/v/read:0"
        ],
        op_type_regex="Identity")

    out = self._registry.dispatch_command("list_tensors",
                                          ["-t", "(Add|MatMul)"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/add:0", "simple_mul_add/matmul:0"
        ],
        op_type_regex="(Add|MatMul)")

  def testListTensorFilterByNodeNameRegexAndOpTypeRegex(self):
    out = self._registry.dispatch_command(
        "list_tensors", ["-t", "(Add|MatMul)", "-n", ".*add$"])
    assert_listed_tensors(
        self,
        out, [
            "simple_mul_add/add:0"
        ],
        node_name_regex=".*add$",
        op_type_regex="(Add|MatMul)")

  def testListTensorsFilterNanOrInf(self):
    """Test register and invoke a tensor filter."""

    # First, register the filter.
    self._analyzer.add_tensor_filter("has_inf_or_nan",
                                     debug_data.has_inf_or_nan)

    # Use shorthand alias for the command prefix.
    out = self._registry.dispatch_command("lt", ["-f", "has_inf_or_nan"])

    # This TF graph run did not generate any bad numerical values.
    assert_listed_tensors(self, out, [], tensor_filter_name="has_inf_or_nan")
    # TODO(cais): A test with some actual bad numerical values.

  def testListTensorNonexistentFilter(self):
    """Test attempt to use a nonexistent tensor filter."""

    out = self._registry.dispatch_command("lt", ["-f", "foo_filter"])

    self.assertEqual(["ERROR: There is no tensor filter named \"foo_filter\"."],
                     out.lines)

  def testListTensorsInvalidOptions(self):
    out = self._registry.dispatch_command("list_tensors", ["--foo"])
    check_syntax_error_output(self, out, "list_tensors")

  def testNodeInfoByNodeName(self):
    out = self._registry.dispatch_command("node_info",
                                          ["simple_mul_add/matmul"])

    recipients = [("Add", "simple_mul_add/add"), ("Add", "simple_mul_add/add")]

    assert_node_attribute_lines(self, out, "simple_mul_add/matmul", "MatMul",
                                self._main_device,
                                [("Identity", "simple_mul_add/u/read"),
                                 ("Identity", "simple_mul_add/v/read")], [],
                                recipients, [])

  def testNodeInfoShowAttributes(self):
    out = self._registry.dispatch_command("node_info",
                                          ["-a", "simple_mul_add/matmul"])

    assert_node_attribute_lines(
        self,
        out,
        "simple_mul_add/matmul",
        "MatMul",
        self._main_device, [("Identity", "simple_mul_add/u/read"),
                            ("Identity", "simple_mul_add/v/read")], [],
        [("Add", "simple_mul_add/add"), ("Add", "simple_mul_add/add")], [],
        attr_key_val_pairs=[("transpose_a", "b: false"),
                            ("transpose_b", "b: false"),
                            ("T", "type: DT_DOUBLE")])

  def testNodeInfoShowDumps(self):
    out = self._registry.dispatch_command("node_info",
                                          ["-d", "simple_mul_add/matmul"])

    assert_node_attribute_lines(
        self,
        out,
        "simple_mul_add/matmul",
        "MatMul",
        self._main_device, [("Identity", "simple_mul_add/u/read"),
                            ("Identity", "simple_mul_add/v/read")], [],
        [("Add", "simple_mul_add/add"), ("Add", "simple_mul_add/add")], [],
        num_dumped_tensors=1)

  def testNodeInfoByTensorName(self):
    out = self._registry.dispatch_command("node_info",
                                          ["simple_mul_add/u/read:0"])

    assert_node_attribute_lines(self, out, "simple_mul_add/u/read", "Identity",
                                self._main_device,
                                [("Variable", "simple_mul_add/u")], [],
                                [("MatMul", "simple_mul_add/matmul")], [])

  def testNodeInfoNonexistentNodeName(self):
    out = self._registry.dispatch_command("node_info", ["bar"])
    self.assertEqual(
        ["ERROR: There is no node named \"bar\" in the partition graphs"],
        out.lines)
    # Check color indicating error.
    self.assertEqual({0: [(0, 59, "red")]}, out.font_attr_segs)

  def testPrintTensor(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["simple_mul_add/matmul:0"], screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"simple_mul_add/matmul:0:DebugIdentity\":",
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    self.assertIn(5, out.annotations)

  def testPrintTensorHighlightingRanges(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["simple_mul_add/matmul:0", "--ranges", "[-inf, 0.0]"],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"simple_mul_add/matmul:0:DebugIdentity\": "
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
        "print_tensor",
        ["simple_mul_add/matmul:0", "--ranges", "[[-inf, -5.5], [5.5, inf]]"],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"simple_mul_add/matmul:0:DebugIdentity\": "
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

  def testPrintTensorWithSlicing(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["simple_mul_add/matmul:0[1, :]"],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"simple_mul_add/matmul:0:DebugIdentity[1, :]\":",
        "  dtype: float64", "  shape: (1,)", "", "array([-2.])"
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)

  def testPrintTensorInvalidSlicingString(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["simple_mul_add/matmul:0[1, foo()]"],
        screen_info={"cols": 80})

    self.assertEqual("Error occurred during handling of command: print_tensor "
                     "simple_mul_add/matmul:0[1, foo()]:", out.lines[0])
    self.assertEqual("ValueError: Invalid tensor-slicing string.",
                     out.lines[-2])

  def testPrintTensorValidExplicitNumber(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["simple_mul_add/matmul:0", "-n", "0"],
        screen_info={"cols": 80})

    self.assertEqual([
        "Tensor \"simple_mul_add/matmul:0:DebugIdentity\":",
        "  dtype: float64",
        "  shape: (2, 1)",
        "",
        "array([[ 7.],",
        "       [-2.]])",
    ], out.lines)

    self.assertIn("tensor_metadata", out.annotations)
    self.assertIn(4, out.annotations)
    self.assertIn(5, out.annotations)

  def testPrintTensorInvalidExplicitNumber(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["simple_mul_add/matmul:0", "-n", "1"],
        screen_info={"cols": 80})

    self.assertEqual([
        "ERROR: Invalid number (1) for tensor simple_mul_add/matmul:0, "
        "which generated one dump."
    ], out.lines)

    self.assertNotIn("tensor_metadata", out.annotations)

  def testPrintTensorMissingOutputSlot(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["simple_mul_add/matmul"])

    self.assertEqual([
        "ERROR: \"simple_mul_add/matmul\" is not a valid tensor name"
    ], out.lines)

  def testPrintTensorNonexistentNodeName(self):
    out = self._registry.dispatch_command(
        "print_tensor", ["simple_mul_add/matmul/foo:0"])

    self.assertEqual([
        "ERROR: Node \"simple_mul_add/matmul/foo\" does not exist in partition "
        "graphs"
    ], out.lines)

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


class AnalyzerCLIPrintLargeTensorTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    cls._dump_root = tempfile.mkdtemp()

    with session.Session() as sess:
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
      cls._main_device = "/job:localhost/replica:0/task:0/gpu:0"
    else:
      cls._main_device = "/job:localhost/replica:0/task:0/cpu:0"

    with session.Session() as sess:
      x_init_val = np.array([5.0, 3.0])
      x_init = constant_op.constant(x_init_val, shape=[2])
      x = variables.Variable(x_init, name="control_deps/x")

      y = math_ops.add(x, x, name="control_deps/y")
      y = control_flow_ops.with_dependencies(
          [x], y, name="control_deps/ctrl_dep_y")

      z = math_ops.mul(x, y, name="control_deps/z")

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
        [("Variable", "control_deps/x")],
        [("Mul", "control_deps/z")],
        [("Identity", "control_deps/ctrl_dep_z")])

    # Call node info on a node with control recipients.
    out = self._registry.dispatch_command("ni", ["control_deps/x"])

    assert_node_attribute_lines(self, out, "control_deps/x", "Variable",
                                self._main_device, [], [],
                                [("Identity", "control_deps/x/read")],
                                [("Identity", "control_deps/ctrl_dep_y"),
                                 ("Identity", "control_deps/ctrl_dep_z")])

  def testListInputsNonRecursiveNoControl(self):
    """List inputs non-recursively, without any control inputs."""

    # Do not include node op types.
    out = self._registry.dispatch_command("list_inputs", ["control_deps/z"])

    self.assertEqual([
        "Inputs to node \"control_deps/z\" (Depth limit = 1):",
        "|- (1) control_deps/x/read",
        "|  |- ...",
        "|- (1) control_deps/ctrl_dep_y",
        "   |- ...",
        "", "Legend:", "  (d): recursion depth = d."], out.lines)

    # Include node op types.
    out = self._registry.dispatch_command("li", ["-t", "control_deps/z"])

    self.assertEqual([
        "Inputs to node \"control_deps/z\" (Depth limit = 1):",
        "|- (1) [Identity] control_deps/x/read",
        "|  |- ...",
        "|- (1) [Identity] control_deps/ctrl_dep_y",
        "   |- ...",
        "", "Legend:", "  (d): recursion depth = d.",
        "  [Op]: Input node has op type Op."], out.lines)

  def testListInputsNonRecursiveNoControlUsingTensorName(self):
    """List inputs using the name of an output tensor of the node."""

    # Do not include node op types.
    out = self._registry.dispatch_command("list_inputs", ["control_deps/z:0"])

    self.assertEqual([
        "Inputs to node \"control_deps/z\" (Depth limit = 1):",
        "|- (1) control_deps/x/read",
        "|  |- ...",
        "|- (1) control_deps/ctrl_dep_y",
        "   |- ...",
        "", "Legend:", "  (d): recursion depth = d."], out.lines)

  def testListInputsNonRecursiveWithControls(self):
    """List inputs non-recursively, with control inputs."""

    out = self._registry.dispatch_command(
        "li", ["-t", "control_deps/ctrl_dep_z", "-c"])

    self.assertEqual([
        "Inputs to node \"control_deps/ctrl_dep_z\" (Depth limit = 1, "
        "control inputs included):",
        "|- (1) [Mul] control_deps/z",
        "|  |- ...",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y",
        "|  |- ...",
        "|- (1) (Ctrl) [Variable] control_deps/x",
        "", "Legend:", "  (d): recursion depth = d.",
        "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."], out.lines)

  def testListInputsRecursiveWithControls(self):
    """List inputs recursively, with control inputs."""

    out = self._registry.dispatch_command(
        "li", ["-c", "-r", "-t", "control_deps/ctrl_dep_z"])

    self.assertEqual([
        "Inputs to node \"control_deps/ctrl_dep_z\" (Depth limit = 20, "
        "control inputs included):",
        "|- (1) [Mul] control_deps/z",
        "|  |- (2) [Identity] control_deps/x/read",
        "|  |  |- (3) [Variable] control_deps/x",
        "|  |- (2) [Identity] control_deps/ctrl_dep_y",
        "|     |- (3) [Add] control_deps/y",
        "|     |  |- (4) [Identity] control_deps/x/read",
        "|     |  |  |- (5) [Variable] control_deps/x",
        "|     |  |- (4) [Identity] control_deps/x/read",
        "|     |     |- (5) [Variable] control_deps/x",
        "|     |- (3) (Ctrl) [Variable] control_deps/x",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y",
        "|  |- (2) [Add] control_deps/y",
        "|  |  |- (3) [Identity] control_deps/x/read",
        "|  |  |  |- (4) [Variable] control_deps/x",
        "|  |  |- (3) [Identity] control_deps/x/read",
        "|  |     |- (4) [Variable] control_deps/x",
        "|  |- (2) (Ctrl) [Variable] control_deps/x",
        "|- (1) (Ctrl) [Variable] control_deps/x",
        "", "Legend:", "  (d): recursion depth = d.",
        "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."], out.lines)

  def testListInputsRecursiveWithControlsWithDepthLimit(self):
    """List inputs recursively, with control inputs and a depth limit."""

    out = self._registry.dispatch_command(
        "li", ["-c", "-r", "-t", "-d", "2", "control_deps/ctrl_dep_z"])

    self.assertEqual([
        "Inputs to node \"control_deps/ctrl_dep_z\" (Depth limit = 2, "
        "control inputs included):",
        "|- (1) [Mul] control_deps/z",
        "|  |- (2) [Identity] control_deps/x/read",
        "|  |  |- ...",
        "|  |- (2) [Identity] control_deps/ctrl_dep_y",
        "|     |- ...",
        "|- (1) (Ctrl) [Identity] control_deps/ctrl_dep_y",
        "|  |- (2) [Add] control_deps/y",
        "|  |  |- ...",
        "|  |- (2) (Ctrl) [Variable] control_deps/x",
        "|- (1) (Ctrl) [Variable] control_deps/x",
        "", "Legend:", "  (d): recursion depth = d.",
        "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."], out.lines)

  def testListInputsNodeWithoutInputs(self):
    """List the inputs to a node without any input."""
    out = self._registry.dispatch_command(
        "li", ["-c", "-r", "-t", "control_deps/x"])

    self.assertEqual([
        "Inputs to node \"control_deps/x\" (Depth limit = 20, control inputs "
        "included):",
        "  [None]",
        "", "Legend:", "  (d): recursion depth = d.",
        "  (Ctrl): Control input.",
        "  [Op]: Input node has op type Op."], out.lines)

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


class AnalyzerCLIWhileLoopTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    cls._dump_root = tempfile.mkdtemp()

    with session.Session() as sess:
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
        "Use the -n (--number) flag to specify which dump to print.",
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
      self.assertEqual("array(%d, dtype=int32)" % i, output.lines[4])

  def testMultipleDumpsPrintTensorInvalidNumber(self):
    output = self._registry.dispatch_command("pt",
                                             ["while/Identity:0", "-n", "10"])

    self.assertEqual([
        "ERROR: Specified number (10) exceeds the number of available dumps "
        "(10) for tensor while/Identity:0"
    ], output.lines)


if __name__ == "__main__":
  googletest.main()
