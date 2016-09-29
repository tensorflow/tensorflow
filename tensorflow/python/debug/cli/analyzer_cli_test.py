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
                          op_type_regex=None):
  """Check RichTextLines output for list_tensors commands.

  Args:
    tst: A test_util.TensorFlowTestCase instance.
    out: The RichTextLines object to be checked.
    expected_tensor_names: Expected tensor names in the list.
    node_name_regex: Optional: node name regex filter.
    op_type_regex: Optional: op type regex filter.
  """

  line_iter = iter(out.lines)

  num_tensors = len(expected_tensor_names)
  tst.assertEqual("%d dumped tensor(s):" % num_tensors, next(line_iter))

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


def check_error_output(tst, out, command_prefix, args):
  """Check RichTextLines output from invalid/erroneous commands.

  Args:
    tst: A test_util.TensorFlowTestCase instance.
    out: The RichTextLines object to be checked.
    command_prefix: The command prefix of the command that caused the error.
    args: The arguments (excluding prefix) of the command that caused the error.
  """

  tst.assertEqual(2, len(out.lines))
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

    debug_dump = debug_data.DebugDumpDir(
        cls._dump_root, partition_graphs=run_metadata.partition_graphs)

    # Construct the analyzer.
    analyzer = analyzer_cli.DebugAnalyzer(debug_dump)

    # Construct the handler registry.
    cls._registry = debugger_cli_common.CommandHandlerRegistry()

    # Register command handlers.
    cls._registry.register_command_handler(
        "list_tensors",
        analyzer.list_tensors,
        analyzer.get_help("list_tensors"),
        prefix_aliases=["lt"])
    cls._registry.register_command_handler(
        "node_info",
        analyzer.node_info,
        analyzer.get_help("node_info"),
        prefix_aliases=["ni"])
    cls._registry.register_command_handler(
        "print_tensor",
        analyzer.print_tensor,
        analyzer.get_help("print_tensor"),
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

  def testListTensorsInvalidOptions(self):
    out = self._registry.dispatch_command("list_tensors", ["--foo"])
    check_error_output(self, out, "list_tensors", ["--foo"])

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


if __name__ == "__main__":
  googletest.main()
