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
"""Tests of the Stepper CLI Backend."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.cli import stepper_cli
from tensorflow.python.debug.lib import stepper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import gradient_descent

# Regex pattern for a node line in the stepper CLI output.
NODE_LINE_PATTERN = re.compile(r".*\(.*\).*\[.*\].*")


def _parse_sorted_nodes_list(lines):
  """Parsed a list of lines to extract the node list.

  Args:
    lines: (list of str) Lines from which the node list and associated
      information will be extracted.

  Returns:
    (list of str) The list of node names.
    (list of str) The list of status labels.
    (int) 0-based index among the nodes for the node pointed by the next-node
      pointer. If no such node exists, -1.
  """

  node_names = []
  status_labels = []
  node_pointer = -1

  node_line_counter = 0
  for line in lines:
    if NODE_LINE_PATTERN.match(line):
      node_names.append(line.split(" ")[-1])

      idx_left_bracket = line.index("[")
      idx_right_bracket = line.index("]")
      status_labels.append(line[idx_left_bracket + 1:idx_right_bracket])
      if line.strip().startswith(
          stepper_cli.NodeStepperCLI.NEXT_NODE_POINTER_STR):
        node_pointer = node_line_counter

      node_line_counter += 1

  return node_names, status_labels, node_pointer


def _parsed_used_feeds(lines):
  feed_types = {}

  begin_line = -1
  for i, line in enumerate(lines):
    if line.startswith("Stepper used feeds:"):
      begin_line = i + 1
      break

  if begin_line == -1:
    return feed_types

  for line in lines[begin_line:]:
    line = line.strip()
    if not line:
      return feed_types
    else:
      feed_name = line.split(" : ")[0].strip()
      feed_type = line.split(" : ")[1].strip()
      feed_types[feed_name] = feed_type


def _parse_updated(lines):
  """Parse the Updated section in the output text lines.

  Args:
    lines: (list of str) The output text lines to be parsed.

  Returns:
    If the Updated section does not exist, returns None.
    Otherwise, returns the Tensor names included in the section.
  """
  updated = None

  begin_line = -1
  for i, line in enumerate(lines):
    if line.startswith("Updated:"):
      updated = []
      begin_line = i + 1
      break

  if begin_line == -1:
    return updated

  for line in lines[begin_line:]:
    line = line.strip()
    if not line:
      return updated
    else:
      updated.append(line.strip())

  return updated


class NodeStepperSimpleGraphTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self.a = variables.VariableV1(10.0, name="a")
    self.b = variables.VariableV1(20.0, name="b")

    self.c = math_ops.add(self.a, self.b, name="c")  # Should be 30.0.
    self.d = math_ops.subtract(self.a, self.c, name="d")  # Should be -20.0.
    self.e = math_ops.multiply(self.c, self.d, name="e")  # Should be -600.0.

    self.ph = array_ops.placeholder(dtypes.float32, shape=(2, 2), name="ph")
    self.f = math_ops.multiply(self.e, self.ph, name="f")

    self.opt = gradient_descent.GradientDescentOptimizer(0.1).minimize(
        self.e, name="opt")

    rewriter_config = rewriter_config_pb2.RewriterConfig(
        disable_model_pruning=True)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    config = config_pb2.ConfigProto(graph_options=graph_options)
    self.sess = session.Session(config=config)

    self.sess.run(self.a.initializer)
    self.sess.run(self.b.initializer)

  def tearDown(self):
    ops.reset_default_graph()

  def _assert_nodes_topologically_sorted_with_target_e(self, node_names):
    """Check the topologically sorted order of the node names."""

    self.assertGreaterEqual(len(node_names), 7)
    self.assertLess(node_names.index("a"), node_names.index("a/read"))
    self.assertLess(node_names.index("b"), node_names.index("b/read"))
    self.assertLess(node_names.index("a/read"), node_names.index("c"))
    self.assertLess(node_names.index("b/read"), node_names.index("c"))
    self.assertLess(node_names.index("a/read"), node_names.index("d"))
    self.assertLess(node_names.index("c"), node_names.index("d"))
    self.assertLess(node_names.index("c"), node_names.index("e"))
    self.assertLess(node_names.index("d"), node_names.index("e"))

  def _assert_nodes_topologically_sorted_with_target_f(self, node_names):
    self._assert_nodes_topologically_sorted_with_target_e(node_names)

    self.assertGreaterEqual(len(node_names), 9)
    self.assertLess(node_names.index("ph"), node_names.index("f"))
    self.assertLess(node_names.index("e"), node_names.index("f"))

  def testListingSortedNodesPresentsTransitveClosure(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.list_sorted_nodes([])
      node_names, stat_labels, node_pointer = _parse_sorted_nodes_list(
          output.lines)

      self._assert_nodes_topologically_sorted_with_target_e(node_names)
      self.assertEqual(len(node_names), len(stat_labels))
      for stat_label in stat_labels:
        self.assertEqual("      ", stat_label)
      self.assertEqual(0, node_pointer)

  def testListingSortedNodesLabelsPlaceholders(self):
    with stepper.NodeStepper(self.sess, self.f) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.list_sorted_nodes([])
      node_names, stat_labels, node_pointer = _parse_sorted_nodes_list(
          output.lines)

      self._assert_nodes_topologically_sorted_with_target_f(node_names)

      index_ph = node_names.index("ph")
      self.assertEqual(len(node_names), len(stat_labels))
      for i in xrange(len(stat_labels)):
        if index_ph == i:
          self.assertIn(stepper_cli.NodeStepperCLI.STATE_IS_PLACEHOLDER,
                        stat_labels[i])
        else:
          self.assertNotIn(stepper_cli.NodeStepperCLI.STATE_IS_PLACEHOLDER,
                           stat_labels[i])

      self.assertEqual(0, node_pointer)

  def testContToNonexistentNodeShouldError(self):
    with stepper.NodeStepper(self.sess, self.f) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.cont(["foobar"])
      self.assertEqual([
          "ERROR: foobar is not in the transitive closure of this stepper "
          "instance."
      ], output.lines)

  def testContToNodeOutsideTransitiveClosureShouldError(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.cont(["f"])
      self.assertEqual([
          "ERROR: f is not in the transitive closure of this stepper "
          "instance."
      ], output.lines)

  def testContToValidNodeShouldUpdateStatus(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.list_sorted_nodes([])
      node_names, stat_labels, node_pointer = _parse_sorted_nodes_list(
          output.lines)

      index_c = node_names.index("c")
      self.assertEqual("      ", stat_labels[index_c])
      self.assertEqual(0, node_pointer)

      output = cli.cont("c")
      self.assertIsNone(_parse_updated(output.lines))
      node_names, stat_labels, node_pointer = _parse_sorted_nodes_list(
          output.lines)

      self.assertGreaterEqual(len(node_names), 3)
      self.assertIn("c", node_names)
      index_c = node_names.index("c")
      self.assertEqual(index_c, node_pointer)
      self.assertIn(stepper_cli.NodeStepperCLI.STATE_CONT, stat_labels[index_c])

      output = cli.cont("d")
      self.assertIsNone(_parse_updated(output.lines))
      node_names, stat_labels, node_pointer = _parse_sorted_nodes_list(
          output.lines)

      used_feed_types = _parsed_used_feeds(output.lines)
      self.assertEqual({
          "c:0": stepper.NodeStepper.FEED_TYPE_HANDLE,
          "a/read:0": stepper.NodeStepper.FEED_TYPE_DUMPED_INTERMEDIATE,
      }, used_feed_types)

      self.assertGreaterEqual(len(node_names), 3)
      self.assertIn("d", node_names)
      index_d = node_names.index("d")
      self.assertEqual(index_d, node_pointer)
      self.assertIn(stepper_cli.NodeStepperCLI.STATE_CONT, stat_labels[index_d])

  def testSteppingOneStepAtATimeShouldUpdateStatus(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.list_sorted_nodes([])
      orig_node_names, _, node_pointer = _parse_sorted_nodes_list(output.lines)
      self.assertEqual(0, node_pointer)

      for i in xrange(len(orig_node_names)):
        output = cli.step([])
        node_names, stat_labels, node_pointer = _parse_sorted_nodes_list(
            output.lines)

        next_node_name = node_names[node_pointer]
        self.assertEqual(orig_node_names[i], next_node_name)

        self.assertIn(stepper_cli.NodeStepperCLI.STATE_CONT,
                      stat_labels[node_pointer])

        # The order in which the nodes are listed should not change as the
        # stepping happens.
        output = cli.list_sorted_nodes([])
        node_names, _, node_pointer = _parse_sorted_nodes_list(output.lines)
        self.assertEqual(orig_node_names, node_names)

        if i < len(orig_node_names) - 1:
          self.assertEqual(i + 1, node_pointer)
        else:
          # Stepped over the limit. Pointer should be at -1.
          self.assertEqual(-1, node_pointer)

      # Attempt to step once more after the end has been reached should error
      # out.
      output = cli.step([])
      self.assertEqual([
          "ERROR: Cannot step any further because the end of the sorted "
          "transitive closure has been reached."
      ], output.lines)

  def testSteppingMultipleStepsUpdatesStatus(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.list_sorted_nodes([])
      orig_node_names, _, _ = _parse_sorted_nodes_list(output.lines)

      output = cli.step(["-t", "3"])
      node_names, stat_labels, node_pointer = _parse_sorted_nodes_list(
          output.lines)

      self.assertEqual(orig_node_names[2], node_names[node_pointer])

      for i in xrange(node_pointer):
        self.assertIn(stepper_cli.NodeStepperCLI.STATE_CONT, stat_labels[i])

      for i in xrange(node_pointer + 1, len(stat_labels)):
        self.assertNotIn(stepper_cli.NodeStepperCLI.STATE_CONT, stat_labels[i])

  def testContToNodeWithoutOutputTensorInClosureShowsNoHandleCached(self):
    with stepper.NodeStepper(self.sess, self.opt) as node_stepper:
      sorted_nodes = node_stepper.sorted_nodes()
      closure_elements = node_stepper.closure_elements()

      # Find a node which is in the list of sorted nodes, but whose output
      # Tensor is not in the transitive closure.
      no_output_node = None
      for node in sorted_nodes:
        if (node + ":0" not in closure_elements and
            node + ":1" not in closure_elements):
          no_output_node = node
          break

      self.assertIsNotNone(no_output_node)

      cli = stepper_cli.NodeStepperCLI(node_stepper)
      output = cli.cont([no_output_node])
      self.assertIsNone(_parse_updated(output.lines))
      node_names, stat_labels, node_pointer = _parse_sorted_nodes_list(
          output.lines)

      self.assertEqual(no_output_node, node_names[node_pointer])
      self.assertNotIn(stepper_cli.NodeStepperCLI.STATE_CONT,
                       stat_labels[node_pointer])

  def testContToUpdateNodeWithTrackingLeadsToDirtyVariableLabel(self):
    with stepper.NodeStepper(self.sess, self.opt) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)
      output = cli.cont(["opt/update_b/ApplyGradientDescent", "-i"])

      output = cli.list_sorted_nodes([])
      node_names, stat_labels, _ = _parse_sorted_nodes_list(output.lines)
      self.assertIn(stepper_cli.NodeStepperCLI.STATE_DIRTY_VARIABLE,
                    stat_labels[node_names.index("b")])
      self.assertNotIn(stepper_cli.NodeStepperCLI.STATE_DIRTY_VARIABLE,
                       stat_labels[node_names.index("a")])

  def testContToUpdateNodeWithoutTrackingLeadsToNoDirtyVariableLabel(self):
    with stepper.NodeStepper(self.sess, self.opt) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)
      output = cli.cont(["opt/update_b/ApplyGradientDescent"])

      self.assertItemsEqual([self.b.name], _parse_updated(output.lines))

      output = cli.list_sorted_nodes([])
      node_names, stat_labels, _ = _parse_sorted_nodes_list(output.lines)
      self.assertIn(stepper_cli.NodeStepperCLI.STATE_DIRTY_VARIABLE,
                    stat_labels[node_names.index("b")])
      self.assertNotIn(stepper_cli.NodeStepperCLI.STATE_DIRTY_VARIABLE,
                       stat_labels[node_names.index("a")])

  def testContWithRestoreVariablesOptionShouldRestoreVariableValue(self):
    with stepper.NodeStepper(self.sess, self.opt) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)
      output = cli.cont(["opt/update_a/ApplyGradientDescent",
                         "--invalidate_from_updated_variables"])

      self.assertItemsEqual([self.a.name], _parse_updated(output.lines))

      # After cont() call on .../update_a/..., Variable a should have been
      # marked as dirty, whereas b should not have.
      output = cli.list_sorted_nodes([])
      node_names, stat_labels, _ = _parse_sorted_nodes_list(output.lines)
      self.assertIn(stepper_cli.NodeStepperCLI.STATE_DIRTY_VARIABLE,
                    stat_labels[node_names.index("a")])
      self.assertNotIn(stepper_cli.NodeStepperCLI.STATE_DIRTY_VARIABLE,
                       stat_labels[node_names.index("b")])

      output = cli.cont(["opt/update_b/ApplyGradientDescent", "-r", "-i"])

      self.assertItemsEqual([self.b.name], _parse_updated(output.lines))

      # After cont() call on .../update_b/... with the -r flag, Variable b
      # should have been marked as dirty, whereas Variable a should not be
      # because it should have been restored.
      output = cli.list_sorted_nodes([])
      node_names, stat_labels, _ = _parse_sorted_nodes_list(output.lines)
      self.assertIn(stepper_cli.NodeStepperCLI.STATE_DIRTY_VARIABLE,
                    stat_labels[node_names.index("b")])
      self.assertNotIn(stepper_cli.NodeStepperCLI.STATE_DIRTY_VARIABLE,
                       stat_labels[node_names.index("a")])

  def testPrintTensorShouldWorkWithTensorName(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      cli.cont("d")
      output = cli.print_tensor(["d:0"])

      self.assertEqual("Tensor \"d:0\":", output.lines[0])
      self.assertEqual("-20.0", output.lines[-1])

  def testPrintTensorShouldWorkWithNodeNameWithOutputTensor(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      cli.cont("d")
      output = cli.print_tensor(["d"])

      self.assertEqual("Tensor \"d:0\":", output.lines[0])
      self.assertEqual("-20.0", output.lines[-1])

  def testPrintTensorShouldWorkSlicingString(self):
    ph_value = np.array([[1.0, 0.0], [0.0, 2.0]])
    with stepper.NodeStepper(
        self.sess, self.f, feed_dict={self.ph: ph_value}) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.print_tensor(["ph:0[:, 1]"])
      self.assertEqual("Tensor \"ph:0[:, 1]\":", output.lines[0])
      self.assertEqual(repr(ph_value[:, 1]), output.lines[-1])

      output = cli.print_tensor(["ph[:, 1]"])
      self.assertEqual("Tensor \"ph:0[:, 1]\":", output.lines[0])
      self.assertEqual(repr(ph_value[:, 1]), output.lines[-1])

  def testPrintTensorWithNonexistentTensorShouldError(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.print_tensor(["foobar"])
      self.assertEqual([
          "ERROR: foobar is not in the transitive closure of this stepper "
          "instance."
      ], output.lines)

  def testPrintTensorWithNoHandleShouldError(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.print_tensor("e")
      self.assertEqual([
          "This stepper instance does not have access to the value of tensor "
          "\"e:0\""
      ], output.lines)

  def testInjectTensorValueByTensorNameShouldBeReflected(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.cont(["d"])
      node_names, _, node_pointer = _parse_sorted_nodes_list(output.lines)
      self.assertEqual("d", node_names[node_pointer])

      output = cli.list_sorted_nodes([])
      node_names, stat_labels, node_pointer = _parse_sorted_nodes_list(
          output.lines)

      index_d = node_names.index("d")
      self.assertIn(stepper_cli.NodeStepperCLI.STATE_CONT, stat_labels[index_d])
      self.assertNotIn(stepper_cli.NodeStepperCLI.STATE_OVERRIDDEN,
                       stat_labels[index_d])

      self.assertAllClose(-20.0, node_stepper.get_tensor_value("d:0"))

      output = cli.inject_value(["d:0", "20.0"])

      # Verify that the override is available.
      self.assertEqual(["d:0"], node_stepper.override_names())

      # Verify that the list of sorted nodes reflects the existence of the value
      # override (i.e., injection).
      output = cli.list_sorted_nodes([])
      node_names, stat_labels, node_pointer = _parse_sorted_nodes_list(
          output.lines)

      index_d = node_names.index("d")
      self.assertNotIn(stepper_cli.NodeStepperCLI.STATE_CONT,
                       stat_labels[index_d])
      self.assertIn(stepper_cli.NodeStepperCLI.STATE_OVERRIDDEN,
                    stat_labels[index_d])

  def testInjectTensorValueByNodeNameShouldBeReflected(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      cli.inject_value(["d", "20.0"])
      self.assertEqual(["d:0"], node_stepper.override_names())

  def testInjectToNonexistentTensorShouldError(self):
    with stepper.NodeStepper(self.sess, self.e) as node_stepper:
      cli = stepper_cli.NodeStepperCLI(node_stepper)

      output = cli.inject_value(["foobar:0", "20.0"])
      self.assertEqual([
          "ERROR: foobar:0 is not in the transitive closure of this stepper "
          "instance."
      ], output.lines)


if __name__ == "__main__":
  googletest.main()
