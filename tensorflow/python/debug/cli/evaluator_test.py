# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for arbitrary expression evaluator."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.debug.cli import evaluator
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class ParseDebugTensorNameTest(test_util.TensorFlowTestCase):

  def testParseNamesWithoutPrefixOrSuffix(self):
    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name("foo:1"))
    self.assertIsNone(device_name)
    self.assertEqual("foo", node_name)
    self.assertEqual(1, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(0, exec_index)

    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name("hidden_0/Weights:0"))
    self.assertIsNone(device_name)
    self.assertEqual("hidden_0/Weights", node_name)
    self.assertEqual(0, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(0, exec_index)

  def testParseNamesWithoutPrefixWithDebugOpSuffix(self):
    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name("foo:1:DebugNanCount"))
    self.assertIsNone(device_name)
    self.assertEqual("foo", node_name)
    self.assertEqual(1, output_slot)
    self.assertEqual("DebugNanCount", debug_op)
    self.assertEqual(0, exec_index)

    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name(
            "hidden_0/Weights:0:DebugNumericSummary"))
    self.assertIsNone(device_name)
    self.assertEqual("hidden_0/Weights", node_name)
    self.assertEqual(0, output_slot)
    self.assertEqual("DebugNumericSummary", debug_op)
    self.assertEqual(0, exec_index)

  def testParseNamesWithDeviceNamePrefixWithoutDebugOpSuffix(self):
    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name(
            "/job:ps/replica:0/task:2/cpu:0:foo:1"))
    self.assertEqual("/job:ps/replica:0/task:2/cpu:0", device_name)
    self.assertEqual("foo", node_name)
    self.assertEqual(1, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(0, exec_index)

    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name(
            "/job:worker/replica:0/task:3/gpu:0:hidden_0/Weights:0"))
    self.assertEqual("/job:worker/replica:0/task:3/gpu:0", device_name)
    self.assertEqual("hidden_0/Weights", node_name)
    self.assertEqual(0, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(0, exec_index)

  def testParseNamesWithDeviceNamePrefixWithDebugOpSuffix(self):
    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name(
            "/job:ps/replica:0/task:2/cpu:0:foo:1:DebugNanCount"))
    self.assertEqual("/job:ps/replica:0/task:2/cpu:0", device_name)
    self.assertEqual("foo", node_name)
    self.assertEqual(1, output_slot)
    self.assertEqual("DebugNanCount", debug_op)
    self.assertEqual(0, exec_index)

    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name(
            "/job:worker/replica:0/task:3/gpu:0:"
            "hidden_0/Weights:0:DebugNumericSummary"))
    self.assertEqual("/job:worker/replica:0/task:3/gpu:0", device_name)
    self.assertEqual("hidden_0/Weights", node_name)
    self.assertEqual(0, output_slot)
    self.assertEqual("DebugNumericSummary", debug_op)
    self.assertEqual(0, exec_index)

  def testParseMalformedDebugTensorName(self):
    with self.assertRaisesRegex(
        ValueError,
        r"The debug tensor name in the to-be-evaluated expression is "
        r"malformed:"):
      evaluator._parse_debug_tensor_name(
          "/job:ps/replica:0/task:2/cpu:0:foo:1:DebugNanCount:1337")

    with self.assertRaisesRegex(
        ValueError,
        r"The debug tensor name in the to-be-evaluated expression is "
        r"malformed:"):
      evaluator._parse_debug_tensor_name(
          "/job:ps/replica:0/cpu:0:foo:1:DebugNanCount")

    with self.assertRaises(ValueError):
      evaluator._parse_debug_tensor_name(
          "foo:1:DebugNanCount[]")

    with self.assertRaises(ValueError):
      evaluator._parse_debug_tensor_name(
          "foo:1[DebugNanCount]")

  def testParseNamesWithExecIndex(self):
    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name("foo:1[20]"))
    self.assertIsNone(device_name)
    self.assertEqual("foo", node_name)
    self.assertEqual(1, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(20, exec_index)

    device_name, node_name, output_slot, debug_op, exec_index = (
        evaluator._parse_debug_tensor_name("hidden_0/Weights:0[3]"))
    self.assertIsNone(device_name)
    self.assertEqual("hidden_0/Weights", node_name)
    self.assertEqual(0, output_slot)
    self.assertEqual("DebugIdentity", debug_op)
    self.assertEqual(3, exec_index)


class EvaluatorTest(test_util.TensorFlowTestCase):

  def testEvaluateSingleTensor(self):
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      del node_name, output_slot, debug_op, device_name  # Unused.
      return [np.array([[1.0, 2.0, 3.0]])]

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      self.assertEqual(3, ev.evaluate("np.size(`a:0`)"))

      # Whitespace in backticks should be tolerated.
      self.assertEqual(3, ev.evaluate("np.size(` a:0 `)"))

  def testEvaluateTwoTensors(self):
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      del debug_op, device_name  # Unused.
      if node_name == "a" and output_slot == 0:
        return [np.array([[1.0, -2.0], [0.0, 1.0]])]
      elif node_name == "b" and output_slot == 0:
        return [np.array([[-1.0], [1.0]])]

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      self.assertAllClose([[-3.0], [1.0]],
                          ev.evaluate("np.matmul(`a:0`, `b:0`)"))
      self.assertAllClose(
          [[-4.0], [2.0]], ev.evaluate("np.matmul(`a:0`, `b:0`) + `b:0`"))

  def testEvaluateNoneExistentTensorGeneratesError(self):
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      del node_name, output_slot, debug_op, device_name  # Unused.
      raise debug_data.WatchKeyDoesNotExistInDebugDumpDirError()

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      with self.assertRaisesRegex(
          ValueError, "Eval failed due to the value of .* being unavailable"):
        ev.evaluate("np.matmul(`a:0`, `b:0`)")

  def testEvaluateWithMultipleDevicesContainingTheSameTensorName(self):
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      del output_slot, debug_op  # Unused.
      if node_name == "a" and device_name is None:
        raise ValueError(
            "There are multiple (2) devices with nodes named 'a' but "
            "device_name is not specified")
      elif (node_name == "a" and
            device_name == "/job:worker/replica:0/task:0/cpu:0"):
        return [np.array(10.0)]
      elif (node_name == "a" and
            device_name == "/job:worker/replica:0/task:1/cpu:0"):
        return [np.array(20.0)]

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      with self.assertRaisesRegex(ValueError, r"multiple \(2\) devices"):
        ev.evaluate("`a:0` + `a:0`")

      self.assertAllClose(
          30.0,
          ev.evaluate("`/job:worker/replica:0/task:0/cpu:0:a:0` + "
                      "`/job:worker/replica:0/task:1/cpu:0:a:0`"))

  def testEvaluateWithNonDefaultDebugOp(self):
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      del device_name  # Unused.
      if node_name == "a" and output_slot == 0 and debug_op == "DebugIdentity":
        return [np.array([[-1.0], [1.0]])]
      elif node_name == "a" and output_slot == 0 and debug_op == "DebugFoo":
        return [np.array([[-2.0, 2.0]])]

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      self.assertAllClose(
          [[4.0]],
          ev.evaluate("np.matmul(`a:0:DebugFoo`, `a:0:DebugIdentity`)"))

  def testEvaluateWithMultipleExecIndexes(self):
    dump = test.mock.MagicMock()
    def fake_get_tensors(node_name, output_slot, debug_op, device_name=None):
      del debug_op, device_name  # Unused.
      if node_name == "a" and output_slot == 0:
        return [np.array([[-1.0], [1.0]]), np.array([[-2.0], [2.0]])]

    with test.mock.patch.object(
        dump, "get_tensors", side_effect=fake_get_tensors):
      ev = evaluator.ExpressionEvaluator(dump)
      self.assertAllClose(
          [[4.0]], ev.evaluate("np.matmul(`a:0[1]`.T, `a:0[0]`)"))

  def testEvaluateExpressionWithUnmatchedBacktick(self):
    dump = test.mock.MagicMock()
    ev = evaluator.ExpressionEvaluator(dump)
    with self.assertRaises(SyntaxError):
      ev.evaluate("np.matmul(`a:0`, `b:0`) + `b:0")

  def testEvaluateExpressionWithInvalidDebugTensorName(self):
    dump = test.mock.MagicMock()
    ev = evaluator.ExpressionEvaluator(dump)
    with self.assertRaisesRegex(ValueError,
                                r".* tensor name .* expression .* malformed"):
      ev.evaluate("np.matmul(`a`, `b`)")

    with self.assertRaisesRegex(ValueError,
                                r".* tensor name .* expression .* malformed"):
      ev.evaluate("np.matmul(`a:0:DebugIdentity:0`, `b:1:DebugNanCount:2`)")

    with self.assertRaises(ValueError):
      ev.evaluate("np.matmul(`a:0[]`, `b:0[]`)")


if __name__ == "__main__":
  test.main()
