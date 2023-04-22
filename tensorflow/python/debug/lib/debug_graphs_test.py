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
"""Tests for tfdbg module debug_data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class ParseNodeOrTensorNameTest(test_util.TensorFlowTestCase):

  def testParseNodeName(self):
    node_name, slot = debug_graphs.parse_node_or_tensor_name(
        "namespace1/node_1")

    self.assertEqual("namespace1/node_1", node_name)
    self.assertIsNone(slot)

  def testParseTensorName(self):
    node_name, slot = debug_graphs.parse_node_or_tensor_name(
        "namespace1/node_2:3")

    self.assertEqual("namespace1/node_2", node_name)
    self.assertEqual(3, slot)


class GetNodeNameAndOutputSlotTest(test_util.TensorFlowTestCase):

  def testParseTensorNameInputWorks(self):
    self.assertEqual("a", debug_graphs.get_node_name("a:0"))
    self.assertEqual(0, debug_graphs.get_output_slot("a:0"))

    self.assertEqual("_b", debug_graphs.get_node_name("_b:1"))
    self.assertEqual(1, debug_graphs.get_output_slot("_b:1"))

  def testParseNodeNameInputWorks(self):
    self.assertEqual("a", debug_graphs.get_node_name("a"))
    self.assertEqual(0, debug_graphs.get_output_slot("a"))


class NodeNameChecksTest(test_util.TensorFlowTestCase):

  def testIsCopyNode(self):
    self.assertTrue(debug_graphs.is_copy_node("__copy_ns1/ns2/node3_0"))

    self.assertFalse(debug_graphs.is_copy_node("copy_ns1/ns2/node3_0"))
    self.assertFalse(debug_graphs.is_copy_node("_copy_ns1/ns2/node3_0"))
    self.assertFalse(debug_graphs.is_copy_node("_copyns1/ns2/node3_0"))
    self.assertFalse(debug_graphs.is_copy_node("__dbg_ns1/ns2/node3_0"))

  def testIsDebugNode(self):
    self.assertTrue(
        debug_graphs.is_debug_node("__dbg_ns1/ns2/node3:0_0_DebugIdentity"))

    self.assertFalse(
        debug_graphs.is_debug_node("dbg_ns1/ns2/node3:0_0_DebugIdentity"))
    self.assertFalse(
        debug_graphs.is_debug_node("_dbg_ns1/ns2/node3:0_0_DebugIdentity"))
    self.assertFalse(
        debug_graphs.is_debug_node("_dbgns1/ns2/node3:0_0_DebugIdentity"))
    self.assertFalse(debug_graphs.is_debug_node("__copy_ns1/ns2/node3_0"))


class ParseDebugNodeNameTest(test_util.TensorFlowTestCase):

  def testParseDebugNodeName_valid(self):
    debug_node_name_1 = "__dbg_ns_a/ns_b/node_c:1_0_DebugIdentity"
    (watched_node, watched_output_slot, debug_op_index,
     debug_op) = debug_graphs.parse_debug_node_name(debug_node_name_1)

    self.assertEqual("ns_a/ns_b/node_c", watched_node)
    self.assertEqual(1, watched_output_slot)
    self.assertEqual(0, debug_op_index)
    self.assertEqual("DebugIdentity", debug_op)

  def testParseDebugNodeName_invalidPrefix(self):
    invalid_debug_node_name_1 = "__copy_ns_a/ns_b/node_c:1_0_DebugIdentity"

    with self.assertRaisesRegex(ValueError, "Invalid prefix"):
      debug_graphs.parse_debug_node_name(invalid_debug_node_name_1)

  def testParseDebugNodeName_missingDebugOpIndex(self):
    invalid_debug_node_name_1 = "__dbg_node1:0_DebugIdentity"

    with self.assertRaisesRegex(ValueError, "Invalid debug node name"):
      debug_graphs.parse_debug_node_name(invalid_debug_node_name_1)

  def testParseDebugNodeName_invalidWatchedTensorName(self):
    invalid_debug_node_name_1 = "__dbg_node1_0_DebugIdentity"

    with self.assertRaisesRegex(ValueError,
                                "Invalid tensor name in debug node name"):
      debug_graphs.parse_debug_node_name(invalid_debug_node_name_1)


if __name__ == "__main__":
  test.main()
