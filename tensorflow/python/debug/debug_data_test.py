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

import os
import shutil
import tempfile

from tensorflow.python.debug import debug_data
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class NodeNameChecksTest(test_util.TensorFlowTestCase):

  def testIsCopyNode(self):
    self.assertTrue(debug_data.is_copy_node("__copy_ns1/ns2/node3_0"))

    self.assertFalse(debug_data.is_copy_node("copy_ns1/ns2/node3_0"))
    self.assertFalse(debug_data.is_copy_node("_copy_ns1/ns2/node3_0"))
    self.assertFalse(debug_data.is_copy_node("_copyns1/ns2/node3_0"))
    self.assertFalse(debug_data.is_copy_node("__dbg_ns1/ns2/node3_0"))

  def testIsDebugNode(self):
    self.assertTrue(
        debug_data.is_debug_node("__dbg_ns1/ns2/node3:0_0_DebugIdentity"))

    self.assertFalse(
        debug_data.is_debug_node("dbg_ns1/ns2/node3:0_0_DebugIdentity"))
    self.assertFalse(
        debug_data.is_debug_node("_dbg_ns1/ns2/node3:0_0_DebugIdentity"))
    self.assertFalse(
        debug_data.is_debug_node("_dbgns1/ns2/node3:0_0_DebugIdentity"))
    self.assertFalse(debug_data.is_debug_node("__copy_ns1/ns2/node3_0"))


class ParseDebugNodeNameTest(test_util.TensorFlowTestCase):

  def testParseDebugNodeName_valid(self):
    debug_node_name_1 = "__dbg_ns_a/ns_b/node_c:1_0_DebugIdentity"
    (watched_node, watched_output_slot, debug_op_index,
     debug_op) = debug_data.parse_debug_node_name(debug_node_name_1)

    self.assertEqual("ns_a/ns_b/node_c", watched_node)
    self.assertEqual(1, watched_output_slot)
    self.assertEqual(0, debug_op_index)
    self.assertEqual("DebugIdentity", debug_op)

  def testParseDebugNodeName_invalidPrefix(self):
    invalid_debug_node_name_1 = "__copy_ns_a/ns_b/node_c:1_0_DebugIdentity"

    with self.assertRaisesRegexp(ValueError, "Invalid prefix"):
      debug_data.parse_debug_node_name(invalid_debug_node_name_1)

  def testParseDebugNodeName_missingDebugOpIndex(self):
    invalid_debug_node_name_1 = "__dbg_node1:0_DebugIdentity"

    with self.assertRaisesRegexp(ValueError, "Invalid debug node name"):
      debug_data.parse_debug_node_name(invalid_debug_node_name_1)

  def testParseDebugNodeName_invalidWatchedTensorName(self):
    invalid_debug_node_name_1 = "__dbg_node1_0_DebugIdentity"

    with self.assertRaisesRegexp(ValueError,
                                 "Invalid tensor name in debug node name"):
      debug_data.parse_debug_node_name(invalid_debug_node_name_1)


class DebugTensorDatumTest(test_util.TensorFlowTestCase):

  def testDebugDatum(self):
    dump_root = "/tmp/tfdbg_1"
    debug_dump_rel_path = "ns1/ns2/node_a_1_2_DebugIdentity_1472563253536385"

    datum = debug_data.DebugTensorDatum(dump_root, debug_dump_rel_path)

    self.assertEqual("DebugIdentity", datum.debug_op)
    self.assertEqual("ns1/ns2/node_a_1", datum.node_name)
    self.assertEqual(2, datum.output_slot)
    self.assertEqual("ns1/ns2/node_a_1:2", datum.tensor_name)
    self.assertEqual(1472563253536385, datum.timestamp)
    self.assertEqual("ns1/ns2/node_a_1:2:DebugIdentity", datum.watch_key)
    self.assertEqual(
        os.path.join(dump_root, debug_dump_rel_path), datum.file_path)
    self.assertEqual("{DebugTensorDatum: %s:%d @ %s @ %d}" % (datum.node_name,
                                                              datum.output_slot,
                                                              datum.debug_op,
                                                              datum.timestamp),
                     str(datum))
    self.assertEqual("{DebugTensorDatum: %s:%d @ %s @ %d}" % (datum.node_name,
                                                              datum.output_slot,
                                                              datum.debug_op,
                                                              datum.timestamp),
                     repr(datum))


class DebugDumpDirTest(test_util.TensorFlowTestCase):

  def setUp(self):
    self._dump_root = tempfile.mktemp()
    os.mkdir(self._dump_root)

  def tearDown(self):
    # Tear down temporary dump directory.
    shutil.rmtree(self._dump_root)

  def testDebugDumpDir_nonexistentDumpRoot(self):
    with self.assertRaisesRegexp(IOError, "does not exist"):
      debug_data.DebugDumpDir(tempfile.mktemp() + "_foo")

  def testDebugDumpDir_invalidFileNamingPattern(self):
    # File name with too few underscores should lead to an exception.
    open(os.path.join(self._dump_root, "node1_DebugIdentity_1234"), "wb")

    with self.assertRaisesRegexp(ValueError,
                                 "does not conform to the naming pattern"):
      debug_data.DebugDumpDir(self._dump_root)

  def testDebugDumpDir_emptyDumpDir(self):
    dump_dir = debug_data.DebugDumpDir(self._dump_root)

    self.assertIsNone(dump_dir.t0)
    self.assertEqual([], dump_dir.dumped_tensor_data)


if __name__ == "__main__":
  googletest.main()
