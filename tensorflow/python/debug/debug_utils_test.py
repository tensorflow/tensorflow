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
"""Tests for TensorFlow Debugger (tfdbg) Utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class DebugUtilsTest(test_util.TensorFlowTestCase):

  @classmethod
  def setUpClass(cls):
    cls._sess = session.Session()
    with cls._sess:
      cls._a_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
      cls._b_init_val = np.array([[2.0], [-1.0]])
      cls._c_val = np.array([[-4.0], [np.nan]])

      cls._a_init = constant_op.constant(
          cls._a_init_val, shape=[2, 2], name="a1_init")
      cls._b_init = constant_op.constant(
          cls._b_init_val, shape=[2, 1], name="b_init")

      cls._a = variables.Variable(cls._a_init, name="a1")
      cls._b = variables.Variable(cls._b_init, name="b")
      cls._c = constant_op.constant(cls._c_val, shape=[2, 1], name="c")

      # Matrix product of a and b.
      cls._p = math_ops.matmul(cls._a, cls._b, name="p1")

      # Sum of two vectors.
      cls._s = math_ops.add(cls._p, cls._c, name="s")

    cls._graph = cls._sess.graph

    # These are all the expected nodes in the graph:
    #   Two variables (a, b), each with four nodes (Variable, init, Assign,
    #       read).
    #   One constant (c).
    #   One add operation and one matmul operation.
    cls._expected_num_nodes = 4 * 2 + 1 + 1 + 1

  def setUp(self):
    self._run_options = config_pb2.RunOptions()

  def _verify_watches(self, watch_opts, expected_output_slot,
                      expected_debug_ops, expected_debug_urls):
    """Verify a list of debug tensor watches.

    This requires all watches in the watch list have exactly the same
    output_slot, debug_ops and debug_urls.

    Args:
      watch_opts: Repeated protobuf field of DebugTensorWatch.
      expected_output_slot: Expected output slot index, as an integer.
      expected_debug_ops: Expected debug ops, as a list of strings.
      expected_debug_urls: Expected debug URLs, as a list of strings.

    Returns:
      List of node names from the list of debug tensor watches.
    """
    node_names = []
    for watch in watch_opts:
      node_names.append(watch.node_name)

      self.assertEqual(expected_output_slot, watch.output_slot)
      self.assertEqual(expected_debug_ops, watch.debug_ops)
      self.assertEqual(expected_debug_urls, watch.debug_urls)

    return node_names

  def testAddDebugTensorWatches_defaultDebugOp(self):
    debug_utils.add_debug_tensor_watch(
        self._run_options, "foo/node_a", 1, debug_urls="file:///tmp/tfdbg_1")
    debug_utils.add_debug_tensor_watch(
        self._run_options, "foo/node_b", 0, debug_urls="file:///tmp/tfdbg_2")

    self.assertEqual(2, len(self._run_options.debug_tensor_watch_opts))

    watch_0 = self._run_options.debug_tensor_watch_opts[0]
    watch_1 = self._run_options.debug_tensor_watch_opts[1]

    self.assertEqual("foo/node_a", watch_0.node_name)
    self.assertEqual(1, watch_0.output_slot)
    self.assertEqual("foo/node_b", watch_1.node_name)
    self.assertEqual(0, watch_1.output_slot)

    # Verify default debug op name.
    self.assertEqual(["DebugIdentity"], watch_0.debug_ops)
    self.assertEqual(["DebugIdentity"], watch_1.debug_ops)

    # Verify debug URLs.
    self.assertEqual(["file:///tmp/tfdbg_1"], watch_0.debug_urls)
    self.assertEqual(["file:///tmp/tfdbg_2"], watch_1.debug_urls)

  def testAddDebugTensorWatches_explicitDebugOp(self):
    debug_utils.add_debug_tensor_watch(
        self._run_options,
        "foo/node_a",
        0,
        debug_ops="DebugNanCount",
        debug_urls="file:///tmp/tfdbg_1")

    self.assertEqual(1, len(self._run_options.debug_tensor_watch_opts))

    watch_0 = self._run_options.debug_tensor_watch_opts[0]

    self.assertEqual("foo/node_a", watch_0.node_name)
    self.assertEqual(0, watch_0.output_slot)

    # Verify default debug op name.
    self.assertEqual(["DebugNanCount"], watch_0.debug_ops)

    # Verify debug URLs.
    self.assertEqual(["file:///tmp/tfdbg_1"], watch_0.debug_urls)

  def testAddDebugTensorWatches_multipleDebugOps(self):
    debug_utils.add_debug_tensor_watch(
        self._run_options,
        "foo/node_a",
        0,
        debug_ops=["DebugNanCount", "DebugIdentity"],
        debug_urls="file:///tmp/tfdbg_1")

    self.assertEqual(1, len(self._run_options.debug_tensor_watch_opts))

    watch_0 = self._run_options.debug_tensor_watch_opts[0]

    self.assertEqual("foo/node_a", watch_0.node_name)
    self.assertEqual(0, watch_0.output_slot)

    # Verify default debug op name.
    self.assertEqual(["DebugNanCount", "DebugIdentity"], watch_0.debug_ops)

    # Verify debug URLs.
    self.assertEqual(["file:///tmp/tfdbg_1"], watch_0.debug_urls)

  def testAddDebugTensorWatches_multipleURLs(self):
    debug_utils.add_debug_tensor_watch(
        self._run_options,
        "foo/node_a",
        0,
        debug_ops="DebugNanCount",
        debug_urls=["file:///tmp/tfdbg_1", "file:///tmp/tfdbg_2"])

    self.assertEqual(1, len(self._run_options.debug_tensor_watch_opts))

    watch_0 = self._run_options.debug_tensor_watch_opts[0]

    self.assertEqual("foo/node_a", watch_0.node_name)
    self.assertEqual(0, watch_0.output_slot)

    # Verify default debug op name.
    self.assertEqual(["DebugNanCount"], watch_0.debug_ops)

    # Verify debug URLs.
    self.assertEqual(["file:///tmp/tfdbg_1", "file:///tmp/tfdbg_2"],
                     watch_0.debug_urls)

  def testWatchGraph_allNodes(self):
    debug_utils.watch_graph(
        self._run_options,
        self._graph,
        debug_ops=["DebugIdentity", "DebugNanCount"],
        debug_urls="file:///tmp/tfdbg_1")

    self.assertEqual(self._expected_num_nodes,
                     len(self._run_options.debug_tensor_watch_opts))

    # Verify that each of the nodes in the graph with output tensors in the
    # graph have debug tensor watch.
    node_names = self._verify_watches(self._run_options.debug_tensor_watch_opts,
                                      0, ["DebugIdentity", "DebugNanCount"],
                                      ["file:///tmp/tfdbg_1"])

    # Verify the node names.
    self.assertTrue("a1_init" in node_names)
    self.assertTrue("a1" in node_names)
    self.assertTrue("a1/Assign" in node_names)
    self.assertTrue("a1/read" in node_names)

    self.assertTrue("b_init" in node_names)
    self.assertTrue("b" in node_names)
    self.assertTrue("b/Assign" in node_names)
    self.assertTrue("b/read" in node_names)

    self.assertTrue("c" in node_names)
    self.assertTrue("p1" in node_names)
    self.assertTrue("s" in node_names)

  def testWatchGraph_nodeNameWhitelist(self):
    debug_utils.watch_graph(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        node_name_regex_whitelist="(a1$|a1_init$|a1/.*|p1$)")

    node_names = self._verify_watches(self._run_options.debug_tensor_watch_opts,
                                      0, ["DebugIdentity"],
                                      ["file:///tmp/tfdbg_1"])
    self.assertEqual(
        sorted(["a1_init", "a1", "a1/Assign", "a1/read", "p1"]),
        sorted(node_names))

  def testWatchGraph_opTypeWhitelist(self):
    debug_utils.watch_graph(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        op_type_regex_whitelist="(Variable|MatMul)")

    node_names = self._verify_watches(self._run_options.debug_tensor_watch_opts,
                                      0, ["DebugIdentity"],
                                      ["file:///tmp/tfdbg_1"])
    self.assertEqual(sorted(["a1", "b", "p1"]), sorted(node_names))

  def testWatchGraph_nodeNameAndOpTypeWhitelists(self):
    debug_utils.watch_graph(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        node_name_regex_whitelist="([a-z]+1$)",
        op_type_regex_whitelist="(MatMul)")

    node_names = self._verify_watches(self._run_options.debug_tensor_watch_opts,
                                      0, ["DebugIdentity"],
                                      ["file:///tmp/tfdbg_1"])
    self.assertEqual(["p1"], node_names)

  def testWatchGraph_nodeNameBlacklist(self):
    debug_utils.watch_graph_with_blacklists(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        node_name_regex_blacklist="(a1$|a1_init$|a1/.*|p1$)")

    node_names = self._verify_watches(self._run_options.debug_tensor_watch_opts,
                                      0, ["DebugIdentity"],
                                      ["file:///tmp/tfdbg_1"])
    self.assertEqual(
        sorted(["b_init", "b", "b/Assign", "b/read", "c", "s"]),
        sorted(node_names))

  def testWatchGraph_opTypeBlacklist(self):
    debug_utils.watch_graph_with_blacklists(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        op_type_regex_blacklist="(Variable|Identity|Assign|Const)")

    node_names = self._verify_watches(self._run_options.debug_tensor_watch_opts,
                                      0, ["DebugIdentity"],
                                      ["file:///tmp/tfdbg_1"])
    self.assertEqual(sorted(["p1", "s"]), sorted(node_names))

  def testWatchGraph_nodeNameAndOpTypeBlacklists(self):
    debug_utils.watch_graph_with_blacklists(
        self._run_options,
        self._graph,
        debug_urls="file:///tmp/tfdbg_1",
        node_name_regex_blacklist="p1$",
        op_type_regex_blacklist="(Variable|Identity|Assign|Const)")

    node_names = self._verify_watches(self._run_options.debug_tensor_watch_opts,
                                      0, ["DebugIdentity"],
                                      ["file:///tmp/tfdbg_1"])
    self.assertEqual(["s"], node_names)


if __name__ == "__main__":
  googletest.main()
