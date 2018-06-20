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
"""Tests for the reconstruction of non-debugger-decorated GraphDefs."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shutil
import tempfile

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_graphs
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class ReconstructNonDebugGraphTest(test_util.TensorFlowTestCase):

  _OP_TYPE_BLACKLIST = (
      "_Send", "_Recv", "_HostSend", "_HostRecv", "_Retval")

  def _no_rewrite_session_config(self):
    rewriter_config = rewriter_config_pb2.RewriterConfig(
        dependency_optimization=rewriter_config_pb2.RewriterConfig.OFF)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    return config_pb2.ConfigProto(graph_options=graph_options)

  def setUp(self):
    super(ReconstructNonDebugGraphTest, self).setUp()
    self._dump_dir = tempfile.mkdtemp()
    self._debug_url = "file://" + self._dump_dir
    ops.reset_default_graph()

  def tearDown(self):
    shutil.rmtree(self._dump_dir)
    super(ReconstructNonDebugGraphTest, self).tearDown()

  def _graphDefWithoutBlacklistedNodes(self, graph_def):
    output_graph_def = graph_pb2.GraphDef()
    for node in graph_def.node:
      if node.op not in self._OP_TYPE_BLACKLIST:
        new_node = output_graph_def.node.add()
        new_node.CopyFrom(node)

        if new_node.op == "Enter":
          # The debugger sets parallel_iterations attribute of while-loop Enter
          # nodes to 1 for debugging.
          for attr_key in new_node.attr:
            if attr_key == "parallel_iterations":
              new_node.attr[attr_key].i = 1
        elif new_node.op == "Switch":
          # We don't check the inputs to Switch ops as their inputs may be
          # Send/Recv nodes.
          del new_node.input[:]

    return output_graph_def

  def _compareOriginalAndReconstructedGraphDefs(self,
                                                sess,
                                                fetches,
                                                feed_dict=None,
                                                expected_output=None):
    run_options = config_pb2.RunOptions(output_partition_graphs=True)
    run_metadata = config_pb2.RunMetadata()
    output = sess.run(fetches, feed_dict=feed_dict, options=run_options,
                      run_metadata=run_metadata)
    if expected_output is not None:
      self.assertAllClose(expected_output, output)
    non_debug_graph_defs = run_metadata.partition_graphs

    debug_utils.watch_graph(
        run_options, sess.graph, debug_urls=self._debug_url)
    run_metadata = config_pb2.RunMetadata()
    output = sess.run(fetches, feed_dict=feed_dict, options=run_options,
                      run_metadata=run_metadata)
    if expected_output is not None:
      self.assertAllClose(expected_output, output)

    dump = debug_data.DebugDumpDir(
        self._dump_dir, partition_graphs=run_metadata.partition_graphs,
        validate=True)
    reconstructed = dump.reconstructed_non_debug_partition_graphs()

    self.assertEqual(len(non_debug_graph_defs), len(reconstructed))
    for i, non_debug_graph_def in enumerate(non_debug_graph_defs):
      device_name = debug_graphs._infer_device_name(non_debug_graph_def)
      test_util.assert_equal_graph_def(
          self._graphDefWithoutBlacklistedNodes(reconstructed[device_name]),
          self._graphDefWithoutBlacklistedNodes(non_debug_graph_def))

      # Test debug_graphs.reconstruct_non_debug_graph_def.
      reconstructed_again = (
          debug_graphs.reconstruct_non_debug_graph_def(
              run_metadata.partition_graphs[i]))
      test_util.assert_equal_graph_def(
          self._graphDefWithoutBlacklistedNodes(reconstructed_again),
          self._graphDefWithoutBlacklistedNodes(non_debug_graph_def))

  def testReconstructSimpleGraph(self):
    with session.Session() as sess:
      u = variables.Variable([12.0], name="u")
      v = variables.Variable([30.0], name="v")
      w = math_ops.add(u, v, name="w")
      sess.run(u.initializer)
      sess.run(v.initializer)

      self._compareOriginalAndReconstructedGraphDefs(
          sess, w, expected_output=[42.0])

  def testReconstructGraphWithControlEdge(self):
    with session.Session() as sess:
      a = variables.Variable(10.0, name="a")
      with ops.control_dependencies([a]):
        b = math_ops.add(a, a, name="b")
      with ops.control_dependencies([a, b]):
        c = math_ops.multiply(b, b, name="c")
      sess.run(a.initializer)

      self._compareOriginalAndReconstructedGraphDefs(
          sess, c, expected_output=400.0)

  def testReonstructGraphWithCond(self):
    with session.Session(config=self._no_rewrite_session_config()) as sess:
      x = variables.Variable(10.0, name="x")
      y = variables.Variable(20.0, name="y")
      cond = control_flow_ops.cond(
          x > y, lambda: math_ops.add(x, 1), lambda: math_ops.add(y, 1))
      sess.run(x.initializer)
      sess.run(y.initializer)

      self._compareOriginalAndReconstructedGraphDefs(
          sess, cond, expected_output=21.0)

  def testReconstructGraphWithWhileLoop(self):
    with session.Session() as sess:
      loop_body = lambda i: math_ops.add(i, 2)
      loop_cond = lambda i: math_ops.less(i, 16)
      i = constant_op.constant(10, name="i")
      loop = control_flow_ops.while_loop(loop_cond, loop_body, [i])

      self._compareOriginalAndReconstructedGraphDefs(sess, loop)

  def testReconstructGraphWithGradients(self):
    with session.Session(config=self._no_rewrite_session_config()) as sess:
      u = variables.Variable(12.0, name="u")
      v = variables.Variable(30.0, name="v")
      x = constant_op.constant(1.1, name="x")
      toy_loss = x * (u - v)
      train_op = gradient_descent.GradientDescentOptimizer(
          learning_rate=0.1).minimize(toy_loss, name="train_op")
      sess.run(u.initializer)
      sess.run(v.initializer)

      self._compareOriginalAndReconstructedGraphDefs(sess, train_op)


if __name__ == "__main__":
  test.main()
