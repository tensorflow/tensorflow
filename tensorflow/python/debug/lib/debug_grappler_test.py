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
"""Tests for debugger functionalities in tf.Session."""
import os
import tempfile

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


def _grappler_enabled_session_config():
  """Constructs a Session config proto that explicitly enables Grappler.

  Returns:
    A config proto that obtains extra safety for the unit tests in this
    file by ensuring that the relevant Grappler rewrites are always enabled.
  """
  rewriter_config = rewriter_config_pb2.RewriterConfig(
      disable_model_pruning=False,
      arithmetic_optimization=rewriter_config_pb2.RewriterConfig.ON)
  graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
  return config_pb2.ConfigProto(graph_options=graph_options)


class SessionDebugGrapplerInteractionTest(test_util.TensorFlowTestCase):

  def setUp(self):
    super(SessionDebugGrapplerInteractionTest, self).setUp()
    self._dump_root = tempfile.mkdtemp()
    self._debug_url = "file://%s" % self._dump_root

  def tearDown(self):
    ops.reset_default_graph()
    if os.path.isdir(self._dump_root):
      file_io.delete_recursively(self._dump_root)
    super(SessionDebugGrapplerInteractionTest, self).tearDown()

  def testArithmeticOptimizationActive(self):
    """Tests that tfdbg can dump the tensor from nodes created by Grappler."""
    with session.Session(config=_grappler_enabled_session_config()) as sess:
      u = variables.VariableV1([[1, 2], [3, 4]], name="u", dtype=dtypes.float32)
      # The next two ops should be optimized by Grappler into a single op:
      # either an AddN op or a Mul op.
      x = math_ops.add(u, u)
      x = math_ops.add(x, u)
      y = math_ops.multiply(x, u)

      sess.run(variables.global_variables_initializer())

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      debug_utils.watch_graph(
          run_options,
          sess.graph,
          debug_ops=["DebugIdentity"],
          debug_urls=[self._debug_url])

      run_metadata = config_pb2.RunMetadata()
      run_result = sess.run(y, options=run_options, run_metadata=run_metadata)
      self.assertAllClose(run_result, [[3, 12], [27, 48]])

      dump_data = debug_data.DebugDumpDir(
          self._dump_root, partition_graphs=run_metadata.partition_graphs,
          validate=True)

      original_node_names = set(op.name for op in sess.graph.get_operations())
      dumped_node_names = set(dump_data.nodes())
      grappler_created_node_names = dumped_node_names - original_node_names
      grappler_removed_node_names = original_node_names - dumped_node_names

      # Assert that Grappler should have replaced some of the nodes from the
      # original graph with new nodes.
      self.assertTrue(grappler_created_node_names)
      self.assertTrue(grappler_removed_node_names)

      # Iterate through the nodes created by Grappler. One of them should be
      # be the result of replacing the original add ops with an AddN op or a
      # Mul op.
      found_optimized_node = False
      for grappler_node_name in grappler_created_node_names:
        node_op_type = dump_data.node_op_type(grappler_node_name)
        # Look for the node created by Grappler's arithmetic optimization.
        if ((test_util.IsMklEnabled() and node_op_type in ("_MklAddN", "Mul"))
            or (node_op_type in ("AddN", "Mul"))):
          datum = dump_data.get_tensors(grappler_node_name, 0, "DebugIdentity")
          self.assertEqual(1, len(datum))
          self.assertAllClose(datum[0], [[3, 6], [9, 12]])
          found_optimized_node = True
          break
      self.assertTrue(
          found_optimized_node,
          "Failed to find optimized node created by Grappler's arithmetic "
          "optimization.")


if __name__ == "__main__":
  googletest.main()
