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
"""Tests for the swig wrapper tf_optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test


class MemoryOptimizerTest(test.TestCase):
  """Tests the Grappler memory optimizer."""

  def testNoSwapping(self):
    """Make sure the graph is preserved when there is nothing to swap."""
    a = constant_op.constant(10, name='a')
    b = constant_op.constant(20, name='b')
    c = math_ops.add_n([a, b], name='c')
    d = math_ops.add_n([b, c], name='d')
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())

    rewriter_config = rewriter_config_pb2.RewriterConfig(
        memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL)
    graph = tf_optimizer.OptimizeGraph(rewriter_config, mg)

    self.assertEqual(len(graph.node), 4)
    self.assertItemsEqual([node.name
                           for node in graph.node], ['a', 'b', 'c', 'd'])

  def testSimpleSwap(self):
    """Check that the swap annotations are followed."""
    a = constant_op.constant(10, name='a')
    b = constant_op.constant(20, name='b')
    c = math_ops.add_n([a, b], name='c')
    d = math_ops.add_n([b, c], name='d')
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    train_op.append(d)

    d.op.node_def.attr['_swap_to_host'].i = 0

    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())

    rewriter_config = rewriter_config_pb2.RewriterConfig(
        memory_optimization=rewriter_config_pb2.RewriterConfig.MANUAL)
    graph = tf_optimizer.OptimizeGraph(rewriter_config, mg)

    self.assertEqual(len(graph.node), 6)
    self.assertItemsEqual([node.name for node in graph.node], [
        'a',
        'b',
        'c',
        'd',
        'swap_in_d_0',
        'swap_out_d_0',
    ])
    for node in graph.node:
      if node.name == 'swap_in_d_0':
        self.assertEqual('swap_out_d_0', node.input[0])
        self.assertEqual('^b', node.input[1])
      elif node.name == 'swap_out_d_0':
        self.assertEqual('b', node.input[0])
      elif node.name == 'd':
        self.assertEqual('swap_in_d_0', node.input[0])
        self.assertEqual('c', node.input[1])


if __name__ == '__main__':
  test.main()
