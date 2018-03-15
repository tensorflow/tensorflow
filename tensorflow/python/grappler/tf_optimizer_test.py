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
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class PyWrapOptimizeGraphTest(test.TestCase):

  def testBasic(self):
    """Make sure arguments can be passed correctly."""
    a = constant_op.constant(10, name='a')
    b = constant_op.constant(20, name='b')
    c = math_ops.add_n([a, b], name='c')
    d = math_ops.add_n([b, c], name='d')
    train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    # Being a train_op will make 'd' to be added as a fetch node.
    train_op.append(d)
    mg = meta_graph.create_meta_graph_def(graph=ops.get_default_graph())

    rewriter_config = rewriter_config_pb2.RewriterConfig()
    rewriter_config.optimizers.append('constfold')

    graph = tf_optimizer.OptimizeGraph(rewriter_config, mg)

    self.assertEqual(len(graph.node), 1)
    self.assertItemsEqual([node.name for node in graph.node], ['d'])

  def testKeepNodes(self):
    g = ops.Graph()
    with g.as_default():
      a1 = variables.Variable(
          1.0)  # Must be preserved since it's in the collection 'variables'.
      a2 = constant_op.constant(0, shape=[50, 50], name='keep')
      ops.add_to_collection('a2', a2)  # Explicitly add to collection.
      b = constant_op.constant(1, shape=[100, 10])
      c = constant_op.constant(0, shape=[10, 30])
      d = math_ops.matmul(b, c)
      ops.add_to_collection('train_op', d)  # d is the fetch node.

    # Optimize the graph.
    mg = meta_graph.create_meta_graph_def(graph=g)
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, mg)

    # Check that the nodes referenced in various collections have been preserved
    self.assertEqual(len(optimized_graph.node), 5)
    self.assertEqual(d.op.name, optimized_graph.node[0].name)
    self.assertEqual(a1.op.name, optimized_graph.node[1].name)
    self.assertEqual('Variable/initial_value', optimized_graph.node[2].name)
    self.assertEqual(a2.op.name, optimized_graph.node[3].name)
    self.assertEqual('Variable/Assign', optimized_graph.node[4].name)


if __name__ == '__main__':
  test.main()
