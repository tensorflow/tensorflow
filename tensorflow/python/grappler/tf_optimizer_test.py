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
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.grappler import item as gitem
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
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
    rewriter_config.min_graph_nodes = -1

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
    rewriter_config.min_graph_nodes = -1
    optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, mg)

    # Check that the nodes referenced in various collections have been preserved
    self.assertEqual(len(optimized_graph.node), 5)
    self.assertEqual(d.op.name, optimized_graph.node[0].name)
    self.assertEqual(a1.op.name, optimized_graph.node[1].name)
    self.assertEqual('Variable/initial_value', optimized_graph.node[2].name)
    self.assertEqual(a2.op.name, optimized_graph.node[3].name)
    self.assertEqual('Variable/Assign', optimized_graph.node[4].name)

  def testLoops(self):
    g = ops.Graph()
    with g.as_default():

      def _Cond(_, counter):
        return counter < end

      def _Body(buf, counter):
        buf = array_ops.concat([buf, [counter]], 0)
        counter += 1
        return [buf, counter]

      start = array_ops.placeholder(shape=[], dtype=dtypes.int32)
      end = array_ops.placeholder(shape=[], dtype=dtypes.int32)
      init_buf = array_ops.zeros(shape=[0], dtype=dtypes.int32)
      loop_vars = [init_buf, start]
      shape_inv = [
          tensor_shape.TensorShape([None]),
          tensor_shape.TensorShape([])
      ]
      buf, _ = control_flow_ops.while_loop(_Cond, _Body, loop_vars, shape_inv)

      f = -array_ops.ones_like(buf, optimize=False)
      buf_shape = array_ops.shape(buf)
      f_shape = array_ops.shape(f)
      ops.add_to_collection('train_op', buf_shape)
      ops.add_to_collection('train_op', f_shape)

    # Optimize the graph.
    mg = meta_graph.create_meta_graph_def(graph=g)
    rewriter_config = rewriter_config_pb2.RewriterConfig()
    rewriter_config.min_graph_nodes = -1
    optimized_graph = tf_optimizer.OptimizeGraph(rewriter_config, mg)
    mg.graph_def.CopyFrom(optimized_graph)

    # Check that the nodes referenced in various collections have been preserved
    item = gitem.Item(mg)
    props = item.GetOpProperties()
    buf_prop = props[buf.op.name]
    f_prop = props[f.op.name]
    self.assertEqual(buf_prop, f_prop)


if __name__ == '__main__':
  test.main()
