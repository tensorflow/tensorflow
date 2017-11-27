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
"""@graph_util tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.framework.python.framework import graph_util
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def GetNewNode(name, op, input_nodes):
  new_node = node_def_pb2.NodeDef()
  new_node.op = op
  new_node.name = name
  for node in input_nodes:
    new_node.input.append(node)
  return new_node


class GraphUtilTest(test.TestCase):

  def testGraphUtil(self):
    graph_def = graph_pb2.GraphDef()
    node_a = GetNewNode('A', 'Placeholder', [])
    node_b = GetNewNode('B', 'Op1', ['A'])
    node_c = GetNewNode('C', 'Op1', ['B'])
    node_d = GetNewNode('D', 'Op1', ['C'])
    node_e = GetNewNode('E', 'Op1', ['D'])
    graph_def.node.extend([node_a, node_b, node_c, node_d, node_e])
    fused_graph_def = graph_util.fuse_op(
        graph_def, ['A'], ['D'], [types_pb2.DT_FLOAT], True, 'FusedOp', 'Op2')
    self.assertEqual(len(fused_graph_def.node), 4)
    self.assertEqual(fused_graph_def.node[0].name, 'A')
    self.assertEqual(fused_graph_def.node[1].name, 'FusedOp')
    self.assertEqual(fused_graph_def.node[1].input[0], 'A')
    self.assertEqual(fused_graph_def.node[1].op, 'Op2')
    self.assertEqual(fused_graph_def.node[1].attr['_output_quantized'].b, True)
    self.assertEqual(fused_graph_def.node[1].attr['_output_types'].list.type,
                     [types_pb2.DT_FLOAT])
    self.assertEqual(fused_graph_def.node[2].name, 'D')
    self.assertEqual(fused_graph_def.node[3].name, 'E')

  def testGraphUtilArtificialDependencyInjection(self):
    graph_def = graph_pb2.GraphDef()
    node_a = GetNewNode('A', 'Placeholder', [])
    node_a1 = GetNewNode('A1', 'Placeholder', [])
    node_b = GetNewNode('B', 'Op1', ['A'])
    node_c = GetNewNode('C', 'Op1', ['B'])
    node_d = GetNewNode('D', 'Op1', ['C'])
    node_e = GetNewNode('E', 'Op1', ['D'])
    graph_def.node.extend([node_a, node_a1, node_b, node_c, node_d, node_e])
    fused_graph_def = graph_util.fuse_op(graph_def, ['A', 'A1'], ['D'],
                                         [types_pb2.DT_FLOAT], True, 'FusedOp',
                                         'Op2')
    self.assertEqual(len(fused_graph_def.node), 5)
    self.assertEqual(fused_graph_def.node[0].name, 'A')
    self.assertEqual(fused_graph_def.node[1].name, 'A1')
    self.assertEqual(fused_graph_def.node[2].name, 'FusedOp')
    self.assertEqual(fused_graph_def.node[2].input[0], 'A')
    self.assertEqual(fused_graph_def.node[2].op, 'Op2')
    self.assertEqual(fused_graph_def.node[2].attr['_output_quantized'].b, True)
    self.assertEqual(fused_graph_def.node[2].attr['_output_types'].list.type,
                     [types_pb2.DT_FLOAT])
    self.assertEqual(fused_graph_def.node[3].name, 'D')
    self.assertEqual(fused_graph_def.node[4].name, 'E')


class GetPlaceholdersTest(test.TestCase):

  def test_get_placeholders(self):
    with ops.Graph().as_default() as g:
      placeholders = [array_ops.placeholder(dtypes.float32) for _ in range(5)]
      results = graph_util.get_placeholders(g)
      self.assertEqual(sorted(placeholders, key=lambda x: x._id),  # pylint: disable=protected-access
                       sorted(results, key=lambda x: x._id))  # pylint: disable=protected-access


if __name__ == '__main__':
  test.main()
