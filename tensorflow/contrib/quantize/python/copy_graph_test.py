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
"""Tests for tensorflow.quantized.mangle.copy_graph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.quantize.python import copy_graph
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class CopyGraphTest(test_util.TensorFlowTestCase):

  def _CompareNodeInGraph(self, node, graph):
    graph_node = graph.get_operation_by_name(node.name)
    self.assertEqual(str(node.node_def), str(graph_node.node_def))

  def testCopyGraph(self):
    graph = ops.Graph()
    with graph.as_default():
      a = constant_op.constant(1.0)
      b = variables.Variable(2.0)
      c = a + b
    graph_copy = copy_graph.CopyGraph(graph)
    # Ensure that the three original nodes are in the new graph.
    # import_meta_graph also adds a saver node to the graph which we don't care
    # about in this specific use case.
    for tensor in [a, b, c]:
      self._CompareNodeInGraph(tensor.op, graph_copy)
    # Test that the graph collections are the same.
    for key in graph.get_all_collection_keys():
      self.assertEqual(
          len(graph.get_collection(key)),
          len(graph_copy.get_collection(key)), 'Collection %s differs.')


if __name__ == '__main__':
  googletest.main()
