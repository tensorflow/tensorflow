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
"""Tests for tensorflow.ops.remote_fused_graph_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# pylint: disable=unused-import,wildcard-import, line-too-long
from tensorflow.contrib.remote_fused_graph.pylib.python.ops import remote_fused_graph_ops
# pylint: enable=unused-import,wildcard-import,line-too-long

from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class RemoteFusedGraphExecuteTest(test_util.TensorFlowTestCase):
  """Tests for RemoteFusedGraphExecute op."""

  def testBuild(self):
    graph = graph_pb2.GraphDef()
    node = graph.node.add()
    node.name = "a"
    node.op = "op0"
    node = graph.node.add()
    node.name = "b"
    node.op = "op1"
    inputs = [ops.convert_n_to_tensor([1], dtypes.int64)]
    output_types = [np.int64, np.int64]
    graph_input_node_names = ["a"]
    graph_output_node_names = ["a", "b"]
    executor_name = ""
    serialized_executor_parameters = b""
    default_graph_input_tensor_type_shapes = [[dtypes.int64, [1]]]
    default_graph_output_tensor_type_shapes = [[dtypes.int64, [1]],
                                               [dtypes.int64, [1]]]

    output_nodes = remote_fused_graph_ops.remote_fused_graph_execute(
        inputs, output_types, graph, graph_input_node_names,
        graph_output_node_names, executor_name, serialized_executor_parameters,
        default_graph_input_tensor_type_shapes,
        default_graph_output_tensor_type_shapes)
    self.assertEqual(2, len(output_nodes))
    for output_node in output_nodes:
      with self.test_session(use_gpu=False):
        output_node.eval()


if __name__ == "__main__":
  googletest.main()
