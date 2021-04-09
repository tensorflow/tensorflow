# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.embedding_ops."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class EmbeddingLookupTest(test_util.TensorFlowTestCase):

  def testEmbeddingLookupOnUninitializedVariableDoesSparseRead(self):
    x = resource_variable_ops.UninitializedVariable(
        trainable=True, shape=[3, 3], dtype=dtypes.float32)

    @def_function.function(input_signature=[])
    def _init():
      return x.assign(np.zeros([3, 3]))

    @def_function.function(input_signature=[])
    def _call():
      return embedding_ops.embedding_lookup_v2(x, [0])

    self.assertAllClose(self.evaluate(_init()), np.zeros([3, 3]))

    concrete_call = _call.get_concrete_function()
    self.assertAllClose(self.evaluate(concrete_call()), [[0., 0., 0.]])

    resource_gather_node = []
    read_var_node = []
    graph = concrete_call.graph.as_graph_def()
    for n in graph.node:
      if n.op == "ResourceGather":
        resource_gather_node.append(n)
      if n.op == "ReadVariableOp":
        read_var_node.append(n)

    for f in graph.library.function:
      for n in f.node_def:
        if n.op == "ResourceGather":
          resource_gather_node.append(n)
        if n.op == "ReadVariableOp":
          read_var_node.append(n)
    # There should be a single ResourceGather, but no ReadVariableOp
    # (dense read).
    self.assertLen(resource_gather_node, 1)
    self.assertLen(read_var_node, 0)


if __name__ == "__main__":
  googletest.main()
