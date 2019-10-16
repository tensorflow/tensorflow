# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for StatSummarizer Python wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import test
from tensorflow.tools.graph_transforms import TransformGraph


class TransformGraphTest(test.TestCase):

  # This test constructs a graph with a relu op that's not used by the normal
  # inference path, and then tests that the strip_unused transform removes it as
  # expected.
  def testTransformGraph(self):
    input_graph_def = graph_pb2.GraphDef()

    const_op1 = input_graph_def.node.add()
    const_op1.op = "Const"
    const_op1.name = "const_op1"
    const_op1.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    const_op1.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            [1, 2], dtypes.float32, [1, 2])))

    const_op2 = input_graph_def.node.add()
    const_op2.op = "Const"
    const_op2.name = "const_op2"
    const_op2.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    const_op2.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            [3, 4], dtypes.float32, [1, 2])))

    # Create an add that has two constants as inputs.
    add_op = input_graph_def.node.add()
    add_op.op = "Add"
    add_op.attr["T"].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    add_op.name = "add_op"
    add_op.input.extend(["const_op1", "const_op2"])

    # Create a relu that reads from the add.
    relu_op = input_graph_def.node.add()
    relu_op.op = "Relu"
    relu_op.attr["T"].CopyFrom(attr_value_pb2.AttrValue(
        type=dtypes.float32.as_datatype_enum))
    relu_op.name = "relu_op"
    relu_op.input.extend(["add_op"])

    # We're specifying that add_op is the final output, and so the relu isn't
    # needed.
    input_names = []
    output_names = ["add_op"]
    transforms = ["strip_unused_nodes"]
    transformed_graph_def = TransformGraph(input_graph_def, input_names,
                                           output_names, transforms)

    # We expect that the relu is no longer present after running the transform.
    for node in transformed_graph_def.node:
      self.assertNotEqual("Relu", node.op)


if __name__ == "__main__":
  test.main()
