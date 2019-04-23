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
"""Tests for Graph Transform Tool Python wrapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import simple_save
from tensorflow.python.training.tracking import util
from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.tools.graph_transforms import TransformSavedModel


def _gen_graph_def_with_unused_relu():
  """
  Constructs a graph with a relu op that's not used by the normal
  inference path. The strip_unused transform should remove the relu op
  while retaining the rest of the graph.

  The input node of the resulting graph will be called "input_op".
  The output node of the resulting graph will be called "add_op".
  """
  ret = graph_pb2.GraphDef()

  const_op1 = ret.node.add()
  const_op1.op = "Const"
  const_op1.name = "const_op"
  const_op1.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(
      type=dtypes.float32.as_datatype_enum))
  const_op1.attr["value"].CopyFrom(
      attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
          [1, 2], dtypes.float32, [1, 2])))

  input_op = ret.node.add()
  input_op.op = "Placeholder"
  input_op.name = "input_op"
  input_op.attr["dtype"].CopyFrom(attr_value_pb2.AttrValue(
      type=dtypes.float32.as_datatype_enum))
  input_op.attr["shape"].CopyFrom(attr_value_pb2.AttrValue(
      shape=tensor_shape.TensorShape([1, 2]).as_proto()))

  # Create an op that adds a constant to the input placeholder
  add_op = ret.node.add()
  add_op.op = "Add"
  add_op.attr["T"].CopyFrom(attr_value_pb2.AttrValue(
      type=dtypes.float32.as_datatype_enum))
  add_op.name = "add_op"
  add_op.input.extend(["const_op", "input_op"])

  # Create a relu that reads from the add.
  relu_op = ret.node.add()
  relu_op.op = "Relu"
  relu_op.attr["T"].CopyFrom(attr_value_pb2.AttrValue(
      type=dtypes.float32.as_datatype_enum))
  relu_op.name = "relu_op"
  relu_op.input.extend(["add_op"])

  return ret


class TransformGraphTest(test.TestCase):

  # This test constructs a graph with a relu op that's not used by the normal
  # inference path, and then tests that the strip_unused transform removes it as
  # expected.
  def testTransformGraph(self):
    input_graph_def = _gen_graph_def_with_unused_relu()

    # We're specifying that add_op is the final output, and so the relu isn't
    # needed.
    input_names = ["input_op"]
    output_names = ["add_op"]
    transforms = ["strip_unused_nodes"]
    transformed_graph_def = TransformGraph(input_graph_def, input_names,
                                           output_names, transforms)

    # We expect that the relu is no longer present after running the transform.
    for node in transformed_graph_def.node:
      self.assertNotEqual("Relu", node.op)


class TransformSavedModelTest(test.TestCase):

  def testV1SavedModel(self):
    """
    Tests that a basic SavedModel in TensorFlow V1 format gets correctly
    rewritten.
    """
    with context.eager_mode():  # v2 SavedModel APIs only work in eager mode
      tmp_dir = self.get_temp_dir()
      before_dir = tmp_dir + "/before"
      after_dir = tmp_dir + "/after"

      # Save model with V1 APIs
      g = ops.Graph()
      with g.as_default():
        with self.session(graph=g) as sess:
          importer.import_graph_def(_gen_graph_def_with_unused_relu(), name="")
          input_tensor = g.get_tensor_by_name("input_op:0")
          output_tensor = g.get_tensor_by_name("add_op:0")
          simple_save.simple_save(
              sess,
              before_dir,
              inputs={"in": input_tensor},
              outputs={"out": output_tensor}
          )

      TransformSavedModel(before_dir, after_dir,
                          transforms=["strip_unused_nodes"])

      trackable = load.load(after_dir)
      function = trackable.signatures["serving_default"]
      result = function(constant_op.constant([3., 4.],
                                             shape=[1, 2],
                                             dtype=dtypes.float32))
      self.assertEqual(result["out"].numpy().tolist(), [[4., 6.]])
      transformed_graph_def = function.graph.as_graph_def()
      for node in transformed_graph_def.node:
        self.assertNotEqual("Relu", node.op)

  def testV2SavedModel(self):
    """
    Tests that a basic SavedModel in TensorFlow V2 format gets correctly
    rewritten.
    """
    with context.eager_mode():  # v2 SavedModel APIs only work in eager mode
      tmp_dir = self.get_temp_dir()
      before_dir = tmp_dir + "/before"
      after_dir = tmp_dir + "/after"

      class MyTrackable(util.Checkpoint):
        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32),
            tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32)])
        def test_func(self, first_arg, second_arg):
          result1 = first_arg + second_arg
          result2 = first_arg - second_arg
          constant_op.constant(42., name="dead_code")
          return {"sum": result1, "difference": result2}

      save.save(MyTrackable(), before_dir)

      # Verify that the exported function has the unused op for the
      # "strip_unused_nodes" rewrite to strip out.
      # NOTE: If save() gets smarter about tree-shaking, we may need to use a
      # different rewrite over a different graph for this test.
      with open(before_dir + "/saved_model.pb", "rb") as f:
        binary_protobuf = f.read()
        sm = saved_model_pb2.SavedModel.FromString(binary_protobuf)
        self.assertTrue("dead_code" in str(sm))

      trackable_before = load.load(before_dir)
      three = constant_op.constant(3, shape=None, dtype=dtypes.int32)
      five = constant_op.constant(5, shape=None, dtype=dtypes.int32)
      result_before = trackable_before.test_func(five, three)
      self.assertEqual(result_before["sum"].numpy(), 8)
      self.assertEqual(result_before["difference"].numpy(), 2)

      TransformSavedModel(before_dir, after_dir,
                          transforms=["strip_unused_nodes"])

      # Dead code should be gone now.
      with open(after_dir + "/saved_model.pb", "rb") as f:
        binary_protobuf = f.read()
        sm = saved_model_pb2.SavedModel.FromString(binary_protobuf)
        self.assertFalse("dead_code" in str(sm))

      # Function should still work
      trackable_after = load.load(after_dir)
      result_after = trackable_after.signatures["serving_default"](
          first_arg=five, second_arg=three)
      self.assertEqual(result_after["sum"].numpy(), 8)
      self.assertEqual(result_after["difference"].numpy(), 2)


if __name__ == "__main__":
  test.main()
