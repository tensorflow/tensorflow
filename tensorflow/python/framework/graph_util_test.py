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
"""Tests for tensorflow.python.client.graph_util."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops as math_ops_lib
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.saver import export_meta_graph


# Utility device function to use for testing
def test_device_func_pin_variable_to_cpu(op):
  if op.device:
    return op.device
  return "/cpu:0" if op.node_def.op in ["Variable", "VariableV2"] else op.device


class DeviceFunctionsTest(test.TestCase):

  def testTwoDeviceFunctions(self):
    with ops.Graph().as_default() as g:
      var_0 = gen_state_ops.variable(
          shape=[1],
          dtype=dtypes.float32,
          name="var_0",
          container="",
          shared_name="")
      with g.device(test_device_func_pin_variable_to_cpu):
        var_1 = gen_state_ops.variable(
            shape=[1],
            dtype=dtypes.float32,
            name="var_1",
            container="",
            shared_name="")
      var_2 = gen_state_ops.variable(
          shape=[1],
          dtype=dtypes.float32,
          name="var_2",
          container="",
          shared_name="")
      var_3 = gen_state_ops.variable(
          shape=[1],
          dtype=dtypes.float32,
          name="var_3",
          container="",
          shared_name="")
      with g.device(test_device_func_pin_variable_to_cpu):
        var_4 = gen_state_ops.variable(
            shape=[1],
            dtype=dtypes.float32,
            name="var_4",
            container="",
            shared_name="")
        with g.device("/device:GPU:0"):
          var_5 = gen_state_ops.variable(
              shape=[1],
              dtype=dtypes.float32,
              name="var_5",
              container="",
              shared_name="")
        var_6 = gen_state_ops.variable(
            shape=[1],
            dtype=dtypes.float32,
            name="var_6",
            container="",
            shared_name="")

    self.assertDeviceEqual(var_0.device, None)
    self.assertDeviceEqual(var_1.device, "/device:CPU:0")
    self.assertDeviceEqual(var_2.device, None)
    self.assertDeviceEqual(var_3.device, None)
    self.assertDeviceEqual(var_4.device, "/device:CPU:0")
    self.assertDeviceEqual(var_5.device, "/device:GPU:0")
    self.assertDeviceEqual(var_6.device, "/device:CPU:0")

  @test_util.run_v1_only("b/120545219")
  def testNestedDeviceFunctions(self):
    with ops.Graph().as_default():
      var_0 = variables.VariableV1(0)
      with ops.device(test_device_func_pin_variable_to_cpu):
        var_1 = variables.VariableV1(1)
        with ops.device(lambda op: "/device:GPU:0"):
          var_2 = variables.VariableV1(2)
        with ops.device("/device:GPU:0"):  # Implicit merging device function.
          var_3 = variables.VariableV1(3)

    self.assertDeviceEqual(var_0.device, None)
    self.assertDeviceEqual(var_1.device, "/device:CPU:0")
    self.assertDeviceEqual(var_2.device, "/device:GPU:0")
    self.assertDeviceEqual(var_3.device, "/device:GPU:0")

  def testExplicitDevice(self):
    with ops.Graph().as_default() as g:
      const_0 = constant_op.constant(5.0)
      with g.device("/device:GPU:0"):
        const_1 = constant_op.constant(5.0)
      with g.device("/device:GPU:1"):
        const_2 = constant_op.constant(5.0)
      with g.device("/device:CPU:0"):
        const_3 = constant_op.constant(5.0)
      with g.device("/device:CPU:1"):
        const_4 = constant_op.constant(5.0)
      with g.device("/job:ps"):
        const_5 = constant_op.constant(5.0)

    self.assertDeviceEqual(const_0.device, None)
    self.assertDeviceEqual(const_1.device, "/device:GPU:0")
    self.assertDeviceEqual(const_2.device, "/device:GPU:1")
    self.assertDeviceEqual(const_3.device, "/device:CPU:0")
    self.assertDeviceEqual(const_4.device, "/device:CPU:1")
    self.assertDeviceEqual(const_5.device, "/job:ps")

  def testDefaultDevice(self):
    with ops.Graph().as_default() as g, g.device(
        test_device_func_pin_variable_to_cpu):
      with g.device("/job:ps"):
        const_0 = constant_op.constant(5.0)
      with g.device("/device:GPU:0"):
        const_1 = constant_op.constant(5.0)
      with g.device("/device:GPU:1"):
        const_2 = constant_op.constant(5.0)
      with g.device("/device:CPU:0"):
        const_3 = constant_op.constant(5.0)
      with g.device("/device:CPU:1"):
        const_4 = constant_op.constant(5.0)
      with g.device("/replica:0"):
        const_5 = constant_op.constant(5.0)

    self.assertDeviceEqual(const_0.device, "/job:ps")
    self.assertDeviceEqual(const_1.device, "/device:GPU:0")
    self.assertDeviceEqual(const_2.device, "/device:GPU:1")
    self.assertDeviceEqual(const_3.device, "/device:CPU:0")
    self.assertDeviceEqual(const_4.device, "/device:CPU:1")
    self.assertDeviceEqual(const_5.device, "/replica:0")

  def testExtractSubGraph(self):
    graph_def = graph_pb2.GraphDef()
    n1 = graph_def.node.add()
    n1.name = "n1"
    n1.input.extend(["n5"])
    n2 = graph_def.node.add()
    n2.name = "n2"
    # Take the first output of the n1 node as the input.
    n2.input.extend(["n1:0"])
    n3 = graph_def.node.add()
    n3.name = "n3"
    # Add a control input (which isn't really needed by the kernel, but
    # rather to enforce execution order between nodes).
    n3.input.extend(["^n2"])
    n4 = graph_def.node.add()
    n4.name = "n4"

    # It is fine to have a loops in the graph as well.
    n5 = graph_def.node.add()
    n5.name = "n5"
    n5.input.extend(["n1"])

    sub_graph = graph_util.extract_sub_graph(graph_def, ["n3"])
    self.assertEqual("n1", sub_graph.node[0].name)
    self.assertEqual("n2", sub_graph.node[1].name)
    self.assertEqual("n3", sub_graph.node[2].name)
    self.assertEqual("n5", sub_graph.node[3].name)

  def testExtractSubGraphWithInvalidDestNodes(self):
    graph_def = graph_pb2.GraphDef()
    n1 = graph_def.node.add()
    n1.name = "n1"
    with self.assertRaisesRegexp(TypeError, "must be a list"):
      graph_util.extract_sub_graph(graph_def, "n1")

  def _test_convert_variables_with_functions(self, inline_functions):
    """Freezes a graph with functions."""

    @function.Defun(dtypes.float32)
    def plus_one(x):
      return x + 1.0

    with ops.Graph().as_default():
      variable_node = variables.Variable(1.0, name="variable_node")
      _ = variables.Variable(1.0, name="unused_variable_node")
      defun_node = plus_one(variable_node)
      _ = math_ops_lib.multiply(defun_node, 2.0, name="output_node")

      with session.Session() as sess:
        self.evaluate(variables.variables_initializer([variable_node]))
        variable_graph_def = sess.graph.as_graph_def()

        if inline_functions:
          # Run Grappler to create the VarOpHandle --> Placeholder -->
          # ResourceVariable pattern.
          meta_graph = export_meta_graph(graph_def=variable_graph_def)
          fetch_collection = meta_graph_pb2.CollectionDef()
          for name in ["variable_node", "output_node"]:
            fetch_collection.node_list.value.append(name)
          meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

          # Initialize RewriterConfig with everything disabled except function
          # inlining.
          config = config_pb2.ConfigProto()
          rewrite_options = config.graph_options.rewrite_options
          rewrite_options.optimizers.append("function")
          variable_graph_def = tf_optimizer.OptimizeGraph(config, meta_graph)

        constant_graph_def = graph_util.convert_variables_to_constants(
            sess, variable_graph_def, ["output_node"])

    # Ensure there are no variables after freezing.
    for node in constant_graph_def.node:
      self.assertNotIn(
          node.op, ["Variable", "VariableV2", "VarHandleOp", "ReadVariableOp"])

  def testConvertVariablesToConstsWithFunctions(self):
    """Freezes a graph with functions."""
    self._test_convert_variables_with_functions(inline_functions=False)

  def testConvertVariableToConstsWithFunctionsInlined(self):
    """Freezes a graph with functions that have been inlined using Grappler."""
    self._test_convert_variables_with_functions(inline_functions=True)

  def _get_tensors(self, sess, tensor_list):
    """Returns a list of Tensor objects from the Session."""
    return [
        sess.graph.get_tensor_by_name(tensor.name) for tensor in tensor_list
    ]

  def _evaluate_graph_def(self, graph_def, inputs, outputs, input_data):
    """Evaluates the GraphDef using Sessions."""
    with ops.Graph().as_default() as graph:
      importer.import_graph_def(graph_def, name="")
      sess = session.Session(graph=graph)

    input_tensors = self._get_tensors(sess, inputs)
    output_tensors = self._get_tensors(sess, outputs)
    return sess.run(
        output_tensors, feed_dict=dict(zip(input_tensors, input_data)))

  @test_util.run_v1_only("Incompatible with TF 2.0")
  def testConvertVariablesToConstsWithEmbeddings(self):
    """Freezes a graph with embeddings."""
    input_data = np.array(np.random.random_sample([1, 1]), dtype=np.int32)

    # Make model.
    state_input = keras.layers.Input(
        shape=(1,), name="state_input", dtype="int32")
    output = keras.layers.Embedding(
        output_dim=16, input_dim=100, input_length=1, name="state")(
            state_input)
    model = keras.models.Model(inputs=[state_input], outputs=[output])
    model.compile(
        loss={"state": "sparse_categorical_crossentropy"}, optimizer="adam")

    # Get associated session.
    sess = keras.backend.get_session()
    variable_graph_def = sess.graph_def
    output_tensor = [tensor.name.split(":")[0] for tensor in model.outputs]
    constant_graph_def = graph_util.convert_variables_to_constants(
        sess, variable_graph_def, output_tensor)

    # Ensure graph has no variables.
    for node in constant_graph_def.node:
      self.assertNotIn(
          node.op, ["Variable", "VariableV2", "VarHandleOp", "ReadVariableOp"])

    # Compare the value of the graphs.
    expected_value = model.predict(input_data)
    actual_value = self._evaluate_graph_def(constant_graph_def, model.inputs,
                                            model.outputs, [input_data])
    np.testing.assert_almost_equal(np.array([expected_value]), actual_value, 5)

  def testConvertVariablesToConsts(self):
    self._test_variable_to_const_conversion(use_resource=False)

  def testConvertResourceVariablesToConsts(self):
    self._test_variable_to_const_conversion(use_resource=True)

  def _test_variable_to_const_conversion(self, use_resource):
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=use_resource):
        variable_node = variable_scope.get_variable(
            "variable_node", initializer=1.0)
        another_variable = variable_scope.get_variable(
            "unused_variable_node", initializer=1.0)
        output_node = math_ops_lib.multiply(
            variable_node, 2.0, name="output_node")
        with session.Session() as sess:
          self.evaluate(variable_node.initializer)
          output = self.evaluate(output_node)
          self.assertNear(2.0, output, 0.00001)
          variable_graph_def = sess.graph.as_graph_def()
          # First get the constant_graph_def when variable_names_whitelist is
          # set, note that if variable_names_whitelist is not set an error will
          # be thrown because unused_variable_node is not initialized.
          constant_graph_def = graph_util.convert_variables_to_constants(
              sess,
              variable_graph_def, ["output_node"],
              variable_names_whitelist=set(["variable_node"]))

          # Then initialize the unused variable, and get another
          # constant_graph_def when variable_names_whitelist is not set.
          self.evaluate(another_variable.initializer)
          constant_graph_def_without_variable_whitelist = (
              graph_util.convert_variables_to_constants(
                  sess, variable_graph_def, ["output_node"]))

          # The unused variable should be cleared so the two graphs should be
          # equivalent.
          self.assertEqual(
              str(constant_graph_def),
              str(constant_graph_def_without_variable_whitelist))

          # Test variable name black list. This should result in the variable
          # not being a const.
          constant_graph_def_with_blacklist = (
              graph_util.convert_variables_to_constants(
                  sess,
                  variable_graph_def, ["output_node"],
                  variable_names_blacklist=set(["variable_node"])))
          variable_node = None
          for node in constant_graph_def_with_blacklist.node:
            if node.name == "variable_node":
              variable_node = node
          self.assertIsNotNone(variable_node)
          if use_resource:
            self.assertEqual(variable_node.op, "VarHandleOp")
          else:
            self.assertEqual(variable_node.op, "VariableV2")

    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with ops.Graph().as_default():
      _ = importer.import_graph_def(constant_graph_def, name="")
      self.assertEqual(4, len(constant_graph_def.node))
      for node in constant_graph_def.node:
        self.assertNotIn(
            node.op,
            ["Variable", "VariableV2", "VarHandleOp", "ReadVariableOp"])
      with session.Session() as sess:
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        output = self.evaluate(output_node)
        self.assertNear(2.0, output, 0.00001)

  def create_node_def(self, op, name, inputs):
    new_node = node_def_pb2.NodeDef()
    new_node.op = op
    new_node.name = name
    for input_name in inputs:
      new_node.input.extend([input_name])
    return new_node

  def create_constant_node_def(self, name, value, dtype,
                               shape=None, inputs=None):
    node = self.create_node_def("Const", name, inputs or [])
    self.set_attr_dtype(node, "dtype", dtype)
    self.set_attr_tensor(node, "value", value, dtype, shape)
    return node

  def set_attr_dtype(self, node, key, value):
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(type=value.as_datatype_enum))

  def set_attr_tensor(self, node, key, value, dtype, shape=None):
    node.attr[key].CopyFrom(
        attr_value_pb2.AttrValue(tensor=tensor_util.make_tensor_proto(
            value, dtype=dtype, shape=shape)))

  def testRemoveTrainingNodes(self):
    a_constant_name = "a_constant"
    b_constant_name = "b_constant"
    a_check_name = "a_check"
    b_check_name = "b_check"
    a_identity_name = "a_identity"
    b_identity_name = "b_identity"
    add_name = "add"
    graph_def = graph_pb2.GraphDef()
    a_constant = self.create_constant_node_def(
        a_constant_name, value=1, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([a_constant])
    a_check_node = self.create_node_def("CheckNumerics", a_check_name,
                                        [a_constant_name])
    graph_def.node.extend([a_check_node])
    a_identity_node = self.create_node_def(
        "Identity", a_identity_name, [a_constant_name, "^" + a_check_name])
    graph_def.node.extend([a_identity_node])
    b_constant = self.create_constant_node_def(
        b_constant_name, value=1, dtype=dtypes.float32, shape=[])
    graph_def.node.extend([b_constant])
    b_check_node = self.create_node_def("CheckNumerics", b_check_name,
                                        [b_constant_name])
    graph_def.node.extend([b_check_node])
    b_identity_node = self.create_node_def(
        "Identity", b_identity_name, [b_constant_name, "^" + b_check_name])
    graph_def.node.extend([b_identity_node])
    add_node = self.create_node_def("Add", add_name,
                                    [a_identity_name, b_identity_name])
    self.set_attr_dtype(add_node, "T", dtypes.float32)
    graph_def.node.extend([add_node])

    expected_output = graph_pb2.GraphDef()
    a_constant = self.create_constant_node_def(
        a_constant_name, value=1, dtype=dtypes.float32, shape=[])
    expected_output.node.extend([a_constant])
    b_constant = self.create_constant_node_def(
        b_constant_name, value=1, dtype=dtypes.float32, shape=[])
    expected_output.node.extend([b_constant])
    add_node = self.create_node_def("Add", add_name,
                                    [a_constant_name, b_constant_name])
    self.set_attr_dtype(add_node, "T", dtypes.float32)
    expected_output.node.extend([add_node])

    output = graph_util.remove_training_nodes(graph_def)
    self.assertProtoEquals(expected_output, output)

  def testRemoveIdentityChains(self):
    """Check that chains of Identity nodes are correctly pruned.

    Create a chain of four nodes, A, B, C, and D where A inputs B, B inputs C,
    and C inputs D. Nodes B and C are "Identity" and should be pruned, resulting
    in the nodes A and D, where A inputs D.
    """
    graph_def = graph_pb2.GraphDef()
    graph_def.node.extend([
        self.create_node_def("Aop", "A", ["B"]), self.create_node_def(
            "Identity", "B", ["C"]), self.create_node_def(
                "Identity", "C", ["D"]), self.create_node_def("Dop", "D", [])
    ])

    expected_graph_def = graph_pb2.GraphDef()
    expected_graph_def.node.extend([
        self.create_node_def("Aop", "A", ["D"]), self.create_node_def(
            "Dop", "D", [])
    ])

    self.assertProtoEquals(expected_graph_def,
                           graph_util.remove_training_nodes(graph_def))

  def testRemoveIdentityUsedAsControlInputInConst(self):
    """Check that Identity nodes used as control inputs are not removed."""
    graph_def = graph_pb2.GraphDef()
    graph_def.node.extend([
        self.create_constant_node_def("C", 1, dtypes.float32, inputs=["^I"]),
        self.create_node_def("Identity", "I", ["Base"]),
        self.create_node_def("BaseOp", "Base", [])
    ])

    self.assertProtoEquals(graph_def,
                           graph_util.remove_training_nodes(graph_def))


if __name__ == "__main__":
  test.main()
