# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for convert_to_constants.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensorflow.python import keras
from tensorflow.python.client import session
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.training.tracking import tracking


# TODO(nupurgarg): Simplify the test cases to use the ConcreteFunction.
class VariablesToConstantsTest(test.TestCase):

  def _hasStatefulPartitionedCallOp(self, graph_def):
    """Determines if a StatefulPartitionedCall op exists in the graph."""
    for node in graph_def.node:
      if node.op == "StatefulPartitionedCall":
        return True
    return False

  def _getNumVariables(self, graph_def):
    """Returns the number of ReadVariableOp in the graph."""
    return sum(node.op == "ReadVariableOp" for node in graph_def.node)

  def _getTensors(self, sess, tensor_list):
    """Returns a list of Tensor objects from the Session."""
    return [
        sess.graph.get_tensor_by_name(tensor.name) for tensor in tensor_list
    ]

  def _evaluateGraphDef(self, graph_def, func, input_data):
    """Evaluates the GraphDef using Sessions."""
    with ops.Graph().as_default() as graph:
      importer.import_graph_def(graph_def, name="")
      func.add_to_graph(graph)
      sess = session.Session(graph=graph)

    input_tensors = self._getTensors(sess, func.inputs)
    output_tensors = self._getTensors(sess, func.outputs)
    return sess.run(
        output_tensors, feed_dict=dict(zip(input_tensors, input_data)))

  @test_util.run_v2_only
  def testConstSavedModel(self):
    """Test a basic model with functions to make sure functions are inlined."""
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save(root, save_dir, to_save)
    saved_model = load(save_dir)
    input_func = saved_model.signatures["serving_default"]

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(variable_graph_def))
    self.assertTrue(variable_graph_def.library.function)

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(constant_graph_def.library.function)

    # Check value.
    expected_value = root.f(input_data)
    actual_value = self._evaluateGraphDef(constant_graph_def, input_func,
                                          [input_data.numpy()])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testVariableModel(self):
    """Test a basic model with Variables."""
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    input_func = root.f.get_concrete_function(input_data)

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(2, self._getNumVariables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    # Check value.
    expected_value = root.f(input_data)
    actual_value = self._evaluateGraphDef(constant_graph_def, input_func,
                                          [input_data.numpy()])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testVariableSavedModel(self):
    """Test a basic model with Variables with saving/loading the SavedModel."""
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    to_save = root.f.get_concrete_function(input_data)

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save(root, save_dir, to_save)
    saved_model = load(save_dir)
    input_func = saved_model.signatures["serving_default"]

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertTrue(self._hasStatefulPartitionedCallOp(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    # Check value.
    expected_value = root.f(input_data)
    actual_value = self._evaluateGraphDef(constant_graph_def, input_func,
                                          [input_data.numpy()])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testMultiFunctionModel(self):
    """Test a basic model with Variables."""

    class BasicModel(tracking.AutoTrackable):

      def __init__(self):
        self.y = None
        self.z = None

      @def_function.function
      def add(self, x):
        if self.y is None:
          self.y = variables.Variable(2.)
        return x + self.y

      @def_function.function
      def sub(self, x):
        if self.z is None:
          self.z = variables.Variable(3.)
        return x - self.z

    input_data = constant_op.constant(1., shape=[1])
    root = BasicModel()
    input_func = root.add.get_concrete_function(input_data)

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(1, self._getNumVariables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    # Check value.
    expected_value = root.add(input_data)
    actual_value = self._evaluateGraphDef(constant_graph_def, input_func,
                                          [input_data.numpy()])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testConstructConcreteFunction(self):
    input_data = constant_op.constant(1., shape=[1])
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    func = root.f.get_concrete_function(input_data)

    input_func = convert_to_constants._construct_concrete_function(
        func, func.graph.as_graph_def())

    # Test if model has enough metadata to be frozen afterwards.
    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(2, self._getNumVariables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    # Check value.
    expected_value = root.f(input_data)
    actual_value = self._evaluateGraphDef(constant_graph_def, input_func,
                                          [input_data.numpy()])
    self.assertEqual(expected_value.numpy(), actual_value)

  @test_util.run_v2_only
  def testKerasModel(self):
    input_data = constant_op.constant(1., shape=[1, 1])

    # Create a simple Keras model.
    x = [-1, 0, 1, 2, 3, 4]
    y = [-3, -1, 1, 3, 5, 7]

    model = keras.models.Sequential(
        [keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer="sgd", loss="mean_squared_error")
    model.fit(x, y, epochs=1)

    # Get the concrete function from the Keras model.
    @def_function.function
    def to_save(x):
      return model(x)

    input_func = to_save.get_concrete_function(input_data)

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(2, self._getNumVariables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    # Check value.
    expected_value = to_save(input_data)
    actual_value = self._evaluateGraphDef(constant_graph_def, input_func,
                                          [input_data.numpy()])
    self.assertEqual(expected_value.numpy(), actual_value)


if __name__ == "__main__":
  test.main()
