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
import numpy as np

from tensorflow.python import keras
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import simple_save
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.training.tracking import tracking
from tensorflow.python.util import nest


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

  def _testConvertedFunction(self, obj, func, converted_concrete_func,
                             input_data):
    # Check that the converted ConcreteFunction produces the same result as the
    # original Function.
    expected_value = nest.flatten(func(**input_data))
    actual_value = nest.flatten(converted_concrete_func(**input_data))

    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(expected.numpy(), actual.numpy())

    # Ensure the shape is retained.
    for tensor in converted_concrete_func.inputs:
      actual_shape = input_data[tensor.name.split(":")[0]].shape
      self.assertEqual(tensor.shape, actual_shape)

    # Save the converted ConcreteFunction as a signature.
    save_dir = os.path.join(self.get_temp_dir(), "frozen_saved_model")
    root = tracking.AutoTrackable()
    root.f = converted_concrete_func
    save(root, save_dir, {"mykey": converted_concrete_func})

    # Load it back and make sure it works.
    loaded_obj = load(save_dir)
    actual_value = nest.flatten(loaded_obj.signatures["mykey"](**input_data))
    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(expected.numpy(), actual.numpy())

  @test_util.run_v2_only
  def testConstSavedModel(self):
    """Test a basic model with functions to make sure functions are inlined."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = tracking.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    to_save = root.f.get_concrete_function(input_data["x"])

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

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testVariableModel(self):
    """Test a basic model with Variables."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    input_func = root.f.get_concrete_function(input_data["x"])

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(2, self._getNumVariables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testScalarModel(self):
    """Test a basic model with Variables."""
    input_data = {"x": constant_op.constant(1., shape=[])}
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    input_func = root.f.get_concrete_function(input_data["x"])

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(2, self._getNumVariables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testVariableSavedModel(self):
    """Test a basic model with Variables with saving/loading the SavedModel."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = tracking.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    to_save = root.f.get_concrete_function(input_data["x"])

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

    self._testConvertedFunction(root, root.f, output_func, input_data)

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

    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = BasicModel()
    input_func = root.add.get_concrete_function(input_data["x"])

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(1, self._getNumVariables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    self._testConvertedFunction(root, root.add, output_func, input_data)

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
    actual_value = nest.flatten(output_func(input_data))
    self.assertEqual(expected_value.numpy(), actual_value)

  def _singleMetaGraphSavedModel(self):
    export_graph = ops.Graph()
    with export_graph.as_default():
      start = array_ops.placeholder(
          shape=[1, 1], dtype=dtypes.float32, name="start")
      distractor = variables.RefVariable(-1., name="distractor")
      v = variables.RefVariable(3., name="v")
      local_variable = variables.VariableV1(
          1.,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          trainable=False,
          use_resource=True)
      output = array_ops.identity(start * v * local_variable, name="output")
      with session_lib.Session() as session:
        session.run([v.initializer, distractor.initializer,
                     local_variable.initializer])
        path = os.path.join(self.get_temp_dir(), "saved_model", str(ops.uid()))
        simple_save.simple_save(
            session,
            path,
            inputs={"start": start},
            outputs={"output": output},
            legacy_init_op=local_variable.initializer)
    return path

  @test_util.run_v2_only
  def testRefVariableImport(self):
    saved = self._singleMetaGraphSavedModel()
    imported = load(saved)
    fn = imported.signatures["serving_default"]
    output_func = convert_to_constants.convert_variables_to_constants_v2(fn)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    input_data = {"start": constant_op.constant(1., shape=[1, 1])}
    root = tracking.AutoTrackable()
    self._testConvertedFunction(root, fn, output_func, input_data)

  @test_util.run_v2_only
  def testIf(self):
    """Test whether If op freezes correctly."""
    input_data = {
        "x": constant_op.constant([1., 2.], shape=[1, 2]),
        "b": constant_op.constant(True)
    }

    weights = variables.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=dtypes.float32)

    def true_fn(x):
      return math_ops.matmul(x, weights)

    def false_fn(x):
      return math_ops.add(x, weights)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[1, 2], dtype=dtypes.float32),
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool)
    ])
    def model(x, b):
      return control_flow_ops.cond(
          b, true_fn=lambda: true_fn(x), false_fn=lambda: false_fn(x))

    root = tracking.AutoTrackable()
    root.f = model
    input_func = root.f.get_concrete_function()
    input_func(**input_data)

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func, lower_control_flow=False)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testStatelessIf(self):
    """Test whether StatelessIf op freezes correctly."""
    input_data = {"b": constant_op.constant(True)}

    x = constant_op.constant([1., 2.], shape=[1, 2], name="x")

    def true_fn():
      return x

    def false_fn():
      return x + 2

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool)])
    def model(b):
      return cond_v2.cond_v2(b, true_fn, false_fn)

    root = tracking.AutoTrackable()
    root.f = model
    input_func = root.f.get_concrete_function()
    input_func(**input_data)

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func, lower_control_flow=False)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testStaticRnn(self):
    input_data = {
        "x":
            constant_op.constant(
                np.array(np.random.random_sample((3, 10)), dtype=np.float32))
    }

    cell = rnn_cell_impl.LSTMCell(10)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[3, 10], dtype=dtypes.float32)
    ])
    def model(x):
      seq = array_ops.split(x, 3, 0)
      return rnn.static_rnn(
          cell, seq, dtype=dtypes.float32, sequence_length=[1])

    root = tracking.AutoTrackable()
    root.f = model
    input_func = root.f.get_concrete_function()

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func, lower_control_flow=False)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testLoop(self):
    input_data = {"x": constant_op.constant([1., 2., 3., 4.], shape=[2, 2])}

    weights = variables.Variable([[0.1, 0.2], [0.3, 0.4]], dtype=dtypes.float32)

    def condition(x):
      return math_ops.reduce_sum(x) < 100

    def body(x):
      return math_ops.add(x, weights)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[2, 2], dtype=dtypes.float32)
    ])
    def model(x):
      return control_flow_ops.while_loop(condition, body, [x])

    root = tracking.AutoTrackable()
    root.f = model
    input_func = root.f.get_concrete_function()
    input_func(**input_data)

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func, lower_control_flow=False)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testDynamicRnn(self):
    input_data = {
        "x":
            constant_op.constant(
                np.array(
                    np.random.random_sample((3, 10, 10)), dtype=np.float32))
    }

    cell = rnn_cell_impl.LSTMCell(10)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[3, 10, 10], dtype=dtypes.float32)
    ])
    def model(x):
      return rnn.dynamic_rnn(cell, x, dtype=dtypes.float32)

    root = tracking.AutoTrackable()
    root.f = model
    input_func = root.f.get_concrete_function()

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func, lower_control_flow=False)
    constant_graph_def = output_func.graph.as_graph_def()
    self.assertEqual(0, self._getNumVariables(constant_graph_def))
    self.assertFalse(self._hasStatefulPartitionedCallOp(constant_graph_def))

    self._testConvertedFunction(root, root.f, output_func, input_data)


if __name__ == "__main__":
  test.main()
