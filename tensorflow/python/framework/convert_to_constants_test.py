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

import os
import re

import numpy as np

from google.protobuf import text_format
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import function_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import convert_to_constants
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.platform import test
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import simple_save
from tensorflow.python.saved_model.load import load
from tensorflow.python.saved_model.save import save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training.saver import export_meta_graph
from tensorflow.python.util import compat
from tensorflow.python.util import nest


class _GraphMerger(object):
  """GraphDef merging methods for testing purposes."""

  @staticmethod
  def merge_any(x1, x2, empty_fn):
    """Merges two values using the message's CopyFrom/MergeFrom methods."""
    merged = empty_fn()
    merged.CopyFrom(x1)
    merged.MergeFrom(x2)
    return merged

  @staticmethod
  def merge_nodes(node1, node2):
    """Merges two NodeDef messages."""
    merged = _GraphMerger.merge_any(node1, node2, node_def_pb2.NodeDef)
    merged_inputs = node1.input[:]
    merged_inputs.extend([i for i in node2.input[:] if i not in merged_inputs])
    merged.input[:] = merged_inputs
    return merged

  @staticmethod
  def merge_lists(repeated1, repeated2, empty_fn, key_fn, merge_fn):
    """Merges two lists representing maps."""
    merged = {}
    xs1 = {key_fn(x): x for x in repeated1}
    xs2 = {key_fn(x): x for x in repeated2}
    for name in set().union(xs1.keys(), xs2.keys()):
      x1 = empty_fn() if name not in xs1 else xs1[name]
      x2 = empty_fn() if name not in xs2 else xs2[name]
      merged[name] = merge_fn(x1, x2)
    return sorted(merged.values(), key=key_fn)

  @staticmethod
  def merge_node_lists(repeated_nodes1, repeated_nodes2):
    """Merges two repeated node fields."""
    return _GraphMerger.merge_lists(repeated_nodes1, repeated_nodes2,
                                    node_def_pb2.NodeDef, lambda n: n.name,
                                    _GraphMerger.merge_nodes)

  @staticmethod
  def merge_functions(fn1, fn2):
    """Merges two FunctionDefs."""
    merged = _GraphMerger.merge_any(fn1, fn2, function_pb2.FunctionDef)

    del merged.signature.input_arg[:]
    merged.signature.input_arg.extend(
        _GraphMerger.merge_lists(
            fn1.signature.input_arg[:], fn2.signature.input_arg[:],
            op_def_pb2.OpDef.ArgDef, lambda a: a.name,
            lambda x, y: _GraphMerger.merge_any(x, y, op_def_pb2.OpDef.ArgDef)))

    del merged.signature.output_arg[:]
    merged.signature.output_arg.extend(
        _GraphMerger.merge_lists(
            fn1.signature.output_arg[:], fn2.signature.output_arg[:],
            op_def_pb2.OpDef.ArgDef, lambda a: a.name,
            lambda x, y: _GraphMerger.merge_any(x, y, op_def_pb2.OpDef.ArgDef)))

    del merged.node_def[:]
    merged.node_def.extend(
        _GraphMerger.merge_node_lists(fn1.node_def[:], fn2.node_def[:]))

    return merged

  @staticmethod
  def merge_graphs(graph1, graph2):
    """Merges two GraphDef messages."""
    merged = graph_pb2.GraphDef()
    merged.node.extend(
        _GraphMerger.merge_node_lists(graph1.node[:], graph2.node[:]))

    merged.library.function.extend(
        _GraphMerger.merge_lists(graph1.library.function,
                                 graph2.library.function,
                                 function_pb2.FunctionDef,
                                 lambda f: f.signature.name,
                                 _GraphMerger.merge_functions))

    return merged


def has_stateful_partitioned_call_op(graph_def):
  """Determines if a StatefulPartitionedCall op exists in the graph."""
  for node in graph_def.node:
    if node.op == "StatefulPartitionedCall":
      return True
  return False


def get_num_variables(graph_def):
  """Returns the number of ReadVariableOp in the graph."""
  return sum(node.op == "ReadVariableOp" for node in graph_def.node)


class VariablesToConstantsTest(test.TestCase):

  def _freezeModel(self, func):
    """Freezes the function.

    Args:
      func: Function.

    Returns:
      root: AutoTrackable object with original ConcreteFunction.
      output_func: frozen ConcreteFunction.
    """
    root = autotrackable.AutoTrackable()
    root.f = func
    input_func = root.f.get_concrete_function()

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func, lower_control_flow=False)
    return root, output_func

  def _testConvertedFunction(self, obj, func, converted_concrete_func,
                             input_data):
    # Ensure the converted graph has no variables and no function calls.
    constant_graph_def = converted_concrete_func.graph.as_graph_def()
    self.assertEqual(0, get_num_variables(constant_graph_def))
    self.assertFalse(has_stateful_partitioned_call_op(constant_graph_def))

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
    root = autotrackable.AutoTrackable()
    root.f = converted_concrete_func
    save(root, save_dir, {"mykey": converted_concrete_func})

    # Load it back and make sure it works.
    loaded_obj = load(save_dir)
    actual_value = nest.flatten(loaded_obj.signatures["mykey"](**input_data))
    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(expected.numpy(), actual.numpy())

  @test_util.run_v2_only
  def testConstSavedModel(self):
    """Test a basic model with constants while saving/loading the SavedModel."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = autotrackable.AutoTrackable()
    root.f = def_function.function(lambda x: 2. * x)
    to_save = root.f.get_concrete_function(input_data["x"])

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save(root, save_dir, to_save)
    saved_model = load(save_dir)
    input_func = saved_model.signatures["serving_default"]

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(0, get_num_variables(variable_graph_def))
    self.assertTrue(variable_graph_def.library.function)

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testVariableModel(self):
    """Test a basic model with Variables."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    input_func = root.f.get_concrete_function(input_data["x"])

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(2, get_num_variables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testScalarModel(self):
    """Test a basic model with Variables."""
    input_data = {"x": constant_op.constant(1., shape=[])}
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    input_func = root.f.get_concrete_function(input_data["x"])

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertEqual(2, get_num_variables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testVariableSavedModel(self):
    """Test a basic model with Variables with saving/loading the SavedModel."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    to_save = root.f.get_concrete_function(input_data["x"])

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save(root, save_dir, to_save)
    saved_model = load(save_dir)
    input_func = saved_model.signatures["serving_default"]

    variable_graph_def = input_func.graph.as_graph_def()
    self.assertTrue(has_stateful_partitioned_call_op(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testMultiFunctionModel(self):
    """Test a basic model with multiple tf.functions."""

    class BasicModel(autotrackable.AutoTrackable):

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
    self.assertEqual(1, get_num_variables(variable_graph_def))

    output_func = convert_to_constants.convert_variables_to_constants_v2(
        input_func)
    self._testConvertedFunction(root, root.add, output_func, input_data)

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
    """Test a model with 1.X ReferenceVariables."""
    input_data = {"start": constant_op.constant(1., shape=[1, 1])}

    saved = self._singleMetaGraphSavedModel()
    imported = load(saved)
    fn = imported.signatures["serving_default"]

    output_func = convert_to_constants.convert_variables_to_constants_v2(fn)
    root = autotrackable.AutoTrackable()
    self._testConvertedFunction(root, fn, output_func, input_data)

  @test_util.run_v2_only
  def testIf(self):
    """Test a model with the If op."""
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

    root, output_func = self._freezeModel(model)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testStatelessIf(self):
    """Test a model with the StatelessIf op."""
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

    root, output_func = self._freezeModel(model)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testStaticRnn(self):
    """Test a StaticRnn containing If ops."""
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

    root, output_func = self._freezeModel(model)

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testWhile(self):
    """Test a While loop."""
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

    root, output_func = self._freezeModel(model)

    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testStatelessWhile(self):
    """Test a StatelessWhile loop."""
    input_data = {"x": constant_op.constant(2.)}

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32)
    ])
    def model(x):
      return while_v2.while_loop(
          lambda v: v < 4.,
          lambda v: v * v, [x],
          return_same_structure=False,
          name="while_1")  # x**2

    root, output_func = self._freezeModel(model)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  def testDynamicRnn(self):
    """Test a DynamicRnn containing While loops."""
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

    root, output_func = self._freezeModel(model)
    self._testConvertedFunction(root, root.f, output_func, input_data)

  @test_util.run_v2_only
  @test_util.disable_tfrt("b/180451239")
  def testSwitchCase(self):
    """Test a switch_case statement."""
    input_data = {
        "i": constant_op.constant(np.random.randint(0, 3, dtype=np.int32)),
        "x": constant_op.constant(
            np.asarray(np.random.random_sample((10, 3)), dtype=np.float32)),
    }

    w0 = variables.Variable(np.random.random_sample((3, 4)), dtype=np.float32)
    w1 = variables.Variable(np.random.random_sample((3, 4)), dtype=np.float32)
    w2 = variables.Variable(np.random.random_sample((4,)), dtype=np.float32)

    def branch0(x):
      return math_ops.matmul(x, w0)

    def branch1(x):
      return math_ops.matmul(x, w1)

    def branch2(x):
      x = array_ops.pad(x, [[0, 0], [0, 1]])
      return x + w2

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32),
        tensor_spec.TensorSpec(shape=[10, 3], dtype=dtypes.float32),
    ])
    def model(i, x):
      return control_flow_ops.switch_case(i, [
          lambda: branch0(x), lambda: branch1(x), lambda: branch2(x)])

    root, output_func = self._freezeModel(model)
    self._testConvertedFunction(root, root.f, output_func, input_data)


class ConvertVariablesToConstantsV2SessionTest(test.TestCase):

  def _freezeModel(self, func):
    """Freezes the function.

    Args:
      func: Function.

    Returns:
      root: AutoTrackable object with original ConcreteFunction.
      output_func: frozen ConcreteFunction.
    """
    root = autotrackable.AutoTrackable()
    root.f = func
    input_func = root.f.get_concrete_function()

    output_func = convert_to_constants.convert_var_to_const_function_in_v1(
        input_func, lower_control_flow=False)
    return root, output_func

  def _testConvertedFunction(self, sess, obj, func, converted_concrete_func,
                             input_data):
    # Ensure the converted graph has no variables and no function calls.
    constant_graph_def = converted_concrete_func.graph.as_graph_def()
    self.assertEqual(0, get_num_variables(constant_graph_def))
    self.assertFalse(has_stateful_partitioned_call_op(constant_graph_def))

    # Check that the converted ConcreteFunction produces the same result as the
    # original Function.
    expected_value = nest.flatten(func(**input_data))
    actual_value = nest.flatten(converted_concrete_func(**input_data))

    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(sess.run(expected), sess.run(actual))

    # Ensure the shape is retained.
    for tensor in converted_concrete_func.inputs:
      actual_shape = input_data[tensor.name.split(":")[0]].shape
      self.assertEqual(tensor.shape, actual_shape)

    # Save the converted ConcreteFunction as a signature.
    save_dir = os.path.join(self.get_temp_dir(), "frozen_saved_model")
    root = autotrackable.AutoTrackable()
    root.f = converted_concrete_func
    save(root, save_dir, {"mykey": converted_concrete_func})

    # Load it back and make sure it works.
    loaded_obj = load(save_dir)
    actual_value = nest.flatten(loaded_obj.signatures["mykey"](**input_data))
    for expected, actual in zip(expected_value, actual_value):
      np.testing.assert_almost_equal(sess.run(expected), sess.run(actual))

  def testRaiseErrorInEagerMode(self):
    """Test the raised exception in Eager mode."""
    input_data = {"x": constant_op.constant(1., shape=[1])}
    root = autotrackable.AutoTrackable()
    root.v1 = variables.Variable(3.)
    root.v2 = variables.Variable(2.)
    root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
    input_func = root.f.get_concrete_function(input_data["x"])

    with self.assertRaisesRegex(RuntimeError,
                                "must be carried out in a Session"):
      convert_to_constants.convert_var_to_const_function_in_v1(
          input_func)

  def testConvertVariables(self):
    """Test a basic model with Variables."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(1., shape=[1])}
        root = autotrackable.AutoTrackable()
        root.v1 = variables.Variable(3.)
        root.v2 = variables.Variable(2.)
        root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
        input_func = root.f.get_concrete_function(input_data["x"])

        variable_graph_def = input_func.graph.as_graph_def()
        self.assertEqual(2, get_num_variables(variable_graph_def))

        output_func = convert_to_constants.convert_var_to_const_function_in_v1(
            input_func)

        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testConvertVariablesWithAssignments(self):
    """Test a basic model with Variables and assignment ops."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(1., shape=[1])}
        root = autotrackable.AutoTrackable()
        root.v1 = variables.Variable(3.)
        root.v2 = variables.Variable(2.)
        root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
        input_func = root.f.get_concrete_function(input_data["x"])

        variable_graph_def = input_func.graph.as_graph_def()
        self.assertEqual(2, get_num_variables(variable_graph_def))

        assign_op_1 = root.v1.assign(1.5)
        assign_op_2 = root.v2.assign(3.0)
        assign_op_3 = root.v1.assign(4.0)
        ops.get_default_graph().add_to_collection(
            convert_to_constants.VAR_ASSIGN_COLLECTION, assign_op_1)
        ops.get_default_graph().add_to_collection(
            convert_to_constants.VAR_ASSIGN_COLLECTION, assign_op_2)
        ops.get_default_graph().add_to_collection(
            convert_to_constants.VAR_ASSIGN_COLLECTION, assign_op_3)

        output_func = convert_to_constants.convert_var_to_const_function_in_v1(
            input_func)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testConstSavedModel(self):
    """Test a basic model with constants while saving/loading the SavedModel."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(1., shape=[1])}
        root = autotrackable.AutoTrackable()
        root.f = def_function.function(lambda x: 2. * x)
        to_save = root.f.get_concrete_function(input_data["x"])

        save_dir = os.path.join(self.get_temp_dir(), "saved_model")
        save(root, save_dir, to_save)
        saved_model = load(save_dir)
        input_func = saved_model.signatures["serving_default"]

        variable_graph_def = input_func.graph.as_graph_def()
        self.assertEqual(0, get_num_variables(variable_graph_def))
        self.assertTrue(variable_graph_def.library.function)

        output_func = convert_to_constants.convert_var_to_const_function_in_v1(
            input_func)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testVariableSavedModel(self):
    """Test a basic model with Variables with saving/loading the SavedModel."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(1., shape=[1])}
        root = autotrackable.AutoTrackable()
        root.v1 = variables.Variable(3.)
        root.v2 = variables.Variable(2.)
        root.f = def_function.function(lambda x: root.v1 * root.v2 * x)
        to_save = root.f.get_concrete_function(input_data["x"])
        sess.run(variables.global_variables_initializer())

        save_dir = os.path.join(self.get_temp_dir(), "saved_model")
        save(root, save_dir, to_save)
        saved_model = load(save_dir)
        input_func = saved_model.signatures["serving_default"]

        variable_graph_def = input_func.graph.as_graph_def()
        self.assertTrue(has_stateful_partitioned_call_op(variable_graph_def))

        output_func = convert_to_constants.convert_var_to_const_function_in_v1(
            input_func)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testMultiFunctionModel(self):
    """Test a basic model with multiple tf.functions."""

    class BasicModel(autotrackable.AutoTrackable):

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

    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(1., shape=[1])}
        root = BasicModel()
        input_func = root.add.get_concrete_function(input_data["x"])

        variable_graph_def = input_func.graph.as_graph_def()
        self.assertEqual(1, get_num_variables(variable_graph_def))

        output_func = convert_to_constants.convert_var_to_const_function_in_v1(
            input_func)
        self._testConvertedFunction(sess, root, root.add, output_func,
                                    input_data)

  def testIf(self):
    """Test a model with the If op."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {
            "x": constant_op.constant([1., 2.], shape=[1, 2]),
            "b": constant_op.constant(True)
        }

        weights = variables.Variable([[0.1, 0.2], [0.3, 0.4]],
                                     dtype=dtypes.float32)

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

        root, output_func = self._freezeModel(model)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testStatelessIf(self):
    """Test a model with the StatelessIf op."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"b": constant_op.constant(True)}

        x = constant_op.constant([1., 2.], shape=[1, 2], name="x")

        def true_fn():
          return x

        def false_fn():
          return x + 2

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool)
        ])
        def model(b):
          return cond_v2.cond_v2(b, true_fn, false_fn)

        root, output_func = self._freezeModel(model)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testStaticRnn(self):
    """Test a StaticRnn containing If ops."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {
            "x":
                constant_op.constant(
                    np.array(
                        np.random.random_sample((3, 10)), dtype=np.float32))
        }

        cell = rnn_cell_impl.LSTMCell(10)

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[3, 10], dtype=dtypes.float32)
        ])
        def model(x):
          seq = array_ops.split(x, 3, 0)
          return rnn.static_rnn(
              cell, seq, dtype=dtypes.float32, sequence_length=[1])

        root, output_func = self._freezeModel(model)

        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testWhile(self):
    """Test a While loop."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant([1., 2., 3., 4.], shape=[2, 2])}

        weights = variables.Variable([[0.1, 0.2], [0.3, 0.4]],
                                     dtype=dtypes.float32)

        def condition(x):
          return math_ops.reduce_sum(x) < 100

        def body(x):
          return math_ops.add(x, weights)

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[2, 2], dtype=dtypes.float32)
        ])
        def model(x):
          return control_flow_ops.while_loop(condition, body, [x])

        root, output_func = self._freezeModel(model)

        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testStatelessWhile(self):
    """Test a StatelessWhile loop."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {"x": constant_op.constant(2.)}

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=(), dtype=dtypes.float32)
        ])
        def model(x):
          return while_v2.while_loop(
              lambda v: v < 4.,
              lambda v: v * v, [x],
              return_same_structure=False,
              name="while_1")  # x**2

        root, output_func = self._freezeModel(model)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  def testDynamicRnn(self):
    """Test a DynamicRnn containing While loops."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
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

        root, output_func = self._freezeModel(model)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)

  @test_util.disable_tfrt("b/180451239")
  def testSwitchCase(self):
    """Test a switch_case statement."""
    with ops.Graph().as_default():
      with session_lib.Session() as sess:
        input_data = {
            "i":
                constant_op.constant(np.random.randint(0, 3, dtype=np.int32)),
            "x":
                constant_op.constant(
                    np.asarray(
                        np.random.random_sample((10, 3)), dtype=np.float32)),
        }

        w0 = variables.Variable(
            np.random.random_sample((3, 4)), dtype=np.float32)
        w1 = variables.Variable(
            np.random.random_sample((3, 4)), dtype=np.float32)
        w2 = variables.Variable(np.random.random_sample((4,)), dtype=np.float32)

        def branch0(x):
          return math_ops.matmul(x, w0)

        def branch1(x):
          return math_ops.matmul(x, w1)

        def branch2(x):
          x = array_ops.pad(x, [[0, 0], [0, 1]])
          return x + w2

        @def_function.function(input_signature=[
            tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32),
            tensor_spec.TensorSpec(shape=[10, 3], dtype=dtypes.float32),
        ])
        def model(i, x):
          return control_flow_ops.switch_case(
              i, [lambda: branch0(x), lambda: branch1(x), lambda: branch2(x)])

        root, output_func = self._freezeModel(model)
        self._testConvertedFunction(sess, root, root.f, output_func, input_data)


class ConvertVariablesToConstantsSessionTest(test.TestCase):

  def _assertGraphContains(self, graph, subgraph):
    """Asserts that the given subgraph is contained within the given graph."""

    def normalize_uids(msg):
      """Replace auto-id function names with something consistent."""
      # These functions have non-deterministic names, the non-determinism coming
      # from having an ops.uid() suffix in their names. We're replacing these
      # with new sequential IDs starting from 0 for each prefix, which is
      # is sufficient for tests.
      if isinstance(msg, graph_pb2.GraphDef):
        msg = text_format.MessageToString(msg)
      name_prefixes = ["case_cond_true.*", "case_cond_false.*"]
      name_regex = r"\b(" + "|".join(name_prefixes) + r")_([0-9]+)\b"
      names = {}
      for (name, index) in re.findall(name_regex, msg):
        names.setdefault(name, set()).add(int(index))
      for name, indices in names.items():
        for new_index, old_index in enumerate(sorted(list(indices))):
          msg = re.sub(r"\b" + name + "_" + str(old_index) + r"\b",
                       name + "_" + str(new_index), msg)
      return msg

    norm_graph = text_format.Parse(normalize_uids(graph), graph_pb2.GraphDef())
    norm_subgraph = text_format.Parse(
        normalize_uids(subgraph), graph_pb2.GraphDef())

    # Graph S is contained in C if and only if merge(C,S) == C.
    # We merge the input graph with an empty graph to normalize repeated fields:
    # assertProtoEquals is sensitive to ordering.
    norm_graph = _GraphMerger.merge_graphs(norm_graph, graph_pb2.GraphDef())
    merged_graph = _GraphMerger.merge_graphs(norm_graph, norm_subgraph)
    self.assertProtoEquals(norm_graph, merged_graph)

  def _ensure_no_variables_in_graph(self, graph_def):
    """Ensures there are no variables in the graph."""
    for node in graph_def.node:
      self.assertNotIn(
          node.op, ["Variable", "VariableV2", "VarHandleOp", "ReadVariableOp"])

  def _test_variable_to_const_conversion(self, use_resource):
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=use_resource):
        variable_node = variable_scope.get_variable(
            "variable_node", initializer=1.0)
        variable_scope.get_variable("unused_variable_node", initializer=1.0)
        output_node = math_ops.multiply(variable_node, 2.0, name="output_node")
        with session_lib.Session() as sess:
          self.evaluate(variable_node.initializer)
          output = self.evaluate(output_node)
          self.assertNear(2.0, output, 0.00001)
          variable_graph_def = sess.graph.as_graph_def()
          constant_graph_def = (
              convert_to_constants
              .convert_variables_to_constants_from_session_graph(
                  session=sess,
                  graph_def=variable_graph_def,
                  output_node_names=["output_node"]))

          self._ensure_no_variables_in_graph(constant_graph_def)

    # Now we make sure the variable is now a constant, and that the graph still
    # produces the expected result.
    with ops.Graph().as_default():
      _ = importer.import_graph_def(constant_graph_def, name="")
      self.assertEqual(4, len(constant_graph_def.node))
      self._ensure_no_variables_in_graph(constant_graph_def)
      with session_lib.Session() as sess:
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        output = self.evaluate(output_node)
        self.assertNear(2.0, output, 0.00001)

  def test_resource_variable_can_be_written_after_denylisting(self):
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        variable_node = variable_scope.get_variable(
            "variable_node", initializer=1.0)
        another_variable = variable_scope.get_variable(
            "unused_variable_node", initializer=2.0)
        with ops.control_dependencies(
            [variable_node.assign(another_variable + variable_node)]):
          output_node = array_ops.identity(variable_node, name="output_node")
        initializer_name = variable_node.initializer.name
        with session_lib.Session() as sess:
          self.evaluate(variable_node.initializer)
          self.evaluate(another_variable.initializer)
          output = self.evaluate(output_node)
          self.assertNear(3.0, output, 0.00001)
          variable_graph_def = sess.graph.as_graph_def()

          # Test variable name black list. This should result in the variable
          # not being a const.  Furthermore, the paths that read from and assign
          # to the denylisted variable should continue to be valid.
          constant_graph_def_with_denylist = (
              convert_to_constants
              .convert_variables_to_constants_from_session_graph(
                  session=sess,
                  graph_def=variable_graph_def,
                  output_node_names=["output_node", initializer_name],
                  variable_names_denylist=set(["variable_node"])))

          variable_node = None
          for node in constant_graph_def_with_denylist.node:
            if node.name == "variable_node":
              variable_node = node
          self.assertIsNotNone(variable_node)
          self.assertEqual(variable_node.op, "VarHandleOp")

    # Now we make sure another_variable is now a constant, but the original
    # variable is not, and that the graph can be executed and update the
    # variable can be updated with each execution.
    with ops.Graph().as_default():
      _ = importer.import_graph_def(constant_graph_def_with_denylist, name="")
      with session_lib.Session() as sess:
        output_node = sess.graph.get_tensor_by_name("output_node:0")
        self.evaluate(sess.graph.get_operation_by_name(initializer_name))
        output = self.evaluate(output_node)
        self.assertNear(3.0, output, 0.00001)
        output = self.evaluate(output_node)
        self.assertNear(5.0, output, 0.00001)

  def _inline_functions(self, graph_def, arrays):
    meta_graph = export_meta_graph(graph_def=graph_def)
    fetch_collection = meta_graph_pb2.CollectionDef()
    for name in arrays:
      fetch_collection.node_list.value.append(name)
    meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)

    # Initialize RewriterConfig with everything disabled except function
    # inlining.
    config = config_pb2.ConfigProto()
    rewrite_options = config.graph_options.rewrite_options
    rewrite_options.optimizers.append("function")
    return tf_optimizer.OptimizeGraph(config, meta_graph)

  def _test_convert_variables_with_functions(self, inline_functions):
    """Freezes a graph with functions."""

    @function.Defun(dtypes.float32)
    def plus_one(x):
      return x + 1.0

    with ops.Graph().as_default():
      variable_node = variables.Variable(1.0, name="variable_node")
      _ = variables.Variable(1.0, name="unused_variable_node")
      defun_node = plus_one(variable_node)
      _ = math_ops.multiply(defun_node, 2.0, name="output_node")

      with session_lib.Session() as sess:
        self.evaluate(variables.variables_initializer([variable_node]))
        variable_graph_def = sess.graph.as_graph_def()

        if inline_functions:
          # Run Grappler to create the VarOpHandle --> Placeholder -->
          # ResourceVariable pattern.
          variable_graph_def = self._inline_functions(
              variable_graph_def, ["variable_node", "output_node"])

        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                session=sess,
                graph_def=variable_graph_def,
                output_node_names=["output_node"]))

    self._ensure_no_variables_in_graph(constant_graph_def)

  def testReferenceVariables(self):
    """Freezes a graph with reference variables."""
    self._test_variable_to_const_conversion(use_resource=False)

  def testResourceVariables(self):
    """Freezes a graph with resource variables."""
    self._test_variable_to_const_conversion(use_resource=True)

  def testWithFunctions(self):
    """Freezes a graph with functions."""
    self._test_convert_variables_with_functions(inline_functions=False)

  def testWithInlinedFunctions(self):
    """Freezes a graph with functions that have been inlined using Grappler."""
    self._test_convert_variables_with_functions(inline_functions=True)

  def testGraphWithSwitch(self):
    """Freezes a graph which contains a Switch with type RESOURCE_DT."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        x = variable_scope.get_variable("var_x", initializer=1.0)
        y = variable_scope.get_variable("var_y", initializer=2.0)
        f1 = lambda: variable_scope.get_variable("var_f1", initializer=17.0)
        f2 = lambda: variable_scope.get_variable("var_f2", initializer=23.0)
        cond_node = control_flow_ops.case([(gen_math_ops.less(x, y), f1)],
                                          default=f2)
        _ = math_ops.multiply(cond_node, 2.0, name="output_node")

        with session_lib.Session() as sess:
          sess.run(variables.global_variables_initializer())
          variable_graph_def = sess.graph.as_graph_def()

          constant_graph_def = (
              convert_to_constants
              .convert_variables_to_constants_from_session_graph(
                  session=sess,
                  graph_def=variable_graph_def,
                  output_node_names=["output_node"]))

    self._ensure_no_variables_in_graph(constant_graph_def)

  def testConvertSingleVariable(self):
    """Tests that a single variable is properly converted to a constant."""

    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=False):
        _ = variable_scope.get_variable("x", initializer=1.0)
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["x/read"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "x/read" op: "Identity" input: "x"
              attr { key: "T" value { type: DT_FLOAT } }
            }""")

  def testConvertSingleResourceVariable(self):
    """Tests that a resource variable is properly converted to a constant."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        _ = variable_scope.get_variable("x", initializer=1.0)
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["x/Read/ReadVariableOp"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "x/Read/ReadVariableOp" op: "Identity" input: "x"
              attr { key: "T" value { type: DT_FLOAT } }
            }""")

  def testConvertOneVariableOfTwo(self):
    """Tests that one variable can be kept unconverted."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=False):
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=1.0)
        _ = math_ops.multiply(x, y, name="out")
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess,
                variable_graph_def, ["out"],
                variable_names_denylist=["y"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "x/read" op: "Identity" input: "x"
              attr { key: "T" value { type: DT_FLOAT } }
            }
            node {
              name: "y" op: "VariableV2"
              attr { key: "dtype" value { type: DT_FLOAT } }
            }
            node {
              name: "y/read" op: "Identity" input: "y"
              attr { key: "T" value { type: DT_FLOAT } }
            }
            node {
              name: "out" op: "Mul" input: "x/read" input: "y/read"
              attr {key: "T" value {type: DT_FLOAT}}
            }""")

  def testConvertOneResourceVariableOfTwo(self):
    """Tests that one variable can be kept unconverted."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=1.0)
        _ = math_ops.multiply(x, y, name="out")
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess,
                variable_graph_def, ["out"],
                variable_names_denylist=["y"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "y" op: "VarHandleOp"
              attr { key: "dtype" value { type: DT_FLOAT } }
            }
            node {
              name: "out/ReadVariableOp" op: "Identity" input: "x"
              attr { key: "T" value { type: DT_FLOAT } }
            }
            node {
              name: "out/ReadVariableOp_1" op: "ReadVariableOp" input: "y"
              attr { key: "dtype" value { type: DT_FLOAT } }
            }
            node {
              name: "out" op: "Mul"
              input: "out/ReadVariableOp" input: "out/ReadVariableOp_1"
              attr {key: "T" value {type: DT_FLOAT}}
            }""")

  def testConvertIdentityChain(self):
    """Tests that a chain of Identity ops is converted properly."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        x = variable_scope.get_variable("x", initializer=1.0)
        y = array_ops.identity(x, name="y")
        _ = array_ops.identity(y, name="z")
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["z"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "y/ReadVariableOp" op: "Identity" input: "x"
              attr { key: "T" value { type: DT_FLOAT } }
            }
            node {
              name: "y" op: "Identity" input: "y/ReadVariableOp"
              attr { key: "T" value { type: DT_FLOAT } }
            }
            node {
              name: "z" op: "Identity" input: "y"
              attr { key: "T" value { type: DT_FLOAT } }
            }""")

  def testConvertCase(self):
    """Tests that a v1 case() construction converts properly."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=False):
        control_flow_v2_toggles.disable_control_flow_v2()
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=2.0)
        _ = control_flow_ops.case([(gen_math_ops.less(x, y), lambda: x)],
                                  default=lambda: y)
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["case/cond/Merge"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "y" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 2 }}}
            }
            node {name: "x/read" op: "Identity" input: "x"}
            node {name: "y/read" op: "Identity" input: "y"}
            node {name: "Less" op: "Less" input: "x/read" input: "y/read"}
            node {name: "case/cond/pred_id" op: "Identity" input: "Less"}
            node {
              name: "case/cond/Switch_1" op: "Switch"
              input: "case/cond/pred_id" input: "x/read"
            }
            node {
              name: "case/cond/Switch_2" op: "Switch"
              input: "case/cond/pred_id" input: "y/read"
            }
            node {
              name: "case/cond/Merge" op: "Merge"
              input: "case/cond/Switch_2" input: "case/cond/Switch_1:1"
              attr {key: "T" value {type: DT_FLOAT}}
            }""")

  def testConvertV2Case(self):
    """Tests that a v2 case() converts properly."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=False):
        control_flow_v2_toggles.enable_control_flow_v2()
        a = variable_scope.get_variable("a", initializer=2.0)
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=2.0)
        _ = control_flow_ops.case([(gen_math_ops.less(x, y), lambda: a)],
                                  default=lambda: y)
        control_flow_v2_toggles.disable_control_flow_v2()
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["case/cond"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {
              name: "x" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 1 }}}
            }
            node {
              name: "y" op: "Const"
              attr { key: "dtype" value { type: DT_FLOAT } }
              attr {
                key: "value"
                value { tensor { dtype: DT_FLOAT tensor_shape{} float_val: 2 }}}
            }
            node {name: "x/read" op: "Identity" input: "x"}
            node {name: "y/read" op: "Identity" input: "y"}
            node {name: "Less" op: "Less" input: "x/read" input: "y/read"}
            node {
              name: "case/cond" op: "StatelessIf"
              input: "Less" input: "a/read" input: "y/read"
              attr {key: "Tcond" value {type: DT_BOOL}}
              attr {key: "Tin" value {list {type: DT_FLOAT type: DT_FLOAT}}}
              attr {key: "Tout" value {list {type: DT_FLOAT}}}
            }
            library {
              function {
                signature {
                  name: "case_cond_false_frozen_0"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "y_read_0" type: DT_FLOAT}
                  output_arg {name: "y_read" type: DT_FLOAT}
                }
              }
              function {
                signature {
                  name: "case_cond_true_frozen_0"
                  input_arg {name: "a_read_0" type: DT_FLOAT}
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  output_arg {name: "a_read" type: DT_FLOAT}
                }
              }
            }""")

  def testConvertV2ResourceCase(self):
    """Tests that a v2 case() with resource variables converts properly."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        control_flow_v2_toggles.enable_control_flow_v2()
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=2.0)
        _ = control_flow_ops.case([(gen_math_ops.less(x, y), lambda: x)],
                                  default=lambda: y)
        control_flow_v2_toggles.disable_control_flow_v2()
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess, variable_graph_def, ["case/cond"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {name: "x" op: "Const"}
            node {name: "y" op: "Const"}
            node {
              name: "case/cond" op: "If" input: "Less" input: "x" input: "y"
              attr {key: "Tcond" value {type: DT_BOOL}}
              attr {key: "Tin" value {list {type: DT_FLOAT type: DT_FLOAT}}}
              attr {key: "Tout" value {list {type: DT_FLOAT}}}
            }
            library {
              function {
                signature {
                  name: "case_cond_false_frozen_0"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "readvariableop_y" type: DT_FLOAT}
                  output_arg {name: "readvariableop" type: DT_FLOAT}
                }
              }
              function {
                signature {
                  name: "case_cond_true_frozen_0"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "readvariableop_x" type: DT_FLOAT}
                  output_arg {name: "readvariableop" type: DT_FLOAT}
                }
              }
            }""")

  def testConvertV2UnconvertedResourceNestedCase(self):
    """Tests unconverted variable propagation through nested functions."""
    with ops.Graph().as_default():
      with variable_scope.variable_scope("", use_resource=True):
        control_flow_v2_toggles.enable_control_flow_v2()
        x = variable_scope.get_variable("x", initializer=1.0)
        y = variable_scope.get_variable("y", initializer=2.0)
        z = variable_scope.get_variable("z", initializer=3.0)
        # pylint: disable=g-long-lambda
        _ = control_flow_ops.case(
            [(gen_math_ops.less(x, y), lambda: x)],
            default=lambda: control_flow_ops.case(
                [(gen_math_ops.less(z, y), lambda: z)], default=lambda: y))
        # pylint: enable=g-long-lambda
        control_flow_v2_toggles.disable_control_flow_v2()
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        variable_graph_def = sess.graph.as_graph_def()
        constant_graph_def = (
            convert_to_constants
            .convert_variables_to_constants_from_session_graph(
                sess,
                variable_graph_def, ["case/cond"],
                variable_names_denylist=["y"]))
        self._assertGraphContains(
            constant_graph_def, """
            node {name: "x" op: "Const"}
            node {name: "y" op: "VarHandleOp"}
            node {name: "z" op: "Const"}

            node {name: "Less/ReadVariableOp" op: "Identity" input: "x"}
            node {name: "Less/ReadVariableOp_1" op: "ReadVariableOp" input: "y"}

            node {
              name: "case/cond" op: "If"
              input: "x" input: "z" input: "y"
              attr {
                key: "Tin"
                value {list
                  {type: DT_FLOAT type: DT_FLOAT type: DT_RESOURCE}}}
              attr {
                key: "_read_only_resource_inputs"
                value {list {i: 1 i: 2 i: 3}}}
              attr {key: "then_branch"
                    value {func {name: "case_cond_true_frozen_0"}}}
              attr {key: "else_branch"
                    value {func {name: "case_cond_false_frozen_0"}}}
              attr {key: "output_shapes" value {list {shape {}}}}
            }
            library {
              function {
                signature {
                  name: "case_cond_true_frozen_0"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "placeholder_1" type: DT_RESOURCE}
                  input_arg {name: "readvariableop_x" type: DT_FLOAT}
                  output_arg {name: "readvariableop" type: DT_FLOAT}
                  is_stateful: true
                }

                node_def {name: "ReadVariableOp" op: "Identity"
                  input: "readvariableop_x"}}

              function {
                signature {
                  name: "case_cond_false_frozen_0"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "less_readvariableop_1_y" type: DT_RESOURCE}
                  input_arg {name: "less_readvariableop_z" type: DT_FLOAT}
                  output_arg {name: "case_cond_identity" type: DT_FLOAT}
                  is_stateful: true
                }

                node_def {name: "Less/ReadVariableOp_1" op: "ReadVariableOp"
                  input: "less_readvariableop_1_y"}

                node_def {name: "Less/ReadVariableOp" op: "Identity"
                  input: "less_readvariableop_z"}

                node_def {name: "case/cond" op: "If"
                  input: "less_readvariableop_z"
                  input: "less_readvariableop_1_y"
                  attr {
                    key: "Tin"
                    value {list {type: DT_FLOAT type: DT_RESOURCE}}}
                  attr {key: "then_branch"
                        value {func {name: "case_cond_true_frozen_1"}}}
                  attr {key: "else_branch"
                        value {func {name: "case_cond_false_frozen_1"}}}
                  attr {
                    key: "_read_only_resource_inputs"
                    value {list {i: 1 i: 2}}}}}

              function {
                signature {
                  name: "case_cond_false_frozen_1"
                  input_arg {name: "placeholder" type: DT_FLOAT}
                  input_arg {name: "readvariableop_y" type: DT_RESOURCE}
                  output_arg {name: "readvariableop" type: DT_FLOAT}
                  is_stateful: true
                }

                node_def {name: "ReadVariableOp" op: "ReadVariableOp"
                  input: "readvariableop_y"}}

              function {
                signature {
                  name: "case_cond_true_frozen_1"
                  input_arg {name: "placeholder" type: DT_RESOURCE}
                  input_arg {name: "readvariableop_z" type: DT_FLOAT}
                  output_arg {name: "readvariableop" type: DT_FLOAT}
                  is_stateful: true
                }

                node_def {name: "ReadVariableOp" op: "Identity"
                  input: "readvariableop_z"}}}""")

  def _addNoinlineAttributeToFunction(self, saved_model_dir, func_name):
    saved_model_proto = loader_impl.parse_saved_model(saved_model_dir)
    new_saved_model = saved_model_pb2.SavedModel()
    new_saved_model.CopyFrom(saved_model_proto)
    new_meta_graph_def = new_saved_model.meta_graphs[0]
    prefix_len = len("__inference_")
    for func_def in new_meta_graph_def.graph_def.library.function:
      func_name_without_prefix = func_def.signature.name[prefix_len:]
      if func_name_without_prefix.startswith(func_name):
        func_def.attr["_noinline"].CopyFrom(attr_value_pb2.AttrValue(b=True))
    old_saved_model_file = os.path.join(saved_model_dir,
                                        constants.SAVED_MODEL_FILENAME_PB)
    if os.path.exists(old_saved_model_file):
      os.remove(old_saved_model_file)
    path = os.path.join(
        compat.as_bytes(saved_model_dir),
        compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
    file_io.write_string_to_file(
        path, new_saved_model.SerializeToString(deterministic=True))

  @test_util.run_v2_only
  def testVariableModelWithFunctionAndFunctionInliningDisabled(self):
    """Test a model with Variables and disable function inlining."""

    class BasicModel:

      def __init__(self):
        self.v1 = None
        self.v2 = variables.Variable(2.)

      @def_function.function(input_signature=[
          tensor_spec.TensorSpec(shape=[1], dtype=dtypes.float32)
      ])
      def add_all(self, x):
        if self.v1 is None:
          self.v1 = variables.Variable(3.)
        return x + self.v1 + self.v2

      def run(self, x):
        y = self.add_all(x)
        return y

    save_dir = os.path.join(self.get_temp_dir(), "frozen_saved_model")
    with ops.Graph().as_default():
      model = BasicModel()
      a = array_ops.placeholder(dtypes.float32, shape=[1])
      b = model.run(a)
      with session_lib.Session() as sess:
        sess.run(variables.global_variables_initializer())
        simple_save.simple_save(sess, save_dir, {"myinput": a}, {"myoutput": b})

    # Add _noinline to the SavedModel.
    self._addNoinlineAttributeToFunction(
        saved_model_dir=save_dir, func_name="add_all")

    saved_model = load(save_dir)
    func = saved_model.signatures["serving_default"]
    frozen_func = convert_to_constants.convert_variables_to_constants_v2(func)
    constant_graph_def = frozen_func.graph.as_graph_def()
    self._ensure_no_variables_in_graph(constant_graph_def)


if __name__ == "__main__":
  test.main()
