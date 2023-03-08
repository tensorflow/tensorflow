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
"""Tests for control_flow_ops.py."""

import collections
import itertools
import time

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.autograph.lang import directives
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_case
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util_v2
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.training import momentum
from tensorflow.python.util import nest

TestTuple = collections.namedtuple("TestTuple", "a b")
SingletonTestTuple = collections.namedtuple("SingletonTestTuple", "a")


class GroupTestCase(test_util.TensorFlowTestCase):

  def _StripNode(self, nd):
    snode = node_def_pb2.NodeDef(name=nd.name, op=nd.op, input=nd.input)
    if nd.device:
      snode.device = nd.device
    return snode

  def _StripGraph(self, gd):
    """Copy gd keeping only, node.name, node.op, node.input, and node.device."""
    return graph_pb2.GraphDef(node=[self._StripNode(nd) for nd in gd.node])

  def testGroup_NoDevices(self):
    with ops.Graph().as_default() as g:
      a = constant_op.constant(0, name="a")
      b = constant_op.constant(0, name="b")
      c = constant_op.constant(0, name="c")
      control_flow_ops.group(a.op, b.op, c.op, name="root")
    gd = g.as_graph_def()
    self.assertProtoEquals(
        """
      node { name: "a" op: "Const"}
      node { name: "b" op: "Const"}
      node { name: "c" op: "Const"}
      node { name: "root" op: "NoOp" input: "^a" input: "^b" input: "^c" }
    """, self._StripGraph(gd))

  def testGroup_OneDevice(self):
    with ops.Graph().as_default() as g:
      with g.device("/task:0"):
        a = constant_op.constant(0, name="a")
        b = constant_op.constant(0, name="b")
      control_flow_ops.group(a.op, b.op, name="root")
    gd = g.as_graph_def()
    self.assertProtoEquals(
        """
      node { name: "a" op: "Const" device: "/task:0" }
      node { name: "b" op: "Const" device: "/task:0" }
      node { name: "root" op: "NoOp" input: "^a" input: "^b" device: "/task:0" }
    """, self._StripGraph(gd))

  def testGroup_MultiDevice(self):
    with ops.Graph().as_default() as g:
      with g.device("/task:0"):
        a = constant_op.constant(0, name="a")
        b = constant_op.constant(0, name="b")
      with g.device("/task:1"):
        c = constant_op.constant(0, name="c")
        d = constant_op.constant(0, name="d")
      with g.device("/task:2"):
        control_flow_ops.group(a.op, b.op, c.op, d.op, name="root")
    gd = g.as_graph_def()
    self.assertProtoEquals(
        """
      node { name: "a" op: "Const" device: "/task:0"}
      node { name: "b" op: "Const" device: "/task:0"}
      node { name: "c" op: "Const" device: "/task:1"}
      node { name: "d" op: "Const" device: "/task:1"}
      node { name: "root/NoOp" op: "NoOp" input: "^a" input: "^b"
             device: "/task:0" }
      node { name: "root/NoOp_1" op: "NoOp" input: "^c" input: "^d"
             device: "/task:1" }
      node { name: "root" op: "NoOp" input: "^root/NoOp" input: "^root/NoOp_1"
             device: "/task:2" }
    """, self._StripGraph(gd))

  def testPassingList(self):
    with ops.Graph().as_default() as g:
      a = constant_op.constant(0, name="a")
      b = constant_op.constant(0, name="b")
      control_flow_ops.group([a.op, b.op], name="root")
    gd = g.as_graph_def()
    self.assertProtoEquals(
        """
      node { name: "a" op: "Const"}
      node { name: "b" op: "Const"}
      node { name: "root" op: "NoOp" input: "^a" input: "^b" }
    """, self._StripGraph(gd))

  @test_util.run_deprecated_v1
  def testPassingNonTensors(self):
    with self.assertRaises(TypeError):
      control_flow_ops.group(1, 2)


class ShapeTestCase(test_util.TensorFlowTestCase):

  def testShape(self):
    tensor = constant_op.constant([1.0, 2.0])
    self.assertEqual([2], tensor.get_shape())
    self.assertEqual([2],
                     control_flow_ops.with_dependencies(
                         [constant_op.constant(1.0)], tensor).get_shape())


class WithDependenciesTestCase(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testTupleDependencies(self):
    counter = variable_scope.get_variable(
        "my_counter", shape=[], initializer=init_ops.zeros_initializer())
    increment_counter = state_ops.assign_add(counter, 1)
    const_with_dep = control_flow_ops.with_dependencies(
        (increment_counter, constant_op.constant(42)), constant_op.constant(7))

    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(0, self.evaluate(counter))
    self.assertEqual(7, self.evaluate(const_with_dep))
    self.assertEqual(1, self.evaluate(counter))

  @test_util.run_deprecated_v1
  def testListDependencies(self):
    counter = variable_scope.get_variable(
        "my_counter", shape=[], initializer=init_ops.zeros_initializer())
    increment_counter = state_ops.assign_add(counter, 1)
    const_with_dep = control_flow_ops.with_dependencies(
        [increment_counter, constant_op.constant(42)], constant_op.constant(7))

    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(0, self.evaluate(counter))
    self.assertEqual(7, self.evaluate(const_with_dep))
    self.assertEqual(1, self.evaluate(counter))


class SwitchTestCase(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testIndexedSlicesWithDenseShape(self):
    with self.cached_session():
      data = indexed_slices.IndexedSlices(
          constant_op.constant([1, 2, 3]),
          constant_op.constant([0, 1, 2]),
          dense_shape=constant_op.constant([3]))
      zero = constant_op.constant(0)
      one = constant_op.constant(1)
      less_op = math_ops.less(zero, one)
      _, switch_true = control_flow_ops.switch(data, less_op)
      self.assertAllEqual([1, 2, 3], switch_true.values)
      self.assertAllEqual([0, 1, 2], switch_true.indices)

  @test_util.run_deprecated_v1
  def testIndexedSlicesGradient(self):
    embedding_matrix = variable_scope.get_variable(
        "embedding_matrix", [5, 5],
        initializer=init_ops.random_normal_initializer())

    def cond(it, _):
      return it < 5

    def body(it, cost):
      embedding = embedding_ops.embedding_lookup(embedding_matrix + 0.0, [0])
      cost += math_ops.reduce_sum(embedding)
      return it + 1, cost

    _, cost = while_loop.while_loop(
        cond, body, [constant_op.constant(0),
                     constant_op.constant(0.0)])
    optimizer = momentum.MomentumOptimizer(0.1, 0.9)
    train_op = optimizer.minimize(cost)
    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      for _ in range(10):
        self.evaluate([train_op])

  def testResourceReadInLoop(self):
    embedding_matrix = variable_scope.get_variable(
        "embedding_matrix", initializer=[[2.0], [3.0]], use_resource=True)

    def cond(it, _):
      return it < 5

    def body(it, cost):
      embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
      cost += math_ops.reduce_sum(embedding)
      return it + 1, cost

    _, cost = while_loop.while_loop(
        cond, body, [constant_op.constant(0),
                     constant_op.constant(0.0)])
    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(10.0, self.evaluate(cost))

  def doTestIndexedSlicesGradientInCondInWhileLoop(self, use_resource=False):
    embedding_matrix = variable_scope.get_variable(
        "embedding_matrix", [5, 5],
        initializer=init_ops.random_normal_initializer(),
        use_resource=use_resource)

    def cond(it, _):
      return it < 5

    def body(it, cost):
      embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
      cost = control_flow_ops.cond(
          math_ops.equal(it, 3), lambda: math_ops.square(cost),
          (lambda: cost + math_ops.reduce_sum(embedding)))
      return it + 1, cost

      _, cost = while_loop.while_loop(
          cond, body, [constant_op.constant(0),
                       constant_op.constant(0.0)])

      dynamic_grads = gradients_impl.gradients(cost, [embedding_matrix])[0]
      dynamic_grads = math_ops.segment_sum(dynamic_grads.values,
                                           dynamic_grads.indices)

      embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
      static = math_ops.square(
          math_ops.reduce_sum(embedding) + math_ops.reduce_sum(embedding) +
          math_ops.reduce_sum(embedding)) + math_ops.reduce_sum(embedding)
      static_grads = gradients_impl.gradients(static, [embedding_matrix])[0]
      static_grads = math_ops.segment_sum(static_grads.values,
                                          static_grads.indices)

      with self.cached_session():
        self.evaluate(variables.global_variables_initializer())
        self.assertAllEqual(*self.evaluate([static_grads, dynamic_grads]))

  def testIndexedSlicesGradientInCondInWhileLoop(self):
    self.doTestIndexedSlicesGradientInCondInWhileLoop(use_resource=False)

  def testIndexedSlicesGradientInCondInWhileLoopResource(self):
    self.doTestIndexedSlicesGradientInCondInWhileLoop(use_resource=True)

  @test_util.run_v1_only("b/120545219")
  def testIndexedSlicesWithShapeGradientInWhileLoop(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session() as sess:
        num_steps = 9

        inputs = array_ops.placeholder(dtype=dtype, shape=[num_steps])
        initial_outputs = tensor_array_ops.TensorArray(
            dtype=dtype, size=num_steps)
        initial_i = constant_op.constant(0, dtype=dtypes.int32)

        def cond(i, _):
          return i < num_steps  # pylint: disable=cell-var-from-loop

        def body(i, outputs):
          x = array_ops.gather(inputs, i)  # pylint: disable=cell-var-from-loop
          outputs = outputs.write(i, x)
          return i + 1, outputs

        _, outputs = while_loop.while_loop(cond, body,
                                           [initial_i, initial_outputs])

        outputs = math_ops.reduce_sum(outputs.stack())
        r = gradients_impl.gradients([outputs], [inputs])[0]
        grad_wr_inputs = ops.convert_to_tensor(r)
        o, grad = sess.run([outputs, grad_wr_inputs],
                           feed_dict={inputs: [4, 6, 0, 7, 0, 0, 1, 2, 0]})
        self.assertEqual(o, 20)
        self.assertAllEqual(grad, [1] * num_steps)

  @test_util.run_v1_only("b/120545219")
  def testIndexedSlicesWithDynamicShapeGradientInWhileLoop(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session() as sess:
        inputs = array_ops.placeholder(dtype=dtype)
        initial_outputs = tensor_array_ops.TensorArray(
            dtype=dtype, dynamic_size=True, size=1)
        initial_i = constant_op.constant(0, dtype=dtypes.int32)

        def cond(i, _):
          return i < array_ops.size(inputs)  # pylint: disable=cell-var-from-loop

        def body(i, outputs):
          x = array_ops.gather(inputs, i)  # pylint: disable=cell-var-from-loop
          outputs = outputs.write(i, x)
          return i + 1, outputs

        _, outputs = while_loop.while_loop(cond, body,
                                           [initial_i, initial_outputs])

        outputs = math_ops.reduce_sum(outputs.stack())
        r = gradients_impl.gradients([outputs], [inputs])[0]
        grad_wr_inputs = ops.convert_to_tensor(r)
        o, grad = sess.run([outputs, grad_wr_inputs],
                           feed_dict={inputs: [1, 3, 2]})
        self.assertEqual(o, 6)
        self.assertAllEqual(grad, [1] * 3)

  @test_util.run_deprecated_v1
  def testGradientThroughSingleBranchOutsideOfContext(self):
    x = constant_op.constant(2.)
    s = constant_op.constant(True)
    x_false, x_true = control_flow_ops.switch(x, s)
    grad_x_true = gradients_impl.gradients(x_true, x)[0]
    grad_x_false = gradients_impl.gradients(x_false, x)[0]
    self.assertEqual(self.evaluate(grad_x_true), 1.)
    self.assertEqual(self.evaluate(grad_x_false), 0.)


class CondTest(test_util.TensorFlowTestCase):

  def testCondTrue(self):
    x = constant_op.constant(2)
    y = constant_op.constant(5)
    z = control_flow_ops.cond(
        math_ops.less(x, y), lambda: math_ops.multiply(x, 17),
        lambda: math_ops.add(y, 23))
    self.assertEqual(self.evaluate(z), 34)

  def testCondFalse(self):
    x = constant_op.constant(2)
    y = constant_op.constant(1)
    z = control_flow_ops.cond(
        math_ops.less(x, y), lambda: math_ops.multiply(x, 17),
        lambda: math_ops.add(y, 23))
    self.assertEqual(self.evaluate(z), 24)

  def testCondTrueLegacy(self):
    x = constant_op.constant(2)
    y = constant_op.constant(5)
    z = control_flow_ops.cond(
        math_ops.less(x, y),
        fn1=lambda: math_ops.multiply(x, 17),
        fn2=lambda: math_ops.add(y, 23))
    self.assertEqual(self.evaluate(z), 34)

  def testCondFalseLegacy(self):
    x = constant_op.constant(2)
    y = constant_op.constant(1)
    z = control_flow_ops.cond(
        math_ops.less(x, y),
        fn1=lambda: math_ops.multiply(x, 17),
        fn2=lambda: math_ops.add(y, 23))
    self.assertEqual(self.evaluate(z), 24)

  @test_util.run_v1_only("Exercises Ref variables")
  def testCondModifyBoolPred(self):
    # We want to use the GPU here because we want to ensure that we can update
    # a boolean ref variable on the GPU.
    with test_util.use_gpu():
      bool_var = variable_scope.get_variable(
          "bool_var", dtype=dtypes.bool, initializer=True)
      cond_on_bool_var = control_flow_ops.cond(
          pred=bool_var,
          true_fn=lambda: state_ops.assign(bool_var, False),
          false_fn=lambda: True)
      self.evaluate(bool_var.initializer)
      self.assertEqual(self.evaluate(cond_on_bool_var), False)
      self.assertEqual(self.evaluate(cond_on_bool_var), True)

  def testCondMissingArg1(self):
    x = constant_op.constant(1)
    with self.assertRaises(TypeError):
      control_flow_ops.cond(True, false_fn=lambda: x)

  def testCondMissingArg2(self):
    x = constant_op.constant(1)
    with self.assertRaises(TypeError):
      control_flow_ops.cond(True, lambda: x)

  def testCondDuplicateArg1(self):
    x = constant_op.constant(1)
    with self.assertRaises(TypeError):
      control_flow_ops.cond(True, lambda: x, lambda: x, fn1=lambda: x)

  def testCondDuplicateArg2(self):
    x = constant_op.constant(1)
    with self.assertRaises(TypeError):
      control_flow_ops.cond(True, lambda: x, lambda: x, fn2=lambda: x)

  @test_util.enable_control_flow_v2
  @test_util.run_in_graph_and_eager_modes
  def testCond_gradient(self):
    true_in, false_in = array_ops.constant(1.), array_ops.constant(5.)
    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(true_in)
      tape.watch(false_in)
      cond_true = control_flow_ops.cond(
          array_ops.constant(True), lambda: true_in**2., lambda: false_in**2.)
      cond_false = control_flow_ops.cond(
          array_ops.constant(False), lambda: true_in**2., lambda: false_in**2.)
    grads_true = tape.gradient(
        cond_true, [true_in, false_in], output_gradients=3.)
    grads_false = tape.gradient(
        cond_false, [true_in, false_in], output_gradients=3.)
    self.assertEqual(3. * 2. * 1., self.evaluate(grads_true[0]))
    self.assertEqual(None if context.executing_eagerly() else 0.,
                     self.evaluate(grads_true[1]))
    self.assertEqual(3. * 2. * 5., self.evaluate(grads_false[1]))
    self.assertEqual(None if context.executing_eagerly() else 0.,
                     self.evaluate(grads_false[0]))

  def testCondWithGroupAndSummaries(self):
    with ops.Graph().as_default():
      writer = summary_ops_v2.create_file_writer(self.get_temp_dir())
      with writer.as_default(), summary_ops_v2.always_record_summaries():
        op = control_flow_ops.cond(
            constant_op.constant(1) >= 0,
            lambda: control_flow_ops.group(summary_ops_v2.scalar("loss", 0.2)),
            control_flow_ops.no_op)
        self.evaluate(variables.global_variables_initializer())
        self.evaluate(summary_ops_v2.summary_writer_initializer_op())
        self.assertEqual(self.evaluate(op), True)


class ContextTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testCondContext(self):
    with self.cached_session() as sess:
      x = constant_op.constant(2)
      y = constant_op.constant(5)
      control_flow_ops.cond(
          math_ops.less(x, y), lambda: math_ops.multiply(x, 17),
          lambda: math_ops.add(y, 23))
      for op in sess.graph.get_operations():
        c = op._get_control_flow_context()
        if c:
          self.assertProtoEquals(
              c.to_proto(),
              control_flow_ops.CondContext.from_proto(c.to_proto()).to_proto())

  def _testWhileContextHelper(self, maximum_iterations=None):
    with self.cached_session() as sess:
      i = constant_op.constant(0)
      c = lambda i: math_ops.less(i, 10)
      b = lambda i: math_ops.add(i, 1)
      while_loop.while_loop(c, b, [i], maximum_iterations=maximum_iterations)
      for op in sess.graph.get_operations():
        control_flow_context = op._get_control_flow_context()
        if control_flow_context:
          self.assertProtoEquals(
              control_flow_context.to_proto(),
              control_flow_ops.WhileContext.from_proto(
                  control_flow_context.to_proto()).to_proto())

  @test_util.run_deprecated_v1
  def testWhileContext(self):
    self._testWhileContextHelper()

  @test_util.run_deprecated_v1
  def testWhileContextWithMaximumIterations(self):
    self._testWhileContextHelper(maximum_iterations=10)

  @test_util.run_deprecated_v1
  def testControlContextImportScope(self):

    class NoABCControlFlowContext(control_flow_ops.ControlFlowContext):
      """A noop wrapper around `ControlFlowContext`.

      `ControlFlowContext` is an ABC and therefore cannot be instantiated.
      """

      # pylint: disable=useless-super-delegation

      def to_control_flow_context_def(self, context_def, export_scope=None):
        super(NoABCControlFlowContext,
              self).to_control_flow_context_def(context_def, export_scope)

    with self.cached_session():
      constant_op.constant(0, name="a")
      constant_op.constant(2, name="test_scope/a")
      b1 = constant_op.constant(1, name="b")
      b2 = constant_op.constant(3, name="test_scope/b")

      c = NoABCControlFlowContext()
      c._values = ["a", "b"]
      c._external_values = {"a": b1}

      c_with_scope = NoABCControlFlowContext(
          values_def=c._to_values_def(), import_scope="test_scope")

      # _values and _external_values should be have scope prepended.
      self.assertEqual(c_with_scope._values,
                       set(["test_scope/a", "test_scope/b"]))
      self.assertEqual(c_with_scope._external_values, {"test_scope/a": b2})

      # Calling _to_proto() with export_scope should remove "test_scope".
      self.assertProtoEquals(
          c._to_values_def(),
          c_with_scope._to_values_def(export_scope="test_scope"))


def _get_nested_shape(nested):

  def _get_shape(tensor):
    if isinstance(tensor, tensor_array_ops.TensorArray):
      return tensor_array_ops.TensorArray
    elif isinstance(tensor, indexed_slices.IndexedSlices):
      return tensor.dense_shape
    else:
      return tensor.get_shape()

  return nest.map_structure(_get_shape, nested)


def _create_tensor_array(size, shape):
  ta = tensor_array_ops.TensorArray(
      dtype=dtypes.float32, size=size, clear_after_read=False)
  for i in range(size):
    ta = ta.write(i, array_ops.zeros(shape))
  return ta


def _raw_nested_shape(nested_shape):

  def _raw_shape(shape):
    if isinstance(shape, tensor_shape.TensorShape) and shape.ndims is not None:
      return [x.value for x in shape.dims]
    else:
      return None

  return nest.map_structure(_raw_shape, nested_shape)


# TODO(yori): Add tests for indexed slices.
class DataTypesTest(test_util.TensorFlowTestCase):

  def assertAllEqualNested(self, a, b):
    if isinstance(a, (list, tuple)):
      for entry_a, entry_b in zip(a, b):
        self.assertAllEqualNested(entry_a, entry_b)
    else:
      self.assertAllEqual(a, b)

  def _testShape(self, fn_true, fn_false, expected_shape, strict=False):
    condition = array_ops.placeholder(dtypes.bool)
    output_cond = control_flow_ops.cond(
        condition, fn_true, fn_false, strict=strict)
    self.assertEqual(
        _raw_nested_shape(_get_nested_shape(output_cond)),
        _raw_nested_shape(expected_shape))

    output_case = control_flow_case.case([(condition, fn_true)],
                                         fn_false,
                                         strict=strict)
    self.assertEqual(
        _raw_nested_shape(_get_nested_shape(output_case)),
        _raw_nested_shape(expected_shape))

  def _testReturnValues(self,
                        fn_true,
                        fn_false,
                        expected_value_true,
                        expected_value_false,
                        strict=False,
                        check_cond=True,
                        feed_dict=None):
    if feed_dict is None:
      feed_dict = {}

    condition = array_ops.placeholder(dtypes.bool)
    output_cond = control_flow_ops.cond(
        condition, fn_true, fn_false, strict=strict)
    output_case = control_flow_case.case([(condition, fn_true)],
                                         fn_false,
                                         strict=strict)

    with self.cached_session() as sess:
      self.evaluate(variables.global_variables_initializer())
      true_feed_dict = {condition: True}
      true_feed_dict.update(feed_dict)
      result_cond, result_case = sess.run([output_cond, output_case],
                                          feed_dict=true_feed_dict)
      self.assertAllEqualNested(result_cond, expected_value_true)
      if check_cond:
        self.assertAllEqualNested(result_case, expected_value_true)
      false_feed_dict = {condition: False}
      false_feed_dict.update(feed_dict)
      result_cond, result_case = sess.run([output_cond, output_case],
                                          feed_dict=false_feed_dict)
      self.assertAllEqualNested(result_cond, expected_value_false)
      if check_cond:
        self.assertAllEqualNested(result_case, expected_value_false)

  @test_util.run_deprecated_v1
  def test_int(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: 1
    fn_false = lambda: 2
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 1, 2)
    self._testShape(fn_true, fn_false, shape, strict=True)
    self._testReturnValues(fn_true, fn_false, 1, 2, strict=True)

  @test_util.run_deprecated_v1
  def test_float(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: 1.0
    fn_false = lambda: 2.0
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 1.0, 2.0)

  @test_util.run_deprecated_v1
  def test_noop(self):
    shape = tensor_shape.TensorShape(None)
    self._testShape(control_flow_ops.no_op, control_flow_ops.no_op, shape)
    self._testReturnValues(
        control_flow_ops.no_op,
        control_flow_ops.no_op,
        True,
        False,
        check_cond=False)

  @test_util.run_deprecated_v1
  def test_string(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: "abc"
    fn_false = lambda: "xyz"
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, b"abc", b"xyz")

  @test_util.run_v1_only("b/138741991")
  def test_variable(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: variables.Variable(3.0)
    fn_false = lambda: variables.Variable(4.0)
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 3.0, 4.0)

  @test_util.run_v1_only("b/120553181")
  def test_none(self):
    fn_none = lambda: None
    fn_tensor = lambda: constant_op.constant(1)

    with self.assertRaises(ValueError):
      control_flow_ops.cond(constant_op.constant(True), fn_none, fn_tensor)

    with self.assertRaises(ValueError):
      control_flow_ops.cond(constant_op.constant(True), fn_tensor, fn_none)

  @test_util.run_deprecated_v1
  def test_tensors(self):

    def _build_true_branch(dtype):

      def _build():
        return (array_ops.zeros([2, 2],
                                dtype=dtype), array_ops.ones([3, 3],
                                                             dtype=dtype))

      return _build

    def _build_false_branch(dtype):

      def _build():
        return (array_ops.ones([2, 2],
                               dtype=dtype), array_ops.zeros([3, 3],
                                                             dtype=dtype))

      return _build

    for dtype in (dtypes.float16, dtypes.int8, dtypes.int32, dtypes.uint8):
      shape = (tensor_shape.TensorShape([2,
                                         2]), tensor_shape.TensorShape([3, 3]))
      fn_true = _build_true_branch(dtype)
      fn_false = _build_false_branch(dtype)
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false,
                             (np.zeros([2, 2]), np.ones([3, 3])),
                             (np.ones([2, 2]), np.zeros([3, 3])))

  @test_util.run_deprecated_v1
  def test_tensors_unknown_shape(self):

    def _build_true_branch(dtype):
      tensor = array_ops.placeholder(dtype=dtype, shape=None)

      def _build():
        return tensor

      return _build, tensor

    def _build_false_branch(dtype):
      tensor = array_ops.placeholder(dtype=dtype, shape=None)

      def _build():
        return tensor

      return _build, tensor

    for dtype in (dtypes.float16, dtypes.int8, dtypes.int32, dtypes.uint8):
      shape = tensor_shape.TensorShape(None)
      fn_true, true_tensor = _build_true_branch(dtype)
      fn_false, false_tensor = _build_false_branch(dtype)
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(
          fn_true,
          fn_false,
          np.zeros([2, 2]),
          np.ones([2, 2]),
          feed_dict={
              true_tensor: np.zeros([2, 2]),
              false_tensor: np.ones([2, 2])
          })

  @test_util.run_deprecated_v1
  def test_sparse_tensors(self):
    shape = tensor_shape.TensorShape([3, 4])

    def true_fn():
      return [
          sparse_tensor.SparseTensor(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
      ]

    def false_fn():
      return [
          sparse_tensor.SparseTensor(
              indices=[[0, 0], [2, 1]], values=[3, 4], dense_shape=[3, 4])
      ]

    value1 = sparse_tensor.SparseTensorValue(
        indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
    value2 = sparse_tensor.SparseTensorValue(
        indices=[[0, 0], [2, 1]], values=[3, 4], dense_shape=[3, 4])
    # Non-strict cond is only available in v1
    if not tf2.enabled():
      self._testShape(true_fn, false_fn, shape)
      self._testReturnValues(true_fn, false_fn, value1, value2)
    self._testShape(true_fn, false_fn, [shape], strict=True)
    self._testReturnValues(true_fn, false_fn, [value1], [value2], strict=True)

  @test_util.run_deprecated_v1
  def test_tensors_with_partially_specified_shapes(self):

    def _build_branch(dtype, shape):
      a = array_ops.placeholder(dtype=dtype, shape=shape[0])
      b = array_ops.placeholder(dtype=dtype, shape=shape[1])
      c = array_ops.placeholder(dtype=dtype, shape=shape[2])

      def _build():
        return a, b, c

      return _build, (a, b, c)

    for dtype in (dtypes.float16, dtypes.int8, dtypes.int32, dtypes.uint8):
      shape = (tensor_shape.TensorShape([None,
                                         2]), tensor_shape.TensorShape([None]),
               tensor_shape.TensorShape([3, None]))
      fn_true, true_tensors = _build_branch(dtype, shape)
      fn_false, false_tensors = _build_branch(dtype, shape)
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(
          fn_true,
          fn_false, (np.zeros([2, 2]), np.zeros(5), np.ones([3, 3])),
          (np.zeros([2, 2]), np.zeros(5), np.ones([3, 3])),
          feed_dict={
              true_tensors[0]: np.zeros([2, 2]),
              false_tensors[0]: np.zeros([2, 2]),
              true_tensors[1]: np.zeros([5]),
              false_tensors[1]: np.zeros([5]),
              true_tensors[2]: np.ones([3, 3]),
              false_tensors[2]: np.ones([3, 3])
          })

  @test_util.run_deprecated_v1
  def test_tensor_arrays(self):
    element_shape = tensor_shape.TensorShape([2])
    ta1 = _create_tensor_array(4, element_shape)
    ta2 = _create_tensor_array(4, element_shape)
    shape = tensor_array_ops.TensorArray
    fn_true = lambda: ta1
    fn_false = lambda: ta2
    self._testShape(fn_true, fn_false, shape)

  @test_util.run_deprecated_v1
  def test_tensor_array_reads(self):
    shape = tensor_shape.TensorShape([2])
    ta = _create_tensor_array(4, shape)
    fn_true = lambda: ta.read(0)
    fn_false = lambda: ta.read(1)
    self._testShape(fn_true, fn_false, shape)

  @test_util.run_v1_only("b/138741991")
  def test_list(self):
    shape = [
        tensor_shape.TensorShape([]),
        tensor_shape.TensorShape([]),
        tensor_shape.TensorShape([])
    ]
    fn_true = lambda: [constant_op.constant(1), 2, variables.Variable(3.0)]
    fn_false = lambda: [constant_op.constant(3), 4, variables.Variable(5.0)]
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, [1, 2, 3.0], [3, 4, 5.0])

  @test_util.run_v1_only("Non-strict cond is only available in v1")
  def test_non_strict(self):
    shape = tensor_shape.TensorShape([])
    fn_tensor = lambda: constant_op.constant(1)
    fn_list = lambda: [constant_op.constant(2)]
    fn_tuple = lambda: (constant_op.constant(3),)
    self._testShape(fn_tensor, fn_list, shape)
    self._testShape(fn_tensor, fn_tuple, shape)
    self._testShape(fn_list, fn_tuple, shape)
    self._testReturnValues(fn_tensor, fn_list, 1, 2)
    self._testReturnValues(fn_tensor, fn_tuple, 1, 3)
    self._testReturnValues(fn_list, fn_tuple, 2, 3)

  @test_util.run_v1_only("b/120553181")
  def test_singleton_strict(self):
    fn_tensor = lambda: constant_op.constant(1)
    fn_list = lambda: [constant_op.constant(2)]
    fn_tuple = lambda: (constant_op.constant(3),)

    with self.assertRaises(ValueError):
      control_flow_ops.cond(
          constant_op.constant(True), fn_tensor, fn_list, strict=True)

    with self.assertRaises(TypeError):
      control_flow_ops.cond(
          constant_op.constant(True), fn_list, fn_tuple, strict=True)

    with self.assertRaises(ValueError):
      control_flow_case.case([(constant_op.constant(True), fn_tensor)],
                             fn_list,
                             strict=True)

    with self.assertRaises(TypeError):
      control_flow_case.case([(constant_op.constant(True), fn_list)],
                             fn_tuple,
                             strict=True)

  @test_util.run_deprecated_v1
  def test_singleton_list(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: [constant_op.constant(1)]
    fn_false = lambda: [constant_op.constant(3)]
    # Non-strict cond is only available in v1
    if not tf2.enabled():
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false, 1, 3)
    self._testShape(fn_true, fn_false, [shape], strict=True)
    self._testReturnValues(fn_true, fn_false, [1], [3], strict=True)

  @test_util.run_deprecated_v1
  def test_singleton_tuple(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: (constant_op.constant(1),)
    fn_false = lambda: (constant_op.constant(3),)
    # Non-strict cond is only available in v1
    if not tf2.enabled():
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false, 1, 3)
    self._testShape(fn_true, fn_false, (shape,), strict=True)
    self._testReturnValues(fn_true, fn_false, (1,), (3,), strict=True)

  @test_util.run_deprecated_v1
  def test_singleton_namedtuple(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: SingletonTestTuple(constant_op.constant(1))
    fn_false = lambda: SingletonTestTuple(constant_op.constant(3))
    # Non-strict cond is only available in v1
    if not tf2.enabled():
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false, 1, 3)
    self._testShape(fn_true, fn_false, SingletonTestTuple(shape), strict=True)
    self._testReturnValues(
        fn_true,
        fn_false,
        SingletonTestTuple(1),
        SingletonTestTuple(3),
        strict=True)

  @test_util.run_deprecated_v1
  def test_tuple(self):
    shape = (tensor_shape.TensorShape([]), tensor_shape.TensorShape([]))
    fn_true = lambda: (constant_op.constant(1), 2)
    fn_false = lambda: (constant_op.constant(3), 4)
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, (1, 2), (3, 4))

  @test_util.run_deprecated_v1
  def test_namedtuple(self):
    shape = TestTuple(
        tensor_shape.TensorShape([]), tensor_shape.TensorShape([]))
    fn_true = lambda: TestTuple(constant_op.constant(1), 2)
    fn_false = lambda: TestTuple(constant_op.constant(3), 4)
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, TestTuple(1, 2), TestTuple(3, 4))

  @test_util.run_deprecated_v1
  def test_nested(self):
    shape = [
        tensor_shape.TensorShape([]),
        TestTuple(
            tensor_shape.TensorShape([]),
            [tensor_shape.TensorShape([]),
             tensor_shape.TensorShape([])]),
        tensor_shape.TensorShape([5, 5]),
        tensor_shape.TensorShape([])
    ]

    def true_fn():
      return [
          constant_op.constant(1),
          TestTuple(constant_op.constant(2), [3, 4]),
          array_ops.zeros([5, 5]), 6
      ]

    def false_fn():
      return [
          constant_op.constant(11),
          TestTuple(constant_op.constant(12), [13, 14]),
          array_ops.ones([5, 5]), 16
      ]

    self._testShape(true_fn, false_fn, shape)
    self._testReturnValues(
        true_fn, false_fn,
        [1, TestTuple(2, [3, 4]), np.zeros([5, 5]), 6],
        [11, TestTuple(12, [13, 14]),
         np.ones([5, 5]), 16])

  @test_util.run_deprecated_v1
  def test_cond_inside_while_loop(self):

    def body(i, matrix):
      result_tuple, unused_matrix = control_flow_ops.cond(
          constant_op.constant(True), lambda:
          (TestTuple(matrix * 2, matrix * 4), matrix), lambda:
          (TestTuple(matrix * 4, matrix * 2), matrix))
      return [i + 1, result_tuple.a]

    iteration, matrix = while_loop.while_loop(
        lambda i, matrix: i < 10,
        body,
        loop_vars=[constant_op.constant(0),
                   array_ops.ones([2, 2])])

    self.assertEqual(iteration.get_shape(), tensor_shape.TensorShape([]))
    self.assertEqual(matrix.get_shape(), tensor_shape.TensorShape([2, 2]))


@test_util.run_all_in_graph_and_eager_modes
class IndexedCaseTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def make_name(self):
    name = self.id().split(".")[-1].replace("(", "_").replace(")", "")
    return name.replace(" ", "_")

  def disabled_testCase_ticklesGpuVsHostMemoryIssueWithInt32(self):
    nbranches = 5

    def make_func(bi):
      return lambda: array_ops.constant(bi * 10, name="br{}_out".format(bi))

    branches = [(i, make_func(i)) for i in range(nbranches)]
    for bi in range(nbranches):
      branch_index = array_ops.placeholder_with_default(bi, [])
      case_out = control_flow_ops.switch_case(branch_index, branches)
      self.assertEqual(bi * 10, self.evaluate(case_out))

  @parameterized.parameters((0,), (2,), (3,))
  def testCase(self, bi):
    nbranches = 5

    def make_func(bi):
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = [(i, make_func(i)) for i in range(nbranches)]
    branch_index = array_ops.placeholder_with_default(bi, [])
    case_out = control_flow_ops.switch_case(
        branch_index, branches, name=self.make_name())
    self.assertEqual(bi * 10., self.evaluate(case_out))

  @parameterized.parameters((-1,), (2,), (4,), (5,), (6,))
  def testCase_withDefault(self, bi):
    nbranches = 5

    def make_func(bi):
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = [(i, make_func(i)) for i in range(nbranches)]
    branch_index = array_ops.placeholder_with_default(bi, [])
    case_out = control_flow_ops.switch_case(
        branch_index, branches, default=make_func(6), name=self.make_name())
    if bi < 0 or bi >= nbranches:
      expected = 60.
    else:
      expected = bi * 10.
    self.assertEqual(expected, self.evaluate(case_out))

  @parameterized.parameters((-1,), (0,), (3,), (5,))
  def testCase_dictWithDefault(self, bi):
    nbranches = 5

    def make_func(bi):
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = {i: make_func(i) for i in range(nbranches)}
    branch_index = array_ops.placeholder_with_default(bi, [])
    case_out = control_flow_ops.switch_case(
        branch_index, branches, default=make_func(6), name=self.make_name())
    if bi < 0 or bi >= nbranches:
      expected = 60.
    else:
      expected = bi * 10.
    self.assertEqual(expected, self.evaluate(case_out))

  @parameterized.parameters((-1,), (1,), (4,), (5,))
  def testCase_gradient_disable_lowering(self, bi):
    self._testCase_gradient(True, bi)

  @parameterized.parameters((-1,), (1,), (4,), (5,))
  def testCase_gradient_enable_lowering(self, bi):
    self._testCase_gradient(False, bi)

  def _testCase_gradient(self, disable_lowering, bi):
    default_lowering = control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE
    control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE = disable_lowering
    nbranches = 5
    inputs = [
        array_ops.constant(float(bi), name="br{}_in".format(bi))
        for bi in range(nbranches)
    ]

    def make_func(bi):
      return lambda: inputs[bi]**2.

    branches = {bi: make_func(bi) for bi in range(nbranches)}

    branch_index = array_ops.placeholder_with_default(bi, [])
    with backprop.GradientTape() as tape:
      for x in inputs:
        tape.watch(x)
      case_out = control_flow_ops.switch_case(branch_index, branches)
    out_grad = 3.
    actual_grads = tape.gradient(case_out, inputs, output_gradients=out_grad)
    expected_grads = [None if context.executing_eagerly() else 0.] * nbranches
    used_branch_idx = nbranches - 1 if bi < 0 or bi >= nbranches - 1 else bi
    expected_grads[used_branch_idx] = out_grad * 2. * used_branch_idx
    self.assertEqual(len(expected_grads), len(actual_grads))
    for expected, actual in zip(expected_grads, actual_grads):
      self.assertEqual(expected, self.evaluate(actual))
    # reset to default value
    control_flow_util_v2._DISABLE_LOWER_USING_SWITCH_MERGE = default_lowering

  @parameterized.parameters((-2,), (2,), (5,))
  def testCase_gradient_diffShapedIntermediates(self, bi):
    nbranches = 5
    inputs = [
        array_ops.constant(
            float(bi), shape=[bi + 1], name="br{}_in".format(bi))
        for bi in range(nbranches)
    ]

    def make_func(bi):

      def f():
        x = inputs[bi]**2 * inputs[bi][:bi + 1, None]
        return math_ops.reduce_sum(x)

      return f

    branches = {bi: make_func(bi) for bi in range(nbranches)}

    branch_index = array_ops.placeholder_with_default(bi, [])
    with backprop.GradientTape() as tape:
      for x in inputs:
        tape.watch(x)
      case_out = control_flow_ops.switch_case(
          branch_index, branches, name=self.make_name())
    out_grad = 3.
    actual_grads = tape.gradient(case_out, inputs, output_gradients=out_grad)
    used_bi = (nbranches - 1) if (bi < 0 or bi >= nbranches - 1) else bi
    expected_grads = []
    for input_idx in range(nbranches):
      if used_bi == input_idx:
        with backprop.GradientTape() as tape:
          tape.watch(inputs[used_bi])
          y = make_func(used_bi)()
        expected_grads.append(
            self.evaluate(
                tape.gradient(y, inputs[used_bi], output_gradients=out_grad)))
      else:
        expected_grads.append(None if context.executing_eagerly() else [0.] *
                              (input_idx + 1))

    self.assertEqual(len(expected_grads), len(actual_grads))
    for expected, actual in zip(expected_grads, actual_grads):
      if expected is None:
        self.assertIsNone(actual)
      else:
        self.assertAllEqual(expected, self.evaluate(actual))

  @test_util.run_gpu_only
  @test_util.disable_xla("Wants RunMetadata")
  def testParallelExecution(self):
    """Verify disjoint branches across while iterations are run in parallel."""
    self.skipTest("b/210666081: Flaky")
    if control_flow_v2_toggles.control_flow_v2_enabled():
      self.skipTest("b/138870290")

    with ops.Graph().as_default() as g:
      nbranches = 7
      matrices = array_ops_stack.unstack(  # Ensure all are ready before while.
          array_ops.matrix_diag(
              random_ops.random_uniform([nbranches, 8, 512]) + 1e-3))

      def make_branch(i, mat, name):

        def branch_fn():
          next_i = i + 1
          with ops.device("gpu:0"):
            return next_i, math_ops.reduce_sum(
                linalg_ops.cholesky(mat, name=name + "_Cholesky"))

        return branch_fn

      def make_branches(i):
        return [
            make_branch(i, matrices[bi], "br{}".format(bi))
            for bi in range(nbranches)
        ]

      def cond(i, _):
        return i < nbranches

      def body(i, result):
        with ops.device("cpu:0"):
          next_i, branch_out = control_flow_ops.switch_case(i, make_branches(i))
        return next_i, result + branch_out

      _, result = while_loop.while_loop(cond, body, [0, 0.])

      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      config = config_pb2.ConfigProto(
          allow_soft_placement=False, log_device_placement=True)

      with session.Session(config=config, graph=g) as sess:
        _ = sess.run(result, options=run_options, run_metadata=run_metadata)
    chol_node_stats = []
    for dev_stats in run_metadata.step_stats.dev_stats:
      for node_stats in dev_stats.node_stats:
        if (node_stats.node_name.endswith("Cholesky") and
            node_stats.all_start_nanos > 0):
          chol_node_stats.append(node_stats)

    self.assertLen(chol_node_stats, nbranches)

    chol_node_stats = sorted(chol_node_stats, key=lambda stats: stats.node_name)
    op_start_nanos = [stats.all_start_nanos for stats in chol_node_stats]
    op_end_nanos = [
        stats.all_start_nanos + stats.op_end_rel_nanos
        for stats in chol_node_stats
    ]

    def overlap(range1, range2):
      s1, e1 = range1
      s2, e2 = range2
      if s1 < s2:
        return 0 if s2 > e1 else e1 - s2
      return 0 if s1 > e2 else e2 - s1

    timespans = list(zip(op_start_nanos, op_end_nanos))
    overlaps_chol0 = [overlap(timespans[0], r2) for r2 in timespans[1:]]
    # There are nbranches-1 overlaps, sometimes all nonzero, but we
    # conservatively check for at least one here, to avoid test flakiness.
    self.assertGreater(np.count_nonzero(overlaps_chol0), 0)

  def testCase_validateIndicesContiguous(self):

    def make_func(bi):
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = {i: make_func(i) for i in range(0, 6, 2)}
    with self.assertRaisesRegex(ValueError, "must form contiguous"):
      control_flow_ops.switch_case(array_ops.constant(0), branches)

  def testCase_validateIndicesDup(self):

    def make_func(bi):
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = [(i, make_func(i)) for i in range(0, 6, 2)]
    branches.append((0, make_func(7)))
    with self.assertRaisesRegex(ValueError, "must form contiguous"):
      control_flow_ops.switch_case(array_ops.constant(0), branches)

  def testCase_validateBranchIndex(self):

    def make_func(bi):
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = {i: make_func(i) for i in range(5)}
    with self.assertRaisesRegex(TypeError, "branch_index.*Tensor"):
      control_flow_ops.switch_case(1, branches)

  def testCase_validateNonIntKeys(self):

    def make_func(bi):
      return lambda: array_ops.constant(bi * 10., name="br{}_out".format(bi))

    branches = [(array_ops.constant(i), make_func(i)) for i in range(5)]
    with self.assertRaisesRegex(TypeError, "must be a Python `int`"):
      control_flow_ops.switch_case(array_ops.constant(1), branches)


class ExecuteFnForDeviceTest(test_util.TensorFlowTestCase):

  # The same test can run with and without XLA compilation.
  # In non-XLA gpu case, it exercises gpu branch.
  # In XLA gpu cases, it exercises the default case.
  # This test is to test the non-XLA case so that we disable XLA.
  @test_util.disable_xla("xla has different execution branch")
  def testCommonCases(self):

    def cpu_fn(x):
      return x + x

    def gpu_fn(x):
      return x * x

    def flexible_fn(a):
      branches = {"CPU": lambda: cpu_fn(a), "GPU": lambda: gpu_fn(a)}
      return control_flow_ops.execute_fn_for_device(branches, lambda: cpu_fn(a))

    @def_function.function
    def flexible_defun(a):
      return flexible_fn(a)

    def run_defun_and_tape(a):
      with backprop.GradientTape() as tape:
        tape.watch(a)
        result = flexible_defun(a)
      grad = tape.gradient(result, a)
      r = flexible_fn(a)
      return r, result, grad

    a = array_ops.constant(3.)
    with ops.device("cpu:0"):
      r, result, grad = run_defun_and_tape(a)
      self.assertEqual(6., self.evaluate(r))
      self.assertEqual(6., self.evaluate(result))
      self.assertEqual([2.], self.evaluate(grad))

    if test_util.is_gpu_available():
      with ops.device("gpu:0"):
        r, result, grad = run_defun_and_tape(a)
        self.assertEqual(9., self.evaluate(r))
        self.assertEqual(9., self.evaluate(result))
        self.assertEqual([6.], self.evaluate(grad))

    # no device annotation
    r, result, grad = run_defun_and_tape(a)
    if test_util.is_gpu_available():
      self.assertEqual(9., self.evaluate(r))
      self.assertEqual(9., self.evaluate(result))
      self.assertEqual([6.], self.evaluate(grad))
    else:
      self.assertEqual(6., self.evaluate(r))
      self.assertEqual(6., self.evaluate(result))
      self.assertEqual([2.], self.evaluate(grad))

  def testCompile(self):
    if not test_util.is_gpu_available():
      return

    def cpu_fn(x):
      return x + x

    def gpu_fn(x):
      return x * x

    @def_function.function(jit_compile=True)
    def flexible_defun(a):
      branches = {"CPU": lambda: cpu_fn(a), "GPU": lambda: gpu_fn(a)}
      return control_flow_ops.execute_fn_for_device(branches, lambda: cpu_fn(a))

    # Always execute the default branch in xla compilation case.
    a = array_ops.constant(3.)
    r = flexible_defun(a)
    self.assertEqual(6., self.evaluate(r))

  def testFallBack(self):

    def default_fn(x):
      return x

    def tpu_fn(x):
      return x * x * x

    def flexible_fn(a):
      branches = {"TPU": lambda: tpu_fn(a)}
      return control_flow_ops.execute_fn_for_device(
          branches, default_fn=lambda: default_fn(a))

    @def_function.function
    def flexible_defun(a):
      return flexible_fn(a)

    a = array_ops.constant(3.)
    with ops.device("cpu:0"):
      result_defun = flexible_defun(a)
      result_defun = flexible_fn(a)
      self.assertEqual(3., self.evaluate(result_defun))
      # execute_fn_for_device is not inside defun_function.
      result = flexible_fn(a)
      self.assertEqual(3., self.evaluate(result))

    if test_util.is_gpu_available():
      with ops.device("gpu:0"):
        result_defun = flexible_defun(a)
        self.assertEqual(3., self.evaluate(result_defun))
        # execute_fn_for_device is not inside defun_function.
        result = flexible_fn(a)
        self.assertEqual(3., self.evaluate(result))


class CaseTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testCase_withDefault(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4))]
    default = lambda: constant_op.constant(6)
    output = control_flow_case.case(conditions, default, exclusive=True)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 2}), 4)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 6)

  @test_util.run_deprecated_v1
  def testCase_multiple_matches_exclusive(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(6))]
    default = lambda: constant_op.constant(8)
    output = control_flow_case.case(conditions, default, exclusive=True)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 8)
      with self.assertRaisesRegex(errors.InvalidArgumentError, "Input error:"):
        sess.run(output, feed_dict={x: 2})

  @test_util.run_deprecated_v1
  def testCase_multiple_matches_non_exclusive(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(6))]
    default = lambda: constant_op.constant(8)
    output = control_flow_case.case(conditions, default, exclusive=False)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 2}), 4)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 8)

  @test_util.run_deprecated_v1
  def testCase_withoutDefault(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4)),
                  (math_ops.equal(x, 3), lambda: constant_op.constant(6))]
    output = control_flow_case.case(conditions, exclusive=True)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 2}), 4)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 6)
      with self.assertRaisesRegex(errors.InvalidArgumentError, "Input error:"):
        sess.run(output, feed_dict={x: 4})

  @test_util.run_deprecated_v1
  def testCase_withoutDefault_oneCondition(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2))]
    output = control_flow_case.case(conditions, exclusive=True)
    with self.cached_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      with self.assertRaisesRegex(errors.InvalidArgumentError, "Input error:"):
        sess.run(output, feed_dict={x: 4})

  @test_util.run_in_graph_and_eager_modes
  def testCase_dict(self):
    x = constant_op.constant(2)
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4))]
    output = control_flow_case.case(conditions, exclusive=True)
    self.assertEqual(4, self.evaluate(output))


class WhileLoopTestCase(test_util.TensorFlowTestCase):

  @test_util.run_in_graph_and_eager_modes
  def testWhileLoopWithSingleVariable(self):
    i = constant_op.constant(0)
    c = lambda i: math_ops.less(i, 10)
    b = lambda i: math_ops.add(i, 1)
    r = while_loop.while_loop(c, b, [i])

    self.assertEqual(self.evaluate(r), 10)

  @test_util.run_in_graph_and_eager_modes
  def testEagerWhileLoopWithSingleVariable_bodyReturnsTuple(self):
    i = constant_op.constant(0)
    c = lambda i: math_ops.less(i, 10)
    b = lambda i: (math_ops.add(i, 1),)
    r = while_loop.while_loop(c, b, [i])

    # Expect a tuple since that is what the body returns.
    self.assertEqual(self.evaluate(r), (10,))

  @test_util.run_v1_only("Unsupported in cfv2")
  def testWhileLoopSameReturnShape_False(self):
    i = constant_op.constant(0)
    c = lambda i, _: math_ops.less(i, 10)

    # Body returns a [tensor, []]
    b = lambda i, _: [math_ops.add(i, 1), []]

    # Should only return the tensor.
    r = while_loop.while_loop(c, b, [i, []])
    self.assertEqual(self.evaluate(r), 10)

    # Adding maximum_iterations should yield the same result.
    r = while_loop.while_loop(c, b, [i, []], maximum_iterations=50)
    # Note: this result is still incorrect - it should be just 10.
    self.assertEqual(self.evaluate(r), [10, []])

  def testWhileLoopSameReturnShape_FalseSingleLoopVar(self):
    i = constant_op.constant(0)
    c = lambda i: math_ops.less(i, 10)

    # Body return must be unpacked in this case.
    b = lambda i: math_ops.add(i, 1)

    # Should only return the tensor.
    r = while_loop.while_loop(c, b, [i])
    self.assertEqual(self.evaluate(r), 10)

    # Adding maximum_iterations should yield the same result.
    r = while_loop.while_loop(c, b, [i], maximum_iterations=50)
    self.assertEqual(self.evaluate(r), 10)

  def testWhileLoopSameReturnShape_True(self):
    i = constant_op.constant(0)
    c = lambda i, _: math_ops.less(i, 10)

    # Body returns a [tensor, []]
    b = lambda i, _: [math_ops.add(i, 1), []]

    # Should only return the original structure.
    r = while_loop.while_loop(c, b, [i, []], return_same_structure=True)
    self.assertEqual(self.evaluate(r), [10, []])

    # Adding maximum_iterations should yield the same result.
    r = while_loop.while_loop(
        c, b, [i, []], return_same_structure=True, maximum_iterations=50)
    self.assertEqual(self.evaluate(r), [10, []])

  def testWhileLoopSameReturnShape_TrueSingleLoopVar(self):
    i = constant_op.constant(0)
    c = lambda i: math_ops.less(i, 10)

    b = lambda i: [math_ops.add(i, 1)]

    # Should not unpack the single variable
    r = while_loop.while_loop(c, b, [i], return_same_structure=True)
    self.assertEqual(self.evaluate(r), [10])

    # Adding maximum_iterations should yield the same result.
    r = while_loop.while_loop(
        c, b, [i], return_same_structure=True, maximum_iterations=50)
    self.assertEqual(self.evaluate(r), [10])

  @test_util.enable_control_flow_v2
  @test_util.run_in_graph_and_eager_modes
  def testSkipsUnnecessaryCaptureGradients(self):

    @custom_gradient.custom_gradient
    def gradient_trap(t):

      def grad(w):
        # Computing this gradient should fail the test
        check_ops.assert_equal(0, 1)
        return w

      return t, grad

    x = array_ops.constant(0.0, name="x")
    y = array_ops.constant(1.0, name="y")

    def cond(s):
      return s < 10.0

    def body(s):
      return s + 2 * x + gradient_trap(y)

    with backprop.GradientTape() as tape:
      tape.watch(x)
      out = while_loop.while_loop(cond, body, (array_ops.constant(0.0),))

    grad = tape.gradient(out, x)
    self.assertAllEqual(grad, 20.0)


class WhileLoopParallelismTest(test_util.TensorFlowTestCase,
                               parameterized.TestCase):

  @parameterized.parameters(*itertools.product(
      (False, True),
      (False, True),
      (False, True),
      (False, True),
      (False, True),
  ))
  def testResourceHandlingInLoop(self, read_before, read_after, modify_in_loop,
                                 modify_before, modify_after):

    if not tf2.enabled():
      self.skipTest("V2-only test.")

    ticker = variables.Variable(0)

    @def_function.function
    def run_loop(n):
      ticker.assign(0)
      i = constant_op.constant(0)
      t_acc = tensor_array_ops.TensorArray(
          dtypes.int32, size=0, dynamic_size=True)

      if read_before:
        rb = ticker.read_value()
      else:
        rb = constant_op.constant(0)
      if modify_before:
        ticker.assign_add(1)

      while i < n:
        directives.set_loop_options(parallel_iterations=10)
        if modify_in_loop:
          ticker.assign_add(1)
        t_acc = t_acc.write(i, ticker.read_value())
        i += 1

      if read_after:
        ra = ticker.read_value()
      else:
        ra = constant_op.constant(0)
      if modify_after:
        ticker.assign_add(1)

      return t_acc.stack(), rb, ra

    # Warm-up.
    self.evaluate(run_loop(1))

    self.evaluate(ticker.assign(123))
    acc, rb, ra = run_loop(3)
    self.assertEqual(
        self.evaluate(math_ops.reduce_max(acc)),
        int(modify_before) + 3 * int(modify_in_loop))

    # Double check variable reads are still sequenced.
    self.assertEqual(self.evaluate(rb), 0)

    if read_after:
      expected_ra = int(modify_before) + 3 * int(modify_in_loop)
    else:
      expected_ra = 0
    self.assertEqual(self.evaluate(ra), expected_ra)

    # Double-check that the loop ran completely.
    self.assertEqual(
        self.evaluate(ticker.read_value()),
        int(modify_before) + 3 * int(modify_in_loop) + int(modify_after))

  def testStatefulParallelism(self):

    if not tf2.enabled():
      self.skipTest("V2-only test.")

    ticker = variables.Variable(0)
    # Secondary state for the pyfunc that lets us verify that things ran in
    # the correct relative order.
    ticker_state = []

    def wait_then_tick(i):
      # The contents of py_funcs is opaque, so TF doesn't see this variable
      # assignment. In turn, this allows us to run it in parallel with
      # the variable read.
      def wait_then_tick_py_fn(i):
        time.sleep(1)
        ticker.assign_add(1)
        ticker_state.append(i.numpy().item())
        return 1

      return script_ops.eager_py_func(wait_then_tick_py_fn, [i],
                                      [dtypes.int32])[0]

    @def_function.function
    def run_loop(n):
      ticker.assign(0)
      i = constant_op.constant(0)
      t_acc = tensor_array_ops.TensorArray(
          dtypes.int32, size=0, dynamic_size=True)

      while i < n:
        directives.set_loop_options(parallel_iterations=10)
        wait_then_tick(i + 1)
        # The read is expected to run in much less than `wait_then_tick`,
        # which sleeps for 1s. Hence all reads should complete before the first
        # `wait_then_tick` increments the `ticker` variable.
        t_acc = t_acc.write(i, ticker.read_value())
        i += 1

      return t_acc.stack()

    # Warm-up.
    self.evaluate(run_loop(1))

    # This test is deterministic so long as the runtime is fast enough to
    # execute `t_acc = t_acc.write(i, ticker.read_value())` in much less than
    # one second.
    self.evaluate(ticker.assign(123))
    ticker_state.clear()
    acc = run_loop(3)
    # Because the loop runs entirely sequentially, the reads in each iteration
    # see the effects of the pyfunc from the previous iteration.
    self.assertEqual(self.evaluate(math_ops.reduce_max(acc)), 2)

    # Double-check that the loop ran completely.
    self.assertEqual(self.evaluate(ticker.read_value()), 3)
    # Double check that the pyfuncs ran in order.
    self.assertListEqual(ticker_state, [1, 2, 3])


class AssertTest(test_util.TensorFlowTestCase):

  @test_util.run_deprecated_v1
  def testAssert(self):
    i = constant_op.constant(0)
    c = control_flow_assert.Assert(i < 10, [i, [10], [i + 1]])
    self.evaluate(c)

    i = constant_op.constant(10)
    c = control_flow_assert.Assert(i < 10, [i, [10], [i + 1]])
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(c)

  @test_util.run_in_graph_and_eager_modes
  def testAssertInFunction(self):
    # TODO(fishx): Re-enable this test for GPU.
    # NOTE(fishx): Disable this test for now because, in GPU, multiple errors
    # will be thrown. But since the root cause error is marked as "derived"
    # error. So it might be ignored.
    if test_util.is_gpu_available():
      self.skipTest("Skip GPU Test")

    @def_function.function
    def whiny(value):
      control_flow_assert.Assert(value, ["Raised false"])
      return constant_op.constant(5)

    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(whiny(False))

    self.assertAllEqual(whiny(True), 5)


if __name__ == "__main__":
  googletest.main()
