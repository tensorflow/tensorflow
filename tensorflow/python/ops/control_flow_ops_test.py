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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import googletest
from tensorflow.python.training import momentum
from tensorflow.python.util import nest


TestTuple = collections.namedtuple("TestTuple", "a b")
SingletonTestTuple = collections.namedtuple("SingletonTestTuple", "a")


class GroupTestCase(TensorFlowTestCase):

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
    self.assertProtoEquals("""
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
    self.assertProtoEquals("""
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
    self.assertProtoEquals("""
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


class ShapeTestCase(TensorFlowTestCase):

  def testShape(self):
    with ops.Graph().as_default():
      tensor = constant_op.constant([1.0, 2.0])
      self.assertEquals([2], tensor.get_shape())
      self.assertEquals([2],
                        control_flow_ops.with_dependencies(
                            [constant_op.constant(1.0)], tensor).get_shape())


class WithDependenciesTestCase(TensorFlowTestCase):

  def testTupleDependencies(self):
    with ops.Graph().as_default():
      counter = variable_scope.get_variable(
          "my_counter", shape=[], initializer=init_ops.zeros_initializer())
      increment_counter = state_ops.assign_add(counter, 1)
      const_with_dep = control_flow_ops.with_dependencies(
          (increment_counter, constant_op.constant(42)),
          constant_op.constant(7))
      with self.test_session():
        variables.global_variables_initializer().run()
        self.assertEquals(0, counter.eval())
        self.assertEquals(7, const_with_dep.eval())
        self.assertEquals(1, counter.eval())

  def testListDependencies(self):
    with ops.Graph().as_default():
      counter = variable_scope.get_variable(
          "my_counter", shape=[], initializer=init_ops.zeros_initializer())
      increment_counter = state_ops.assign_add(counter, 1)
      const_with_dep = control_flow_ops.with_dependencies(
          [increment_counter, constant_op.constant(42)],
          constant_op.constant(7))
      with self.test_session():
        variables.global_variables_initializer().run()
        self.assertEquals(0, counter.eval())
        self.assertEquals(7, const_with_dep.eval())
        self.assertEquals(1, counter.eval())


class SwitchTestCase(TensorFlowTestCase):

  def testIndexedSlicesWithDenseShape(self):
    with self.test_session():
      data = ops.IndexedSlices(
          constant_op.constant([1, 2, 3]),
          constant_op.constant([0, 1]),
          dense_shape=constant_op.constant([3]))
      zero = constant_op.constant(0)
      one = constant_op.constant(1)
      less_op = math_ops.less(zero, one)
      switch_false, switch_true = control_flow_ops.switch(data, less_op)
      self.assertAllEqual([1, 2, 3], switch_true.values.eval())
      self.assertAllEqual([0, 1], switch_true.indices.eval())

  def testIndexedSlicesGradient(self):
    with ops.Graph().as_default():
      embedding_matrix = variable_scope.get_variable(
          "embedding_matrix", [5, 5],
          initializer=init_ops.random_normal_initializer())

      def Cond(it, _):
        return it < 5

      def Body(it, cost):
        embedding = embedding_ops.embedding_lookup(embedding_matrix + 0.0, [0])
        cost += math_ops.reduce_sum(embedding)
        return it + 1, cost

      _, cost = control_flow_ops.while_loop(
          Cond, Body, [constant_op.constant(0), constant_op.constant(0.0)])
      optimizer = momentum.MomentumOptimizer(0.1, 0.9)
      train_op = optimizer.minimize(cost)
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        for _ in range(10):
          sess.run([train_op])

  def testResourceReadInLoop(self):
    with ops.Graph().as_default():
      embedding_matrix = variable_scope.get_variable(
          "embedding_matrix",
          initializer=[[2.0], [3.0]],
          use_resource=True)

      def Cond(it, _):
        return it < 5

      def Body(it, cost):
        embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
        cost += math_ops.reduce_sum(embedding)
        return it + 1, cost

      _, cost = control_flow_ops.while_loop(
          Cond, Body, [constant_op.constant(0), constant_op.constant(0.0)])
      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self.assertAllEqual(10.0, cost.eval())

  def doTestIndexedSlicesGradientInCondInWhileLoop(self, use_resource=False):
    with ops.Graph().as_default():
      embedding_matrix = variable_scope.get_variable(
          "embedding_matrix", [5, 5],
          initializer=init_ops.random_normal_initializer(),
          use_resource=use_resource)

      def Cond(it, _):
        return it < 5

      def Body(it, cost):
        embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
        cost = control_flow_ops.cond(
            math_ops.equal(it, 3), lambda: math_ops.square(cost),
            lambda: cost + math_ops.reduce_sum(embedding))
        return it + 1, cost

      _, cost = control_flow_ops.while_loop(
          Cond, Body, [constant_op.constant(0), constant_op.constant(0.0)])

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

      with self.test_session() as sess:
        sess.run(variables.global_variables_initializer())
        self.assertAllEqual(*sess.run([static_grads, dynamic_grads]))

  def testIndexedSlicesGradientInCondInWhileLoop(self):
    self.doTestIndexedSlicesGradientInCondInWhileLoop(use_resource=False)

  def testIndexedSlicesGradientInCondInWhileLoopResource(self):
    self.doTestIndexedSlicesGradientInCondInWhileLoop(use_resource=True)

  def testIndexedSlicesWithShapeGradientInWhileLoop(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.test_session() as sess:
        num_steps = 9

        inputs = array_ops.placeholder(dtype=dtype, shape=[num_steps])
        initial_outputs = tensor_array_ops.TensorArray(
            dtype=dtype, size=num_steps)
        initial_i = constant_op.constant(0, dtype=dtypes.int32)

        def Cond(i, _):
          return i < num_steps  # pylint: disable=cell-var-from-loop

        def Body(i, outputs):
          x = array_ops.gather(inputs, i)  # pylint: disable=cell-var-from-loop
          outputs = outputs.write(i, x)
          return i + 1, outputs

        _, outputs = control_flow_ops.while_loop(Cond, Body,
                                                 [initial_i, initial_outputs])

        outputs = math_ops.reduce_sum(outputs.stack())
        r = gradients_impl.gradients([outputs], [inputs])[0]
        grad_wr_inputs = ops.convert_to_tensor(r)
        o, grad = sess.run([outputs, grad_wr_inputs],
                           feed_dict={inputs: [4, 6, 0, 7, 0, 0, 1, 2, 0]})
        self.assertEquals(o, 20)
        self.assertAllEqual(grad, [1] * num_steps)

  def testIndexedSlicesWithDynamicShapeGradientInWhileLoop(self):
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.test_session() as sess:
        inputs = array_ops.placeholder(dtype=dtype)
        initial_outputs = tensor_array_ops.TensorArray(
            dtype=dtype, dynamic_size=True, size=1)
        initial_i = constant_op.constant(0, dtype=dtypes.int32)

        def Cond(i, _):
          return i < array_ops.size(inputs)  # pylint: disable=cell-var-from-loop

        def Body(i, outputs):
          x = array_ops.gather(inputs, i)  # pylint: disable=cell-var-from-loop
          outputs = outputs.write(i, x)
          return i + 1, outputs

        _, outputs = control_flow_ops.while_loop(Cond, Body,
                                                 [initial_i, initial_outputs])

        outputs = math_ops.reduce_sum(outputs.stack())
        r = gradients_impl.gradients([outputs], [inputs])[0]
        grad_wr_inputs = ops.convert_to_tensor(r)
        o, grad = sess.run([outputs, grad_wr_inputs],
                           feed_dict={inputs: [1, 3, 2]})
        self.assertEquals(o, 6)
        self.assertAllEqual(grad, [1] * 3)

  def testGradientThroughSingleBranchOutsideOfContext(self):
    with self.test_session():
      x = constant_op.constant(2.)
      s = constant_op.constant(True)
      x_false, x_true = control_flow_ops.switch(x, s)
      grad_x_true = gradients_impl.gradients(x_true, x)[0]
      grad_x_false = gradients_impl.gradients(x_false, x)[0]
      self.assertEquals(grad_x_true.eval(), 1.)
      self.assertEquals(grad_x_false.eval(), 0.)


class CondTest(TensorFlowTestCase):

  def testCondTrue(self):
    with self.test_session():
      x = constant_op.constant(2)
      y = constant_op.constant(5)
      z = control_flow_ops.cond(
          math_ops.less(x, y), lambda: math_ops.multiply(x, 17),
          lambda: math_ops.add(y, 23))
      self.assertEquals(z.eval(), 34)

  def testCondFalse(self):
    with self.test_session():
      x = constant_op.constant(2)
      y = constant_op.constant(1)
      z = control_flow_ops.cond(
          math_ops.less(x, y), lambda: math_ops.multiply(x, 17),
          lambda: math_ops.add(y, 23))
      self.assertEquals(z.eval(), 24)

  def testCondTrueLegacy(self):
    with self.test_session():
      x = constant_op.constant(2)
      y = constant_op.constant(5)
      z = control_flow_ops.cond(
          math_ops.less(x, y), fn1=lambda: math_ops.multiply(x, 17),
          fn2=lambda: math_ops.add(y, 23))
      self.assertEquals(z.eval(), 34)

  def testCondFalseLegacy(self):
    with self.test_session():
      x = constant_op.constant(2)
      y = constant_op.constant(1)
      z = control_flow_ops.cond(
          math_ops.less(x, y), fn1=lambda: math_ops.multiply(x, 17),
          fn2=lambda: math_ops.add(y, 23))
      self.assertEquals(z.eval(), 24)

  def testCondMissingArg1(self):
    with self.test_session():
      x = constant_op.constant(1)
      with self.assertRaises(TypeError):
        control_flow_ops.cond(True, false_fn=lambda: x)

  def testCondMissingArg2(self):
    with self.test_session():
      x = constant_op.constant(1)
      with self.assertRaises(TypeError):
        control_flow_ops.cond(True, lambda: x)

  def testCondDuplicateArg1(self):
    with self.test_session():
      x = constant_op.constant(1)
      with self.assertRaises(TypeError):
        control_flow_ops.cond(True, lambda: x, lambda: x, fn1=lambda: x)

  def testCondDuplicateArg2(self):
    with self.test_session():
      x = constant_op.constant(1)
      with self.assertRaises(TypeError):
        control_flow_ops.cond(True, lambda: x, lambda: x, fn2=lambda: x)


class ContextTest(TensorFlowTestCase):

  def testCondContext(self):
    with self.test_session() as sess:
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

  def testWhileContext(self):
    with self.test_session() as sess:
      i = constant_op.constant(0)
      c = lambda i: math_ops.less(i, 10)
      b = lambda i: math_ops.add(i, 1)
      control_flow_ops.while_loop(c, b, [i])
      for op in sess.graph.get_operations():
        c = op._get_control_flow_context()
        if c:
          self.assertProtoEquals(
              c.to_proto(),
              control_flow_ops.WhileContext.from_proto(c.to_proto()).to_proto())

  def testControlContextImportScope(self):
    with self.test_session():
      constant_op.constant(0, name="a")
      constant_op.constant(2, name="test_scope/a")
      b1 = constant_op.constant(1, name="b")
      b2 = constant_op.constant(3, name="test_scope/b")

      c = control_flow_ops.ControlFlowContext()
      c._values = ["a", "b"]
      c._external_values = {"a": b1}

      c_with_scope = control_flow_ops.ControlFlowContext._from_proto(
          c._to_proto(), import_scope="test_scope")

      # _values and _external_values should be have scope prepended.
      self.assertEquals(
          c_with_scope._values, set(["test_scope/a", "test_scope/b"]))
      self.assertEquals(
          c_with_scope._external_values, {"test_scope/a": b2})

      # Calling _to_proto() with export_scope should remove "test_scope".
      self.assertProtoEquals(
          c._to_proto(),
          c_with_scope._to_proto(export_scope="test_scope"))


def _GetNestedShape(nested):
  def _GetShape(tensor):
    if isinstance(tensor, tensor_array_ops.TensorArray):
      return tensor_array_ops.TensorArray
    elif isinstance(tensor, ops.IndexedSlices):
      return tensor.dense_shape
    else:
      return tensor.get_shape()

  return nest.map_structure(_GetShape, nested)


def _CreateTensorArray(size, shape):
  ta = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=size,
                                    clear_after_read=False)
  for i in range(size):
    ta = ta.write(i, array_ops.zeros(shape))
  return ta


def _RawNestedShape(nested_shape):
  def _RawShape(shape):
    if isinstance(shape, tensor_shape.TensorShape) and shape.ndims is not None:
      return [x.value for x in shape]
    else:
      return None
  return nest.map_structure(_RawShape, nested_shape)


# TODO(yori): Add tests for indexed slices.
class DataTypesTest(TensorFlowTestCase):

  def assertAllEqualNested(self, a, b):
    if isinstance(a, (list, tuple)):
      for entry_a, entry_b in zip(a, b):
        self.assertAllEqualNested(entry_a, entry_b)
    else:
      self.assertAllEqual(a, b)

  def _testShape(self, fn_true, fn_false, expected_shape,
                 strict=False):
    condition = array_ops.placeholder(dtypes.bool)
    output_cond = control_flow_ops.cond(condition, fn_true, fn_false,
                                        strict=strict)
    self.assertEqual(_RawNestedShape(_GetNestedShape(output_cond)),
                     _RawNestedShape(expected_shape))

    output_case = control_flow_ops.case([(condition, fn_true)], fn_false,
                                        strict=strict)
    self.assertEqual(_RawNestedShape(_GetNestedShape(output_case)),
                     _RawNestedShape(expected_shape))

  def _testReturnValues(self, fn_true, fn_false, expected_value_true,
                        expected_value_false, strict=False,
                        check_cond=True):
    condition = array_ops.placeholder(dtypes.bool)
    output_cond = control_flow_ops.cond(condition, fn_true, fn_false,
                                        strict=strict)
    output_case = control_flow_ops.case([(condition, fn_true)], fn_false,
                                        strict=strict)

    with self.test_session() as sess:
      variables.global_variables_initializer().run()
      result_cond, result_case = sess.run([output_cond, output_case],
                                          feed_dict={condition: True})
      self.assertAllEqualNested(result_cond, expected_value_true)
      if check_cond:
        self.assertAllEqualNested(result_case, expected_value_true)
      result_cond, result_case = sess.run([output_cond, output_case],
                                          feed_dict={condition: False})
      self.assertAllEqualNested(result_cond, expected_value_false)
      if check_cond:
        self.assertAllEqualNested(result_case, expected_value_false)

  def test_int(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: 1
    fn_false = lambda: 2
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 1, 2)
    self._testShape(fn_true, fn_false, shape, strict=True)
    self._testReturnValues(fn_true, fn_false, 1, 2, strict=True)

  def test_float(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: 1.0
    fn_false = lambda: 2.0
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 1.0, 2.0)

  def test_noop(self):
    shape = tensor_shape.TensorShape(None)
    self._testShape(control_flow_ops.no_op, control_flow_ops.no_op, shape)
    self._testReturnValues(control_flow_ops.no_op, control_flow_ops.no_op,
                           True, False, check_cond=False)

  def test_string(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: "abc"
    fn_false = lambda: "xyz"
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, b"abc", b"xyz")

  def test_variable(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: variables.Variable(3.0)
    fn_false = lambda: variables.Variable(4.0)
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 3.0, 4.0)

  def test_none(self):
    fn_none = lambda: None
    fn_tensor = lambda: constant_op.constant(1)

    with self.assertRaises(ValueError):
      control_flow_ops.cond(constant_op.constant(True), fn_none, fn_tensor)

    with self.assertRaises(ValueError):
      control_flow_ops.cond(constant_op.constant(True), fn_tensor, fn_none)

  def test_tensors(self):
    def _BuildTrueBranch(dtype):
      def _Build():
        return (array_ops.zeros([2, 2], dtype=dtype),
                array_ops.ones([3, 3], dtype=dtype))
      return _Build

    def _BuildFalseBranch(dtype):
      def _Build():
        return (array_ops.ones([2, 2], dtype=dtype),
                array_ops.zeros([3, 3], dtype=dtype))
      return _Build

    for dtype in (dtypes.float16, dtypes.int8, dtypes.int32, dtypes.uint8):
      shape = (tensor_shape.TensorShape([2, 2]),
               tensor_shape.TensorShape([3, 3]))
      fn_true = _BuildTrueBranch(dtype)
      fn_false = _BuildFalseBranch(dtype)
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false,
                             (np.zeros([2, 2]), np.ones([3, 3])),
                             (np.ones([2, 2]), np.zeros([3, 3])))

  def test_tensors_unknown_shape(self):
    def _BuildTrueBranch(dtype):
      def _Build():
        tensor = array_ops.zeros([2, 2], dtype=dtype)
        tensor._shape = tensor_shape.TensorShape(None)
        return tensor
      return _Build

    def _BuildFalseBranch(dtype):
      def _Build():
        tensor = array_ops.ones([2, 2], dtype=dtype)
        tensor._shape = tensor_shape.TensorShape(None)
        return tensor
      return _Build

    for dtype in (dtypes.float16, dtypes.int8, dtypes.int32, dtypes.uint8):
      shape = tensor_shape.TensorShape(None)
      fn_true = _BuildTrueBranch(dtype)
      fn_false = _BuildFalseBranch(dtype)
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false,
                             np.zeros([2, 2]), np.ones([2, 2]))

  def test_sparse_tensors(self):
    shape = tensor_shape.TensorShape([None, None])

    def FnTrue():
      return [sparse_tensor.SparseTensor(indices=[[0, 0], [1, 2]],
                                         values=[1, 2], dense_shape=[3, 4])]

    def FnFalse():
      return [sparse_tensor.SparseTensor(indices=[[0, 0], [2, 1]],
                                         values=[3, 4], dense_shape=[3, 4])]

    value1 = sparse_tensor.SparseTensorValue(indices=[[0, 0], [1, 2]],
                                             values=[1, 2], dense_shape=[3, 4])
    value2 = sparse_tensor.SparseTensorValue(indices=[[0, 0], [2, 1]],
                                             values=[3, 4], dense_shape=[3, 4])
    self._testShape(FnTrue, FnFalse, shape)
    self._testReturnValues(FnTrue, FnFalse, value1, value2)
    self._testShape(FnTrue, FnFalse, [shape], strict=True)
    self._testReturnValues(FnTrue, FnFalse, [value1], [value2], strict=True)

  def test_tensors_with_partially_specified_shapes(self):
    def _BuildBranch(dtype, shape):
      def _Build():
        a = array_ops.zeros([2, 2], dtype=dtype)
        b = array_ops.zeros([5], dtype=dtype)
        c = array_ops.ones([3, 3], dtype=dtype)
        a._shape = tensor_shape.TensorShape(shape[0])
        b._shape = tensor_shape.TensorShape(shape[1])
        c._shape = tensor_shape.TensorShape(shape[2])
        return a, b, c
      return _Build

    for dtype in (dtypes.float16, dtypes.int8, dtypes.int32, dtypes.uint8):
      shape = (tensor_shape.TensorShape([None, 2]),
               tensor_shape.TensorShape([None]),
               tensor_shape.TensorShape([3, None]))
      fn_true = _BuildBranch(dtype, shape)
      fn_false = _BuildBranch(dtype, shape)
      self._testShape(fn_true, fn_false, shape)
      self._testReturnValues(fn_true, fn_false,
                             (np.zeros([2, 2]), np.zeros(5), np.ones([3, 3])),
                             (np.zeros([2, 2]), np.zeros(5), np.ones([3, 3])))

  def test_tensor_arrays(self):
    element_shape = tensor_shape.TensorShape([2])
    ta1 = _CreateTensorArray(4, element_shape)
    ta2 = _CreateTensorArray(4, element_shape)
    shape = tensor_array_ops.TensorArray
    fn_true = lambda: ta1
    fn_false = lambda: ta2
    self._testShape(fn_true, fn_false, shape)

  def test_tensor_array_reads(self):
    shape = tensor_shape.TensorShape([2])
    ta = _CreateTensorArray(4, shape)
    fn_true = lambda: ta.read(0)
    fn_false = lambda: ta.read(1)
    self._testShape(fn_true, fn_false, shape)

  def test_list(self):
    shape = [tensor_shape.TensorShape([]), tensor_shape.TensorShape([]),
             tensor_shape.TensorShape([])]
    fn_true = lambda: [constant_op.constant(1), 2, variables.Variable(3.0)]
    fn_false = lambda: [constant_op.constant(3), 4, variables.Variable(5.0)]
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, [1, 2, 3.0], [3, 4, 5.0])

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

  def test_singleton_strict(self):
    fn_tensor = lambda: constant_op.constant(1)
    fn_list = lambda: [constant_op.constant(2)]
    fn_tuple = lambda: (constant_op.constant(3),)

    with self.assertRaises(ValueError):
      control_flow_ops.cond(constant_op.constant(True), fn_tensor, fn_list,
                            strict=True)

    with self.assertRaises(TypeError):
      control_flow_ops.cond(constant_op.constant(True), fn_list, fn_tuple,
                            strict=True)

    with self.assertRaises(ValueError):
      control_flow_ops.case([(constant_op.constant(True), fn_tensor)], fn_list,
                            strict=True)

    with self.assertRaises(TypeError):
      control_flow_ops.case([(constant_op.constant(True), fn_list)], fn_tuple,
                            strict=True)

  def test_singleton_list(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: [constant_op.constant(1)]
    fn_false = lambda: [constant_op.constant(3)]
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 1, 3)
    self._testShape(fn_true, fn_false, [shape], strict=True)
    self._testReturnValues(fn_true, fn_false, [1], [3], strict=True)

  def test_singleton_tuple(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: (constant_op.constant(1),)
    fn_false = lambda: (constant_op.constant(3),)
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 1, 3)
    self._testShape(fn_true, fn_false, (shape,), strict=True)
    self._testReturnValues(fn_true, fn_false, (1,), (3,),
                           strict=True)

  def test_singleton_namedtuple(self):
    shape = tensor_shape.TensorShape([])
    fn_true = lambda: SingletonTestTuple(constant_op.constant(1))
    fn_false = lambda: SingletonTestTuple(constant_op.constant(3))
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, 1, 3)
    self._testShape(fn_true, fn_false, SingletonTestTuple(shape),
                    strict=True)
    self._testReturnValues(fn_true, fn_false, SingletonTestTuple(1),
                           SingletonTestTuple(3), strict=True)

  def test_tuple(self):
    shape = (tensor_shape.TensorShape([]), tensor_shape.TensorShape([]))
    fn_true = lambda: (constant_op.constant(1), 2)
    fn_false = lambda: (constant_op.constant(3), 4)
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, (1, 2), (3, 4))

  def test_namedtuple(self):
    shape = TestTuple(tensor_shape.TensorShape([]),
                      tensor_shape.TensorShape([]))
    fn_true = lambda: TestTuple(constant_op.constant(1), 2)
    fn_false = lambda: TestTuple(constant_op.constant(3), 4)
    self._testShape(fn_true, fn_false, shape)
    self._testReturnValues(fn_true, fn_false, TestTuple(1, 2), TestTuple(3, 4))

  def test_nested(self):
    shape = [tensor_shape.TensorShape([]),
             TestTuple(tensor_shape.TensorShape([]),
                       [tensor_shape.TensorShape([]),
                        tensor_shape.TensorShape([])]),
             tensor_shape.TensorShape([5, 5]),
             tensor_shape.TensorShape([])]

    def FnTrue():
      return [constant_op.constant(1),
              TestTuple(constant_op.constant(2), [3, 4]),
              array_ops.zeros([5, 5]), 6]

    def FnFalse():
      return [constant_op.constant(11),
              TestTuple(constant_op.constant(12), [13, 14]),
              array_ops.ones([5, 5]), 16]

    self._testShape(FnTrue, FnFalse, shape)
    self._testReturnValues(FnTrue, FnFalse,
                           [1, TestTuple(2, [3, 4]), np.zeros([5, 5]), 6],
                           [11, TestTuple(12, [13, 14]), np.ones([5, 5]), 16])

  def test_cond_inside_while_loop(self):
    def Body(i, matrix):
      result_tuple, unused_matrix = control_flow_ops.cond(
          constant_op.constant(True),
          lambda: (TestTuple(matrix * 2, matrix * 4), matrix),
          lambda: (TestTuple(matrix * 4, matrix * 2), matrix))
      return [i+1, result_tuple.a]

    iteration, matrix = control_flow_ops.while_loop(
        lambda i, matrix: i < 10,
        Body,
        loop_vars=[constant_op.constant(0), array_ops.ones([2, 2])])

    self.assertEqual(iteration.get_shape(), tensor_shape.TensorShape([]))
    self.assertEqual(matrix.get_shape(), tensor_shape.TensorShape([2, 2]))


class CaseTest(TensorFlowTestCase):

  def testCase_withDefault(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4))]
    default = lambda: constant_op.constant(6)
    output = control_flow_ops.case(conditions, default, exclusive=True)
    with self.test_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 2}), 4)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 6)

  def testCase_multiple_matches_exclusive(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(6))]
    default = lambda: constant_op.constant(8)
    output = control_flow_ops.case(conditions, default, exclusive=True)
    with self.test_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 8)
      with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                   "More than one condition evaluated as True"):
        sess.run(output, feed_dict={x: 2})

  def testCase_multiple_matches_non_exclusive(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(6))]
    default = lambda: constant_op.constant(8)
    output = control_flow_ops.case(conditions, default, exclusive=False)
    with self.test_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 2}), 4)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 8)

  def testCase_withoutDefault(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2)),
                  (math_ops.equal(x, 2), lambda: constant_op.constant(4)),
                  (math_ops.equal(x, 3), lambda: constant_op.constant(6))]
    output = control_flow_ops.case(conditions, exclusive=True)
    with self.test_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      self.assertEqual(sess.run(output, feed_dict={x: 2}), 4)
      self.assertEqual(sess.run(output, feed_dict={x: 3}), 6)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r"\[None of the conditions evaluated as True. "
          r"Conditions: \(Equal:0, Equal_1:0, Equal_2:0\), Values:\] "
          r"\[0 0 0\]"):
        sess.run(output, feed_dict={x: 4})

  def testCase_withoutDefault_oneCondition(self):
    x = array_ops.placeholder(dtype=dtypes.int32, shape=[])
    conditions = [(math_ops.equal(x, 1), lambda: constant_op.constant(2))]
    output = control_flow_ops.case(conditions, exclusive=True)
    with self.test_session() as sess:
      self.assertEqual(sess.run(output, feed_dict={x: 1}), 2)
      with self.assertRaisesRegexp(
          errors.InvalidArgumentError,
          r"\[None of the conditions evaluated as True. "
          r"Conditions: \(Equal:0\), Values:\] \[0\]"):
        sess.run(output, feed_dict={x: 4})


if __name__ == "__main__":
  googletest.main()
