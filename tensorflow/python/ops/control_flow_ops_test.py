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

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
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
from tensorflow.python.util.protobuf import compare


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

  def testIndexedSlicesGradientInCondInWhileLoop(self):
    with ops.Graph().as_default():
      embedding_matrix = variable_scope.get_variable(
          "embedding_matrix", [5, 5],
          initializer=init_ops.random_normal_initializer())

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
          compare.ProtoEq(
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
          compare.ProtoEq(
              c.to_proto(),
              control_flow_ops.WhileContext.from_proto(c.to_proto()).to_proto())


if __name__ == "__main__":
  googletest.main()
