# Copyright 2015 Google Inc. All Rights Reserved.
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
from tensorflow.python.framework import ops
from tensorflow.python.framework.test_util import TensorFlowTestCase
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import standard_ops as tf
from tensorflow.python.platform import googletest
from tensorflow.python.training import momentum


class GroupTestCase(TensorFlowTestCase):

  def _StripNode(self, nd):
    snode = graph_pb2.NodeDef(name=nd.name, op=nd.op, input=nd.input)
    if nd.device:
      snode.device = nd.device
    return snode

  def _StripGraph(self, gd):
    """Copy gd keeping only, node.name, node.op, node.input, and node.device."""
    return graph_pb2.GraphDef(node=[self._StripNode(nd) for nd in gd.node])

  def testGroup_NoDevices(self):
    with ops.Graph().as_default() as g:
      a = tf.constant(0, name="a")
      b = tf.constant(0, name="b")
      c = tf.constant(0, name="c")
      tf.group(a.op, b.op, c.op, name="root")
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
        a = tf.constant(0, name="a")
        b = tf.constant(0, name="b")
      tf.group(a.op, b.op, name="root")
    gd = g.as_graph_def()
    self.assertProtoEquals("""
      node { name: "a" op: "Const" device: "/task:0" }
      node { name: "b" op: "Const" device: "/task:0" }
      node { name: "root" op: "NoOp" input: "^a" input: "^b" device: "/task:0" }
    """, self._StripGraph(gd))

  def testGroup_MultiDevice(self):
    with ops.Graph().as_default() as g:
      with g.device("/task:0"):
        a = tf.constant(0, name="a")
        b = tf.constant(0, name="b")
      with g.device("/task:1"):
        c = tf.constant(0, name="c")
        d = tf.constant(0, name="d")
      with g.device("/task:2"):
        tf.group(a.op, b.op, c.op, d.op, name="root")
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
      tensor = tf.constant([1.0, 2.0])
      self.assertEquals([2], tensor.get_shape())
      self.assertEquals([2],
                        control_flow_ops.with_dependencies(
                            [tf.constant(1.0)], tensor).get_shape())


class SwitchTestCase(TensorFlowTestCase):

  def testIndexedSlicesWithDenseShape(self):
    with self.test_session():
      data = ops.IndexedSlices(tf.constant([1, 2, 3]),
                               tf.constant([0, 1]),
                               dense_shape=tf.constant([3]))
      zero = tf.constant(0)
      one = tf.constant(1)
      less_op = tf.less(zero, one)
      switch_false, switch_true = control_flow_ops.switch(data, less_op)
      self.assertAllEqual([1, 2, 3], switch_true.values.eval())
      self.assertAllEqual([0, 1], switch_true.indices.eval())

  def testIndexedSlicesGradient(self):
    with ops.Graph().as_default():
      embedding_matrix = tf.get_variable(
          "embedding_matrix", [5, 5],
          initializer=tf.random_normal_initializer())
      def Cond(it, _):
        return it < 5
      def Body(it, cost):
        embedding = embedding_ops.embedding_lookup(embedding_matrix + 0.0, [0])
        cost += tf.reduce_sum(embedding)
        return it + 1, cost
      _, cost = control_flow_ops.While(
          Cond, Body, [tf.constant(0), tf.constant(0.0)])
      optimizer = momentum.MomentumOptimizer(0.1, 0.9)
      train_op = optimizer.minimize(cost)
      with self.test_session() as sess:
        sess.run(tf.initialize_all_variables())
        for _ in range(10):
          sess.run([train_op])

  def testIndexedSlicesGradientInCondInWhileLoop(self):
    with ops.Graph().as_default():
      embedding_matrix = tf.get_variable(
          "embedding_matrix", [5, 5],
          initializer=tf.random_normal_initializer())

      def Cond(it, _):
        return it < 5
      def Body(it, cost):
        embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
        cost = tf.cond(tf.equal(it, 3),
                       lambda: tf.square(cost),
                       lambda: cost + tf.reduce_sum(embedding))
        return it + 1, cost
      _, cost = control_flow_ops.While(
          Cond, Body, [tf.constant(0), tf.constant(0.0)])

      dynamic_grads = tf.gradients(cost, [embedding_matrix])[0]
      dynamic_grads = tf.segment_sum(dynamic_grads.values,
                                     dynamic_grads.indices)

      embedding = embedding_ops.embedding_lookup(embedding_matrix, [0])
      static = tf.square(
          tf.reduce_sum(embedding) +
          tf.reduce_sum(embedding) +
          tf.reduce_sum(embedding)) + tf.reduce_sum(embedding)
      static_grads = tf.gradients(static, [embedding_matrix])[0]
      static_grads = tf.segment_sum(static_grads.values, static_grads.indices)

      with self.test_session() as sess:
        sess.run(tf.initialize_all_variables())
        self.assertAllEqual(*sess.run([static_grads, dynamic_grads]))


if __name__ == "__main__":
  googletest.main()
