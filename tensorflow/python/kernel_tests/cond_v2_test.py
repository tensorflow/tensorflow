# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for cond_v2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver
from tensorflow.python.util import compat


class NewCondTest(test.TestCase):

  def _testCond(self, true_fn, false_fn, train_vals):
    with self.test_session() as sess:
      pred = array_ops.placeholder(dtypes.bool, name="pred")

      expected = control_flow_ops.cond(pred, true_fn, false_fn, name="expected")
      actual = cond_v2.cond_v2(pred, true_fn, false_fn, name="actual")

      expected_grad = gradients_impl.gradients(expected, train_vals)
      actual_grad = gradients_impl.gradients(actual, train_vals)

      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), {pred: True})
      self.assertEqual(expected_val, actual_val)
      self.assertEqual(expected_grad_val, actual_grad_val)

      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), {pred: False})
      self.assertEqual(expected_val, actual_val)
      self.assertEqual(expected_grad_val, actual_grad_val)

  def testBasic(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():
      return x * 2.0

    def false_fn():
      return y * 3.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  def testBasic2(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():
      return x * y * 2.0

    def false_fn():
      return 2.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  def testNoInputs(self):
    with self.test_session() as sess:
      pred = array_ops.placeholder(dtypes.bool, name="pred")

      def true_fn():
        return constant_op.constant(1.0)

      def false_fn():
        return constant_op.constant(2.0)

      out = cond_v2.cond_v2(pred, true_fn, false_fn)

      self.assertEqual(sess.run(out, {pred: True}), [1.0])
      self.assertEqual(sess.run(out, {pred: False}), [2.0])

  def _createCond(self, name):
    pred = array_ops.placeholder(dtypes.bool, name="pred")
    x = constant_op.constant(1.0, name="x")

    def true_fn():
      return x

    def false_fn():
      return x + 1

    return cond_v2.cond_v2(pred, true_fn, false_fn, name=name)[0].op

  def testDefaultName(self):
    with ops.Graph().as_default():
      cond = self._createCond(None)
      self.assertEqual(cond.name, "cond")
      self.assertIn("cond_true", ops.get_default_graph()._functions)
      self.assertIn("cond_false", ops.get_default_graph()._functions)

    with ops.Graph().as_default():
      with ops.name_scope("foo"):
        cond = self._createCond("")
        self.assertEqual(cond.name, "foo/cond")
        self.assertIn("foo_cond_true", ops.get_default_graph()._functions)
        self.assertIn("foo_cond_false", ops.get_default_graph()._functions)

        cond2 = self._createCond(None)
        self.assertEqual(cond2.name, "foo/cond_1")
        self.assertIn("foo_cond_1_true", ops.get_default_graph()._functions)
        self.assertIn("foo_cond_1_false", ops.get_default_graph()._functions)

  def testSecondDerivative(self):
    with self.test_session() as sess:
      pred = array_ops.placeholder(dtypes.bool, name="pred")
      x = constant_op.constant(3.0, name="x")

      def true_fn():
        return math_ops.pow(x, 3)

      def false_fn():
        return x

      cond = cond_v2.cond_v2(pred, true_fn, false_fn, name="cond")
      cond_grad = gradients_impl.gradients(cond, [x])
      cond_grad_grad = gradients_impl.gradients(cond_grad, [x])

      # d[x^3]/dx = 3x^2
      true_val = sess.run(cond_grad, {pred: True})
      self.assertEqual(true_val, [27.0])
      # d[x]/dx = 1
      false_val = sess.run(cond_grad, {pred: False})
      self.assertEqual(false_val, [1.0])

      true_val = sess.run(cond_grad_grad, {pred: True})
      # d2[x^3]/dx2 = 6x
      self.assertEqual(true_val, [18.0])
      false_val = sess.run(cond_grad_grad, {pred: False})
      # d2[x]/dx2 = 0
      self.assertEqual(false_val, [0.0])

  def testGradientOfDeserializedCond(self):
    with ops.Graph().as_default():
      pred = array_ops.placeholder(dtypes.bool, name="pred")
      x = constant_op.constant(3.0, name="x")
      ops.add_to_collection("x", x)

      def true_fn():
        return math_ops.pow(x, 3)

      def false_fn():
        return x

      ops.add_to_collection("pred", pred)
      cond = cond_v2.cond_v2(pred, true_fn, false_fn, name="cond")
      for c in cond:
        ops.add_to_collection("cond", c)
      meta_graph = saver.export_meta_graph()

    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        saver.import_meta_graph(meta_graph)
        x = ops.get_collection("x")[0]
        pred = ops.get_collection("pred")[0]
        cond = ops.get_collection("cond")
        cond_grad = gradients_impl.gradients(cond, [x], name="cond_grad")
        cond_grad_grad = gradients_impl.gradients(
            cond_grad, [x], name="cond_grad_grad")
        # d[x^3]/dx = 3x^2
        true_val = sess.run(cond_grad, {pred: True})
        self.assertEqual(true_val, [27.0])
        # d[x]/dx = 1
        false_val = sess.run(cond_grad, {pred: False})
        self.assertEqual(false_val, [1.0])

        true_val = sess.run(cond_grad_grad, {pred: True})
        # d2[x^3]/dx2 = 6x
        self.assertEqual(true_val, [18.0])
        false_val = sess.run(cond_grad_grad, {pred: False})
        # d2[x]/dx2 = 0
        self.assertEqual(false_val, [0.0])


class CondV2CollectionTest(test.TestCase):

  def testCollectionIntValueAccessInCond(self):
    """Read values from graph collections inside of cond_v2."""
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g):
        x = 2
        y = 5
        ops.add_to_collection("x", x)
        ops.add_to_collection("y", y)
        def fn():
          x_const = constant_op.constant(ops.get_collection("x")[0])
          y_const = constant_op.constant(ops.get_collection("y")[0])
          return math_ops.add(x_const, y_const)

        cnd = cond_v2.cond_v2(True, fn, fn)
        self.assertEquals(cnd[0].eval(), 7)

  def testCollectionTensorValueAccessInCond(self):
    """Read tensors from collections inside of cond_v2 & use them."""
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g):
        x = constant_op.constant(2)
        y = constant_op.constant(5)
        ops.add_to_collection("x", x)
        ops.add_to_collection("y", y)

        def fn():
          x_read = ops.get_collection("x")[0]
          y_read = ops.get_collection("y")[0]
          return math_ops.add(x_read, y_read)

        cnd = cond_v2.cond_v2(math_ops.less(x, y), fn, fn)
        self.assertEquals(cnd[0].eval(), 7)

  def testCollectionIntValueWriteInCond(self):
    """Make sure Int writes to collections work inside of cond_v2."""
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g):
        x = constant_op.constant(2)
        y = constant_op.constant(5)
        def true_fn():
          z = math_ops.add(x, y)
          ops.add_to_collection("z", 7)
          return math_ops.mul(x, z)

        def false_fn():
          z = math_ops.add(x, y)
          return math_ops.mul(x, z)

        cnd = cond_v2.cond_v2(
            True, true_fn,
            false_fn)
        self.assertEquals(cnd[0].eval(), 14)

        read_z_collection = ops.get_collection("z")
        self.assertEquals(read_z_collection, [7])


class CondV2ContainerTest(test.TestCase):

  def testContainer(self):
    """Set containers outside & inside of cond_v2.

    Make sure the containers are set correctly for both variable creation
    (tested by variables.Variable) and for stateful ops (tested by FIFOQueue)
    """
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g):

        v0 = variables.Variable([0])
        q0 = data_flow_ops.FIFOQueue(1, dtypes.float32)

        def container(node):
          return node.op.get_attr("container")

        self.assertEqual(compat.as_bytes(""), container(v0))
        self.assertEqual(compat.as_bytes(""), container(q0.queue_ref))

        def true_fn():
          # When this branch is created in cond below,
          # the container should begin with 'l1'
          v1 = variables.Variable([1])
          q1 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          with ops.container("l2t"):
            v2 = variables.Variable([2])
            q2 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          v3 = variables.Variable([1])
          q3 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          self.assertEqual(compat.as_bytes("l1"), container(v1))
          self.assertEqual(compat.as_bytes("l1"), container(q1.queue_ref))
          self.assertEqual(compat.as_bytes("l2t"), container(v2))
          self.assertEqual(compat.as_bytes("l2t"), container(q2.queue_ref))
          self.assertEqual(compat.as_bytes("l1"), container(v3))
          self.assertEqual(compat.as_bytes("l1"), container(q3.queue_ref))

          return constant_op.constant(2.0)

        def false_fn():
          # When this branch is created in cond below,
          # the container should begin with 'l1'
          v1 = variables.Variable([1])
          q1 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          with ops.container("l2f"):
            v2 = variables.Variable([2])
            q2 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          v3 = variables.Variable([1])
          q3 = data_flow_ops.FIFOQueue(1, dtypes.float32)

          self.assertEqual(compat.as_bytes("l1"), container(v1))
          self.assertEqual(compat.as_bytes("l1"), container(q1.queue_ref))
          self.assertEqual(compat.as_bytes("l2f"), container(v2))
          self.assertEqual(compat.as_bytes("l2f"), container(q2.queue_ref))
          self.assertEqual(compat.as_bytes("l1"), container(v3))
          self.assertEqual(compat.as_bytes("l1"), container(q3.queue_ref))

          return constant_op.constant(6.0)

        with ops.container("l1"):
          cnd_true = cond_v2.cond_v2(True, true_fn, false_fn)
          self.assertEquals(cnd_true[0].eval(), 2)

          cnd_false = cond_v2.cond_v2(False, true_fn, false_fn)
          self.assertEquals(cnd_false[0].eval(), 6)

          v4 = variables.Variable([3])
          q4 = data_flow_ops.FIFOQueue(1, dtypes.float32)
        v5 = variables.Variable([4])
        q5 = data_flow_ops.FIFOQueue(1, dtypes.float32)

      self.assertEqual(compat.as_bytes("l1"), container(v4))
      self.assertEqual(compat.as_bytes("l1"), container(q4.queue_ref))
      self.assertEqual(compat.as_bytes(""), container(v5))
      self.assertEqual(compat.as_bytes(""), container(q5.queue_ref))


class CondV2ColocationGroupAndDeviceTest(test.TestCase):

  def testColocateWithBeforeCond(self):
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g):

        a = constant_op.constant([2.0], name="a")
        b = constant_op.constant([2.0], name="b")

        def fn():
          c = constant_op.constant(3.0)
          self.assertEqual([b"loc:@a"], c.op.colocation_groups())
          return c

        with ops.colocate_with(a.op):
          self.assertEquals(cond_v2.cond_v2(True, fn, fn)[0].eval(), 3)

        def fn2():
          c = constant_op.constant(3.0)
          self.assertEqual([b"loc:@a", b"loc:@b"], c.op.colocation_groups())
          return c

        with ops.colocate_with(a.op):
          with ops.colocate_with(b.op):
            self.assertEquals(cond_v2.cond_v2(True, fn2, fn2)[0].eval(), 3)

  def testColocateWithInAndOutOfCond(self):
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g):

        a = constant_op.constant([2.0], name="a")
        b = constant_op.constant([2.0], name="b")

        def fn2():
          with ops.colocate_with(b.op):
            c = constant_op.constant(3.0)
            self.assertEqual([b"loc:@a", b"loc:@b"], c.op.colocation_groups())
            return c

        with ops.colocate_with(a.op):
          self.assertEquals(cond_v2.cond_v2(True, fn2, fn2)[0].eval(), 3)

          d = constant_op.constant([2.0], name="d")
          self.assertEqual([b"loc:@a"], d.op.colocation_groups())

  def testDeviceBeforeCond(self):
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g):
        def fn():
          c = constant_op.constant(3.0)
          self.assertEqual("/device:CPU:0", c.op.device)
          return c

        with ops.device("/device:CPU:0"):
          self.assertEquals(cond_v2.cond_v2(True, fn, fn)[0].eval(), 3)

        def fn2():
          c = constant_op.constant(3.0)
          self.assertEqual("/device:GPU:0", c.op.device)
          return c

        with ops.device("/device:GPU:0"):
          self.assertEquals(cond_v2.cond_v2(True, fn2, fn2)[0].eval(), 3)

  def testDeviceInAndOutOfCond(self):
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g):
        def fn2():
          with ops.device("/device:GPU:0"):
            c = constant_op.constant(3.0)
            self.assertEqual("/device:GPU:0", c.op.device)
            return c

        with ops.device("/device:CPU:0"):
          self.assertEquals(cond_v2.cond_v2(True, fn2, fn2)[0].eval(), 3)

          d = constant_op.constant(4.0)
          self.assertEqual("/device:CPU:0", d.op.device)

if __name__ == "__main__":
  test.main()
