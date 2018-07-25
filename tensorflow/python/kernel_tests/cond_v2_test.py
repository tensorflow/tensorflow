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

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
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


class CondV2Test(test.TestCase):

  def _testCond(self, true_fn, false_fn, train_vals, feed_dict=None):
    if not feed_dict:
      feed_dict = {}
    with self.test_session(graph=ops.get_default_graph()) as sess:
      pred = array_ops.placeholder(dtypes.bool, name="pred")

      expected = control_flow_ops.cond(pred, true_fn, false_fn, name="expected")
      actual = cond_v2.cond_v2(pred, true_fn, false_fn, name="actual")

      expected_grad = gradients_impl.gradients(expected, train_vals)
      actual_grad = gradients_impl.gradients(actual, train_vals)

      sess_run_args = {pred: True}
      sess_run_args.update(feed_dict)
      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), sess_run_args)
      self.assertEqual(expected_val, actual_val)
      self.assertEqual(expected_grad_val, actual_grad_val)

      sess_run_args = {pred: False}
      sess_run_args.update(feed_dict)
      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), sess_run_args)
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
    pred = constant_op.constant(True, name="pred")
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

  def testDefunInCond(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():

      @function.Defun()
      def fn():
        return x * y * 2.0

      return fn()

    def false_fn():
      return 2.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  def testNestedDefunInCond(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():
      return 2.0

    def false_fn():

      @function.Defun()
      def fn():

        @function.Defun()
        def nested_fn():
          return x * y * 2.0

        return nested_fn()

      return fn()

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  def testDoubleNestedDefunInCond(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():

      @function.Defun()
      def fn():

        @function.Defun()
        def nested_fn():

          @function.Defun()
          def nested_nested_fn():
            return x * y * 2.0

          return nested_nested_fn()

        return nested_fn()

      return fn()

    def false_fn():
      return 2.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  def testNestedCond(self):

    def run_test(pred_value):

      def build_graph():
        pred = array_ops.placeholder(dtypes.bool, name="pred")
        x = constant_op.constant(1.0, name="x")
        y = constant_op.constant(2.0, name="y")

        def true_fn():
          return 2.0

        def false_fn():

          def false_true_fn():
            return x * y * 2.0

          def false_false_fn():
            return x * 5.0

          return _cond(pred, false_true_fn, false_false_fn, "inside_false_fn")

        return x, y, pred, true_fn, false_fn

      with ops.Graph().as_default():
        x, y, pred, true_fn, false_fn = build_graph()
        self._testCond(true_fn, false_fn, [x, y], {pred: pred_value})
        self._testCond(true_fn, false_fn, [x], {pred: pred_value})
        self._testCond(true_fn, false_fn, [y], {pred: pred_value})

    run_test(True)
    run_test(False)

  def testDoubleNestedCond(self):

    def run_test(pred1_value, pred2_value):

      def build_graph():
        pred1 = array_ops.placeholder(dtypes.bool, name="pred1")
        pred2 = array_ops.placeholder(dtypes.bool, name="pred2")
        x = constant_op.constant(1.0, name="x")
        y = constant_op.constant(2.0, name="y")

        def true_fn():
          return 2.0

        def false_fn():

          def false_true_fn():

            def false_true_true_fn():
              return x * y * 2.0

            def false_true_false_fn():
              return x * 10.0

            return _cond(
                pred1,
                false_true_true_fn,
                false_true_false_fn,
                name="inside_false_true_fn")

          def false_false_fn():
            return x * 5.0

          return _cond(
              pred2, false_true_fn, false_false_fn, name="inside_false_fn")

        return x, y, pred1, pred2, true_fn, false_fn

      with ops.Graph().as_default():
        x, y, pred1, pred2, true_fn, false_fn = build_graph()
        self._testCond(true_fn, false_fn, [x, y], {
            pred1: pred1_value,
            pred2: pred2_value
        })
        x, y, pred1, pred2, true_fn, false_fn = build_graph()
        self._testCond(true_fn, false_fn, [x], {
            pred1: pred1_value,
            pred2: pred2_value
        })
        x, y, pred1, pred2, true_fn, false_fn = build_graph()
        self._testCond(true_fn, false_fn, [y], {
            pred1: pred1_value,
            pred2: pred2_value
        })

    run_test(True, True)
    run_test(True, False)
    run_test(False, False)
    run_test(False, True)

  def testGradientFromInsideDefun(self):

    def build_graph():
      pred_outer = array_ops.placeholder(dtypes.bool, name="pred_outer")
      pred_inner = array_ops.placeholder(dtypes.bool, name="pred_inner")
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(2.0, name="y")

      def true_fn():
        return 2.0

      def false_fn():

        def inner_true_fn():
          return x * y * 2.0

        def inner_false_fn():
          return x * 5.0

        return cond_v2.cond_v2(
            pred_inner, inner_true_fn, inner_false_fn, name="inner_cond")

      cond_outer = cond_v2.cond_v2(
          pred_outer, true_fn, false_fn, name="outer_cond")

      # Compute grads inside a Defun.
      @function.Defun()
      def nesting_fn():
        return gradients_impl.gradients(cond_outer, [x, y])

      grads = nesting_fn()

      return grads, pred_outer, pred_inner

    with ops.Graph().as_default():
      grads, pred_outer, pred_inner = build_graph()
      with self.test_session(graph=ops.get_default_graph()) as sess:
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: True,
                pred_inner: True
            }), [0., 0.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: True,
                pred_inner: False
            }), [0., 0.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: False,
                pred_inner: True
            }), [4., 2.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: False,
                pred_inner: False
            }), [5., 0.])

  def testGradientFromInsideNestedDefun(self):

    def build_graph():
      pred_outer = array_ops.placeholder(dtypes.bool, name="pred_outer")
      pred_inner = array_ops.placeholder(dtypes.bool, name="pred_inner")
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(2.0, name="y")

      def true_fn():
        return 2.0

      def false_fn():

        def inner_true_fn():
          return x * y * 2.0

        def inner_false_fn():
          return x * 5.0

        return cond_v2.cond_v2(
            pred_inner, inner_true_fn, inner_false_fn, name="inner_cond")

      cond_outer = cond_v2.cond_v2(
          pred_outer, true_fn, false_fn, name="outer_cond")

      # Compute grads inside a Defun.
      @function.Defun()
      def nesting_fn():

        @function.Defun()
        def inner_nesting_fn():
          return gradients_impl.gradients(cond_outer, [x, y])

        return inner_nesting_fn()

      grads = nesting_fn()

      return grads, pred_outer, pred_inner

    with ops.Graph().as_default():
      grads, pred_outer, pred_inner = build_graph()
      with self.test_session(graph=ops.get_default_graph()) as sess:
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: True,
                pred_inner: True
            }), [0., 0.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: True,
                pred_inner: False
            }), [0., 0.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: False,
                pred_inner: True
            }), [4., 2.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: False,
                pred_inner: False
            }), [5., 0.])

  def testBuildCondAndGradientInsideDefun(self):

    def build_graph():
      pred_outer = array_ops.placeholder(dtypes.bool, name="pred_outer")
      pred_inner = array_ops.placeholder(dtypes.bool, name="pred_inner")
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(2.0, name="y")

      # Build cond and its gradient inside a Defun.
      @function.Defun()
      def fn():

        def true_fn():
          return 2.0

        def false_fn():

          def inner_true_fn():
            return x * y * 2.0

          def inner_false_fn():
            return x * 5.0

          return cond_v2.cond_v2(
              pred_inner, inner_true_fn, inner_false_fn, name="inner_cond")

        cond_outer = cond_v2.cond_v2(
            pred_outer, true_fn, false_fn, name="outer_cond")
        return gradients_impl.gradients(cond_outer, [x, y])

      grads = fn()

      return grads, pred_outer, pred_inner

    with ops.Graph().as_default():
      grads, pred_outer, pred_inner = build_graph()
      with self.test_session(graph=ops.get_default_graph()) as sess:
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: True,
                pred_inner: True
            }), [0., 0.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: True,
                pred_inner: False
            }), [0., 0.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: False,
                pred_inner: True
            }), [4., 2.])
        self.assertSequenceEqual(
            sess.run(grads, {
                pred_outer: False,
                pred_inner: False
            }), [5., 0.])

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

  def testLowering(self):
    with ops.Graph().as_default() as g:
      with self.test_session(graph=g) as sess:
        out_cond = self._createCond("cond")

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        run_metadata = config_pb2.RunMetadata()
        sess.run(out_cond, options=run_options, run_metadata=run_metadata)

        # If lowering was enabled, there should be a `Switch` node
        switch_found = any(
            any(node.op == "Switch" for node in graph.node)
            for graph in run_metadata.partition_graphs
        )

        self.assertTrue(switch_found,
                        "A `Switch` op should exist if the graph was lowered.")

        # If lowering was enabled, there should be no `If` node
        if_found = any(
            any(node.op == "If" for node in graph.node)
            for graph in run_metadata.partition_graphs
        )

        self.assertFalse(if_found,
                         "An `If` op was found, but it should be lowered.")

  def testLoweringDisabledInXLA(self):
    with self.test_session(graph=ops.Graph()) as sess:
      # Build the cond_v2 in an XLA context
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      out_cond = self._createCond("cond")
      xla_context.Exit()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      run_metadata = config_pb2.RunMetadata()
      sess.run(out_cond, options=run_options, run_metadata=run_metadata)

      # Lowering disabled in XLA, there should be no `Switch` node
      switch_found = any(
          any(node.op == "Switch" for node in graph.node)
          for graph in run_metadata.partition_graphs
      )

      self.assertFalse(
          switch_found,
          "A `Switch` op exists, but the graph should not be lowered.")

      # Lowering disabled in XLA, there should still be an `If` node
      if_found = any(
          any(node.op == "If" for node in graph.node)
          for graph in run_metadata.partition_graphs
      )

      self.assertTrue(
          if_found,
          "An `If` op was not found, but the graph should not be lowered.")


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

  def testColocateWithInCondGraphPartitioning(self):
    with ops.Graph().as_default() as g:
      with self.test_session(
          graph=g,
          config=config_pb2.ConfigProto(device_count={"CPU": 2})
      ) as sess:

        with ops.device("/device:CPU:0"):
          a = constant_op.constant([2.0], name="a")
        with ops.device("/device:CPU:1"):
          b = constant_op.constant([2.0], name="b")

        def fn():
          with ops.colocate_with(b.op):
            c = math_ops.add(a, a, name="c")
          return c
        out_cond_2 = cond_v2.cond_v2(True, fn, fn)[0]

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        run_metadata = config_pb2.RunMetadata()
        sess.run(out_cond_2, options=run_options, run_metadata=run_metadata)

        # We expect there to be two partitions because of the
        # colocate_with. We are only running the cond, which has a data
        # dependency on `a` but not on `b`. So, without the colocate_with
        # we would expect execution on just one device.
        self.assertTrue(len(run_metadata.partition_graphs) >= 2)

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

  def testDeviceInCondGraphPartitioning(self):
    with ops.Graph().as_default() as g:
      with self.test_session(
          graph=g,
          config=config_pb2.ConfigProto(device_count={"CPU": 2})
      ) as sess:

        def fn():
          with ops.device("/device:CPU:1"):
            c = math_ops.add(a, a, name="c")
          return c

        with ops.device("/device:CPU:0"):
          a = constant_op.constant([2.0], name="a")
          out_cond_2 = cond_v2.cond_v2(True, fn, fn)[0]

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        run_metadata = config_pb2.RunMetadata()
        sess.run(out_cond_2, options=run_options, run_metadata=run_metadata)

        self.assertTrue(len(run_metadata.partition_graphs) >= 2)


def _cond(pred, true_fn, false_fn, name):
  if _is_old_cond():
    return control_flow_ops.cond(pred, true_fn, false_fn, name=name)
  else:
    return cond_v2.cond_v2(pred, true_fn, false_fn, name=name)


def _is_old_cond():
  return isinstance(ops.get_default_graph()._get_control_flow_context(),
                    control_flow_ops.CondContext)


if __name__ == "__main__":
  test.main()
