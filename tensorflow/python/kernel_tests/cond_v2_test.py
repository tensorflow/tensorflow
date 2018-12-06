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
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import saver
from tensorflow.python.util import compat


class CondV2Test(test.TestCase):

  def _testCond(self, true_fn, false_fn, train_vals, feed_dict=None):
    if not feed_dict:
      feed_dict = {}
    with self.session(graph=ops.get_default_graph()) as sess:
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

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
  def testMultipleOutputs(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(3.0, name="y")

    def true_fn():
      return x * y, y

    def false_fn():
      return x, y * 3.0

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  @test_util.run_deprecated_v1
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

  @test_util.run_deprecated_v1
  def testNoInputs(self):
    with self.cached_session() as sess:
      pred = array_ops.placeholder(dtypes.bool, name="pred")

      def true_fn():
        return constant_op.constant(1.0)

      def false_fn():
        return constant_op.constant(2.0)

      out = cond_v2.cond_v2(pred, true_fn, false_fn)

      self.assertEqual(sess.run(out, {pred: True}), (1.0,))
      self.assertEqual(sess.run(out, {pred: False}), (2.0,))

  def _createCond(self, name):
    """Creates a cond_v2 call and returns the output tensor and the cond op."""
    pred = constant_op.constant(True, name="pred")
    x = constant_op.constant(1.0, name="x")

    def true_fn():
      return x

    def false_fn():
      return x + 1

    output = cond_v2.cond_v2(pred, true_fn, false_fn, name=name)
    cond_op = output.op.inputs[0].op
    self.assertEqual(cond_op.type, "If")
    return output, cond_op

  def testDefaultName(self):
    with ops.Graph().as_default():
      _, cond_op = self._createCond(None)
      self.assertEqual(cond_op.name, "cond")
      self.assertRegexpMatches(
          cond_op.get_attr("then_branch").name, r"cond_true_\d*")
      self.assertRegexpMatches(
          cond_op.get_attr("else_branch").name, r"cond_false_\d*")

    with ops.Graph().as_default():
      with ops.name_scope("foo"):
        _, cond1_op = self._createCond("")
        self.assertEqual(cond1_op.name, "foo/cond")
        self.assertRegexpMatches(
            cond1_op.get_attr("then_branch").name, r"foo_cond_true_\d*")
        self.assertRegexpMatches(
            cond1_op.get_attr("else_branch").name, r"foo_cond_false_\d*")

        _, cond2_op = self._createCond(None)
        self.assertEqual(cond2_op.name, "foo/cond_1")
        self.assertRegexpMatches(
            cond2_op.get_attr("then_branch").name, r"foo_cond_1_true_\d*")
        self.assertRegexpMatches(
            cond2_op.get_attr("else_branch").name, r"foo_cond_1_false_\d*")

  def testDefunInCond(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():

      @function.defun
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

      @function.defun
      def fn():

        @function.defun
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

      @function.defun
      def fn():

        @function.defun
        def nested_fn():

          @function.defun
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

  def testNestedCondBothBranches(self):

    def run_test(pred_value):

      def build_graph():
        pred = array_ops.placeholder(dtypes.bool, name="pred")
        x = constant_op.constant(1.0, name="x")
        y = constant_op.constant(2.0, name="y")

        def true_fn():
          return _cond(pred, lambda: x + y, lambda: x * x, name=None)

        def false_fn():
          return _cond(pred, lambda: x - y, lambda: y * y, name=None)

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
      @function.defun
      def nesting_fn():
        return gradients_impl.gradients(cond_outer, [x, y])

      grads = nesting_fn()

      return grads, pred_outer, pred_inner

    with ops.Graph().as_default():
      grads, pred_outer, pred_inner = build_graph()
      with self.session(graph=ops.get_default_graph()) as sess:
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
      @function.defun
      def nesting_fn():

        @function.defun
        def inner_nesting_fn():
          return gradients_impl.gradients(cond_outer, [x, y])

        return inner_nesting_fn()

      grads = nesting_fn()

      return grads, pred_outer, pred_inner

    with ops.Graph().as_default():
      grads, pred_outer, pred_inner = build_graph()
      with self.session(graph=ops.get_default_graph()) as sess:
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
      @function.defun
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

    with ops.Graph().as_default(), self.session(
        graph=ops.get_default_graph()) as sess:
      grads, pred_outer, pred_inner = build_graph()
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

  @test_util.run_deprecated_v1
  def testSecondDerivative(self):
    with self.cached_session() as sess:
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
      ops.add_to_collection("cond", cond)
      meta_graph = saver.export_meta_graph()

    with ops.Graph().as_default() as g:
      with self.session(graph=g) as sess:
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
      with self.session(graph=g) as sess:
        cond_output, _ = self._createCond("cond")

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        run_metadata = config_pb2.RunMetadata()
        sess.run(cond_output, options=run_options, run_metadata=run_metadata)

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

  @test_util.run_deprecated_v1
  def testLoweringDisabledInXLA(self):
    with self.session(graph=ops.Graph()) as sess:
      # Build the cond_v2 in an XLA context
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      cond_output, _ = self._createCond("cond")
      xla_context.Exit()

      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      run_metadata = config_pb2.RunMetadata()
      sess.run(cond_output, options=run_options, run_metadata=run_metadata)

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

  @test_util.run_deprecated_v1
  def testLoweringDisabledWithSingleThreadedExecutorContext(self):
    with self.session(graph=ops.Graph()) as sess:
      @function.defun
      def _add_cond(x):
        return cond_v2.cond_v2(
            constant_op.constant(True, name="pred"),
            lambda: x,
            lambda: x + 1)

      x = array_ops.placeholder(shape=None, dtype=dtypes.float32)
      with context.function_executor_type("SINGLE_THREADED_EXECUTOR"):
        out_cond = _add_cond(x)

      # The fact that sess.run() succeeds means lowering is disabled, because
      # the single threaded executor does not support cond v1 ops.
      sess.run(out_cond, feed_dict={x: 1.0})

  @test_util.enable_control_flow_v2
  def testStructuredOutputs(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(3.0, name="y")

    def true_fn():
      return ((x * y,), y)

    def false_fn():
      return ((x,), y * 3.0)

    output = control_flow_ops.cond(
        constant_op.constant(False), true_fn, false_fn)
    self.assertEqual(self.evaluate(output[0][0]), 1.)
    self.assertEqual(self.evaluate(output[1]), 9.)

  @test_util.enable_control_flow_v2
  @test_util.run_deprecated_v1
  def testRaisesOutputStructuresMismatch(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(3.0, name="y")

    def true_fn():
      return x * y, y

    def false_fn():
      return ((x,), y * 3.0)

    with self.assertRaisesRegexp(
        ValueError, "Outputs of true_fn and false_fn must"
        " have the same structure"):
      control_flow_ops.cond(constant_op.constant(False), true_fn, false_fn)

  @test_util.enable_control_flow_v2
  def testCondAndTensorArray(self):
    x = math_ops.range(-5, 5)
    output = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=x.shape[0])

    def loop_body(i, output):

      def if_true():
        return output.write(i, x[i]**2)

      def if_false():
        return output.write(i, x[i])

      output = control_flow_ops.cond(x[i] > 0, if_true, if_false)
      return i + 1, output

    _, output = control_flow_ops.while_loop(
        lambda i, arr: i < x.shape[0],
        loop_body,
        loop_vars=(constant_op.constant(0), output))
    output_t = output.stack()
    self.assertAllEqual(
        self.evaluate(output_t), [-5, -4, -3, -2, -1, 0, 1, 4, 9, 16])

  @test_util.enable_control_flow_v2
  def testCondAndTensorArrayInDefun(self):

    @function.defun
    def f():
      x = math_ops.range(-5, 5)
      output = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=x.shape[0])

      def loop_body(i, output):

        def if_true():
          return output.write(i, x[i]**2)

        def if_false():
          return output.write(i, x[i])

        output = control_flow_ops.cond(x[i] > 0, if_true, if_false)
        return i + 1, output

      _, output = control_flow_ops.while_loop(
          lambda i, arr: i < x.shape[0],
          loop_body,
          loop_vars=(constant_op.constant(0), output))
      return output.stack()

    output_t = f()
    self.assertAllEqual(
        self.evaluate(output_t), [-5, -4, -3, -2, -1, 0, 1, 4, 9, 16])

  def testForwardPassRewrite(self):
    x = constant_op.constant(1.0, name="x")
    output = cond_v2.cond_v2(constant_op.constant(True),
                             lambda: x * 2.0,
                             lambda: x)
    if_op = output.op.inputs[0].op
    self.assertEqual(if_op.type, "If")
    # pylint: disable=g-deprecated-assert
    self.assertEqual(len(if_op.outputs), 1)

    gradients_impl.gradients(output, x)
    # if_op should have been rewritten to output 2.0 intermediate.
    self.assertEqual(len(if_op.outputs), 2)

    gradients_impl.gradients(output, x)
    # Computing the gradient again shouldn't rewrite if_op again.
    self.assertEqual(len(if_op.outputs), 2)
    # pylint: enable=g-deprecated-assert


class CondV2CollectionTest(test.TestCase):

  def testCollectionIntValueAccessInCond(self):
    """Read values from graph collections inside of cond_v2."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = 2
        y = 5
        ops.add_to_collection("x", x)
        ops.add_to_collection("y", y)
        def fn():
          x_const = constant_op.constant(ops.get_collection("x")[0])
          y_const = constant_op.constant(ops.get_collection("y")[0])
          return math_ops.add(x_const, y_const)

        cnd = cond_v2.cond_v2(constant_op.constant(True), fn, fn)
        self.assertEquals(cnd.eval(), 7)

  def testCollectionTensorValueAccessInCond(self):
    """Read tensors from collections inside of cond_v2 & use them."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = constant_op.constant(2)
        y = constant_op.constant(5)
        ops.add_to_collection("x", x)
        ops.add_to_collection("y", y)

        def fn():
          x_read = ops.get_collection("x")[0]
          y_read = ops.get_collection("y")[0]
          return math_ops.add(x_read, y_read)

        cnd = cond_v2.cond_v2(math_ops.less(x, y), fn, fn)
        self.assertEquals(cnd.eval(), 7)

  def testCollectionIntValueWriteInCond(self):
    """Make sure Int writes to collections work inside of cond_v2."""
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = constant_op.constant(2)
        y = constant_op.constant(5)
        def true_fn():
          z = math_ops.add(x, y)
          ops.add_to_collection("z", 7)
          return math_ops.mul(x, z)

        def false_fn():
          z = math_ops.add(x, y)
          return math_ops.mul(x, z)

        cnd = cond_v2.cond_v2(constant_op.constant(True), true_fn, false_fn)
        self.assertEquals(cnd.eval(), 14)

        read_z_collection = ops.get_collection("z")
        self.assertEquals(read_z_collection, [7])


class CondV2ContainerTest(test.TestCase):

  def testContainer(self):
    """Set containers outside & inside of cond_v2.

    Make sure the containers are set correctly for both variable creation
    (tested by variables.Variable) and for stateful ops (tested by FIFOQueue)
    """
    self.skipTest("b/113048653")
    with ops.Graph().as_default() as g:
      with self.session(graph=g):

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
          cnd_true = cond_v2.cond_v2(
              constant_op.constant(True), true_fn, false_fn)
          self.assertEquals(cnd_true.eval(), 2)

          cnd_false = cond_v2.cond_v2(
              constant_op.constant(False), true_fn, false_fn)
          self.assertEquals(cnd_false.eval(), 6)

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
      with self.session(graph=g):

        a = constant_op.constant([2.0], name="a")
        b = constant_op.constant([2.0], name="b")

        def fn():
          c = constant_op.constant(3.0)
          self.assertEqual([b"loc:@a"], c.op.colocation_groups())
          return c

        with ops.colocate_with(a.op):
          self.assertEquals(
              cond_v2.cond_v2(constant_op.constant(True), fn, fn).eval(), 3)

        def fn2():
          c = constant_op.constant(3.0)
          self.assertEqual([b"loc:@a", b"loc:@b"], c.op.colocation_groups())
          return c

        with ops.colocate_with(a.op):
          with ops.colocate_with(b.op):
            self.assertEquals(
                cond_v2.cond_v2(constant_op.constant(True), fn2, fn2).eval(), 3)

  def testColocateWithInAndOutOfCond(self):
    with ops.Graph().as_default() as g:
      with self.session(graph=g):

        a = constant_op.constant([2.0], name="a")
        b = constant_op.constant([2.0], name="b")

        def fn2():
          with ops.colocate_with(b.op):
            c = constant_op.constant(3.0)
            self.assertEqual([b"loc:@a", b"loc:@b"], c.op.colocation_groups())
            return c

        with ops.colocate_with(a.op):
          self.assertEquals(
              cond_v2.cond_v2(constant_op.constant(True), fn2, fn2).eval(), 3)

          d = constant_op.constant([2.0], name="d")
          self.assertEqual([b"loc:@a"], d.op.colocation_groups())

  def testColocateWithInCondGraphPartitioning(self):
    with ops.Graph().as_default() as g:
      with self.session(
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
        out_cond_2 = cond_v2.cond_v2(constant_op.constant(True), fn, fn)

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
      with self.session(graph=g):

        def fn():
          self.assertEqual("", constant_op.constant(3.0).op.device)
          return test_ops.device_placement_op()

        with ops.device("/device:CPU:0"):
          self.assertIn(
              compat.as_bytes("CPU:0"),
              self.evaluate(cond_v2.cond_v2(constant_op.constant(True),
                                            fn, fn)))

        def fn2():
          self.assertEqual("", constant_op.constant(3.0).op.device)
          return test_ops.device_placement_op()

        if test_util.is_gpu_available():
          with ops.device("/device:GPU:0"):
            self.assertIn(
                compat.as_bytes("GPU:0"),
                self.evaluate(cond_v2.cond_v2(constant_op.constant(True),
                                              fn2, fn2)))
        else:
          self.skipTest("Test requires a GPU to check GPU device placement.")

  def testDeviceInAndOutOfCond(self):
    with ops.Graph().as_default() as g:
      with self.session(
          graph=g, config=config_pb2.ConfigProto(device_count={"CPU": 2})):

        def fn2():
          with ops.device("/device:CPU:1"):
            c = constant_op.constant(3.0)
            self.assertEqual("/device:CPU:1", c.op.device)
            return c

        with ops.device("/device:CPU:0"):
          self.assertEquals(
              cond_v2.cond_v2(constant_op.constant(True), fn2, fn2).eval(), 3)

          d = constant_op.constant(4.0)
          self.assertEqual("/device:CPU:0", d.op.device)

  def testDeviceInCondGraphPartitioning(self):
    with ops.Graph().as_default() as g:
      with self.session(
          graph=g,
          config=config_pb2.ConfigProto(device_count={"CPU": 2})
      ) as sess:

        def fn():
          with ops.device("/device:CPU:1"):
            c = math_ops.add(a, a, name="c")
          return c

        with ops.device("/device:CPU:0"):
          a = constant_op.constant([2.0], name="a")
          out_cond_2 = cond_v2.cond_v2(constant_op.constant(True), fn, fn)

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
