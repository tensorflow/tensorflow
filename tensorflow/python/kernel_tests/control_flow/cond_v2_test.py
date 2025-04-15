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

import os

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.module import module as module_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import cond_v2
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_lib
from tensorflow.python.saved_model import save as save_lib
from tensorflow.python.training import saver
from tensorflow.python.util import compat


_OPTIONAL_OPS = frozenset([
    "OptionalFromValue", "OptionalNone", "OptionalHasValue", "OptionalGetValue"
])


class CondV2Test(test.TestCase):

  def _testCond(self, true_fn, false_fn, train_vals, feed_dict=None):
    if not feed_dict:
      feed_dict = {}
    with self.session(graph=ops.get_default_graph()) as sess:
      pred = array_ops.placeholder(dtypes.bool, name="pred")

      expected = tf_cond.cond(
          array_ops.squeeze_v2(pred), true_fn, false_fn, name="expected")
      actual = cond_v2.cond_v2(pred, true_fn, false_fn, name="actual")

      expected_grad = gradients_impl.gradients(expected, train_vals)
      actual_grad = gradients_impl.gradients(actual, train_vals)

      sess_run_args = {pred: True}
      sess_run_args.update(feed_dict)
      expected_val, actual_val, expected_grad_val, actual_grad_val = sess.run(
          (expected, actual, expected_grad, actual_grad), sess_run_args)
      self.assertEqual(expected_val, actual_val)
      self.assertEqual(expected_grad_val, actual_grad_val)

      sess_run_args = {pred: [[True]]}
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

      sess_run_args = {pred: [[False]]}
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

  def testReturnsIndexedSlicesAndNones(self):
    @def_function.function
    def build_cond_with_indexed_slices():
      pred = constant_op.constant(True)
      def true_fn():
        return math_ops._as_indexed_slices(constant_op.constant([1.])), None
      def false_fn():
        return math_ops._as_indexed_slices(constant_op.constant([2.])), None
      result = cond_v2.cond_v2(pred, true_fn, false_fn)
      self.assertIsNone(result[1])
      return ops.convert_to_tensor(result[0])
    output = build_cond_with_indexed_slices()
    self.assertAllEqual(output, [1.])

  def testReturnsNonesAndIndexedSlices(self):

    @def_function.function
    def build_cond_with_indexed_slices():
      pred = constant_op.constant(True)

      def true_fn():
        return (None, None, None,
                math_ops._as_indexed_slices(constant_op.constant([1.])))

      def false_fn():
        return (None, None, None,
                math_ops._as_indexed_slices(constant_op.constant([2.])))

      result = cond_v2.cond_v2(pred, true_fn, false_fn)
      self.assertIsNone(result[0])
      self.assertIsNone(result[1])
      self.assertIsNone(result[2])
      return ops.convert_to_tensor(result[3])

    output = build_cond_with_indexed_slices()
    self.assertAllEqual(output, [1.])

  def testCondNestedFunctionGradientWithSavedModel(self):
    class Model(module_lib.Module):

      def __init__(self):
        self.v = resource_variable_ops.ResourceVariable([[1., 1.], [1., 1.]])

      @def_function.function
      def call(self, x, cond):

        @def_function.function
        def true_fn():
          # Einsum doesn't have a symbolic gradient op registered.
          # Taking gradient of an einsum op will fail if its python gradient
          # function is not found after loaded from a SavedModel.
          return gen_linalg_ops.einsum([x, self.v], "ab,bc->ac")

        @def_function.function
        def false_fn():
          return x

        return cond_v2.cond_v2(cond > 0, true_fn, false_fn)

    model = Model()
    x = constant_op.constant([[1., 1.], [1., 1.]])
    cond = constant_op.constant(1.)
    with backprop.GradientTape() as tape:
      y = tape.gradient(model.call(x, cond), model.v)

    self.assertAllEqual(y, [[2., 2.], [2., 2.]])

    saved_model_dir = os.path.join(self.create_tempdir(), "saved_model")
    save_lib.save(model, saved_model_dir)
    loaded_model = load_lib.load(saved_model_dir)
    with backprop.GradientTape() as tape:
      y = tape.gradient(loaded_model.call(x, cond), loaded_model.v)

    self.assertAllEqual(y, [[2., 2.], [2., 2.]])

  def testCondNestedFunctionGradientWithXlaDynamicCondition(self):

    v = resource_variable_ops.ResourceVariable([[1., 1.], [1., 1.]])

    @def_function.function(
        jit_compile=True,
        input_signature=[
            tensor_spec.TensorSpec([None, 2]),
            tensor_spec.TensorSpec([]),
        ],
    )
    def f(x, cond):

      @def_function.function
      def true_fn():
        return gen_linalg_ops.einsum([x, v], "ab,bc->ac")

      @def_function.function
      def false_fn():
        return x

      return cond_v2.cond_v2(cond > 0, true_fn, false_fn)

    x = constant_op.constant([[1., 1.], [1., 1.]])
    cond = constant_op.constant(1.)
    with backprop.GradientTape() as tape:
      # Shape of x in HLO graph should be [<=2, 2].
      y = tape.gradient(f(x, cond), v)

    self.assertAllEqual(y, [[2., 2.], [2., 2.]])

    x = constant_op.constant([[1., 1.], [1., 1.], [1., 1.]])
    with backprop.GradientTape() as tape:
      # HLO graph should be re-compiled to handle x with shape [<=3, 2].
      y = tape.gradient(f(x, cond), v)

    self.assertAllEqual(y, [[3., 3.], [3., 3.]])

  def testExternalControlDependencies(self):
    with ops.Graph().as_default(), self.test_session():
      v = variables.Variable(1.0)
      self.evaluate(v.initializer)
      op = v.assign_add(1.0)

      def true_branch():
        with ops.control_dependencies([op]):
          return 1.0

      cond_v2.cond_v2(array_ops.placeholder_with_default(False, None),
                      true_branch,
                      lambda: 2.0).eval()
      self.assertAllEqual(self.evaluate(v), 2.0)

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

  def _createCond(self, name, use_fast_cond=False):
    """Creates a cond_v2 call and returns the output tensor and the cond op."""
    pred = constant_op.constant(True, name="pred")
    x = constant_op.constant(1.0, name="x")

    def true_fn():
      return x

    def false_fn():
      return x + 1

    if use_fast_cond:
      output = cond_v2.fast_cond_v2(pred, true_fn, false_fn, name=name)
      cond_op = output.op
    else:
      output = cond_v2.cond_v2(pred, true_fn, false_fn, name=name)
      cond_op = output.op.inputs[0].op
    self.assertEqual(cond_op.type, "StatelessIf")
    return output, cond_op

  def _createNestedCond(self, name):
    """Like _createCond but creates a nested cond_v2 call as well."""
    pred = constant_op.constant(True, name="pred")
    x = constant_op.constant(1.0, name="x")

    def true_fn():
      return cond_v2.cond_v2(pred, lambda: x, lambda: x + 1)

    def false_fn():
      return x + 2

    output = cond_v2.cond_v2(pred, true_fn, false_fn, name=name)
    cond_op = output.op.inputs[0].op
    self.assertEqual(cond_op.type, "StatelessIf")
    return output, cond_op

  def testDefaultName(self):
    with ops.Graph().as_default():
      _, cond_op = self._createCond(None)
      self.assertEqual(cond_op.name, "cond")
      self.assertRegex(cond_op.get_attr("then_branch").name, r"cond_true_\d*")
      self.assertRegex(cond_op.get_attr("else_branch").name, r"cond_false_\d*")

    with ops.Graph().as_default():
      with ops.name_scope("foo"):
        _, cond1_op = self._createCond("")
        self.assertEqual(cond1_op.name, "foo/cond")
        self.assertRegex(
            cond1_op.get_attr("then_branch").name, r"foo_cond_true_\d*")
        self.assertRegex(
            cond1_op.get_attr("else_branch").name, r"foo_cond_false_\d*")

        _, cond2_op = self._createCond(None)
        self.assertEqual(cond2_op.name, "foo/cond_1")
        self.assertRegex(
            cond2_op.get_attr("then_branch").name, r"foo_cond_1_true_\d*")
        self.assertRegex(
            cond2_op.get_attr("else_branch").name, r"foo_cond_1_false_\d*")

  @test_util.run_v2_only
  def testInheritParentNameScope(self):

    @def_function.function
    def f():
      with ops.name_scope("foo"):

        def then_branch():
          with ops.name_scope("then"):
            actual_name_scope = ops.get_name_scope()
            expected_name_scope = "foo/cond/then"
            self.assertEqual(actual_name_scope, expected_name_scope)
          return 0.

        def else_branch():
          with ops.name_scope("else"):
            actual_name_scope = ops.get_name_scope()
            expected_name_scope = "foo/cond/else"
            self.assertEqual(actual_name_scope, expected_name_scope)
          return 0.

        return cond_v2.cond_v2(
            constant_op.constant(True), then_branch, else_branch)

    f()

  @test_util.run_v1_only("b/120545219")
  def testFunctionInCond(self):
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(2.0, name="y")

      def true_fn():

        @def_function.function
        def fn():
          return x * y * 2.0

        return fn()

      def false_fn():
        return 2.0

      self._testCond(true_fn, false_fn, [x])
      self._testCond(true_fn, false_fn, [x, y])
      self._testCond(true_fn, false_fn, [y])

  @test_util.run_deprecated_v1
  def testNestedFunctionInCond(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():
      return 2.0

    def false_fn():

      @def_function.function
      def fn():

        @def_function.function
        def nested_fn():
          return x * y * 2.0

        return nested_fn()

      return fn()

    self._testCond(true_fn, false_fn, [x])
    self._testCond(true_fn, false_fn, [x, y])
    self._testCond(true_fn, false_fn, [y])

  @test_util.run_deprecated_v1
  def testDoubleNestedFunctionInCond(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(2.0, name="y")

    def true_fn():

      @def_function.function
      def fn():

        @def_function.function
        def nested_fn():

          @def_function.function
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

  def testGradientFromInsideFunction(self):

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

      # Compute grads inside a tf function.
      @def_function.function
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

  def testGradientFromInsideNestedFunction(self):

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

      # Compute grads inside a tf function.
      @def_function.function
      def nesting_fn():

        @def_function.function
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

  def testBuildCondAndGradientInsideFunction(self):

    def build_graph():
      pred_outer = array_ops.placeholder(dtypes.bool, name="pred_outer")
      pred_inner = array_ops.placeholder(dtypes.bool, name="pred_inner")
      x = constant_op.constant(1.0, name="x")
      y = constant_op.constant(2.0, name="y")

      # Build cond and its gradient inside a tf function.
      @def_function.function
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

  @test_util.run_deprecated_v1
  def testFuncCond(self):

    @def_function.function
    def fn_with_cond():
      cond_v2.cond_v2(
          constant_op.constant(True),
          lambda: array_ops.zeros([]),
          lambda: array_ops.ones([]),
          name="cond_1")
      return cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: array_ops.zeros([]),
          lambda: array_ops.ones([]),
          name="cond_2")

    concrete_fn = fn_with_cond.get_concrete_function()
    cond_1 = concrete_fn.graph.get_operation_by_name("cond_1")
    cond_2 = concrete_fn.graph.get_operation_by_name("cond_2")
    # Verify that all functional ops are stateless and cond_2 does not have
    # any control inputs.
    self.assertEqual(cond_1.type, "StatelessIf")
    self.assertEqual(cond_2.type, "StatelessIf")
    self.assertLen(cond_2.control_inputs, 0)
    fn_output = concrete_fn()
    self.assertEqual(fn_output.op.type, "PartitionedCall")
    self.assertAllEqual(fn_output, 1.0)

  @test_util.run_deprecated_v1
  def testFuncCondFunc(self):

    @def_function.function
    def fn_with_cond():
      cond_v2.cond_v2(
          constant_op.constant(True),
          lambda: constant_op.constant(1.),
          lambda: constant_op.constant(2.),
          name="cond_1")

      @def_function.function
      def true_branch():
        return constant_op.constant(3.)

      return cond_v2.cond_v2(
          constant_op.constant(True),
          true_branch,
          lambda: constant_op.constant(4.),
          name="cond_2")

    concrete_fn = fn_with_cond.get_concrete_function()
    cond_1 = concrete_fn.graph.get_operation_by_name("cond_1")
    cond_2 = concrete_fn.graph.get_operation_by_name("cond_2")
    # Verify that all functional ops are stateless and cond_2 does not have
    # any control inputs.
    self.assertEqual(cond_1.type, "StatelessIf")
    self.assertEqual(cond_2.type, "StatelessIf")
    self.assertLen(cond_2.control_inputs, 0)
    cond_2_true_graph, _ = cond_v2.get_func_graphs(cond_2)
    cond_2_true_graph_operations = cond_2_true_graph.get_operations()
    self.assertEmpty([
        op for op in cond_2_true_graph_operations
        if op.type == "StatefulPartitionedCall"
    ])
    self.assertLen([
        op for op in cond_2_true_graph_operations
        if op.type == "PartitionedCall"
    ], 1)
    fn_output = concrete_fn()
    self.assertEqual(fn_output.op.type, "PartitionedCall")
    self.assertAllEqual(fn_output, 3.0)

  @test_util.run_deprecated_v1
  def testFuncCondWithVariable(self):
    v1 = variables.Variable(2.)
    v2 = variables.Variable(4.)

    self.evaluate(variables.global_variables_initializer())

    def update_v1():
      v1.assign(v1)
      return v1

    def update_v2():
      v2.assign(v2)
      return v2

    @def_function.function
    def fn_with_cond():
      cond_v2.cond_v2(
          constant_op.constant(True),
          update_v1,
          lambda: constant_op.constant(0.),
          name="cond_1")
      cond_2 = cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: constant_op.constant(0.),
          update_v1,
          name="cond_2")
      cond_v2.cond_v2(
          constant_op.constant(True),
          update_v2,
          lambda: constant_op.constant(0.),
          name="cond_3")
      cond_4 = cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: constant_op.constant(0.),
          lambda: v2,
          name="cond_4")
      stateless_cond = cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: constant_op.constant(5.),
          lambda: constant_op.constant(6.),
          name="stateless_cond")
      return cond_2, cond_4, stateless_cond

    concrete_fn = fn_with_cond.get_concrete_function()
    cond_1 = concrete_fn.graph.get_operation_by_name("cond_1")
    cond_2 = concrete_fn.graph.get_operation_by_name("cond_2")
    cond_3 = concrete_fn.graph.get_operation_by_name("cond_3")
    cond_4 = concrete_fn.graph.get_operation_by_name("cond_4")
    stateless_cond = concrete_fn.graph.get_operation_by_name("stateless_cond")
    self.assertEqual(cond_1.type, "If")
    self.assertEqual(cond_2.type, "If")
    self.assertEqual(cond_3.type, "If")
    self.assertEqual(cond_4.type, "If")
    self.assertEqual(stateless_cond.type, "StatelessIf")
    self.assertEmpty(cond_1.control_inputs)
    self.assertLen(cond_2.control_inputs, 1)
    self.assertIs(cond_2.control_inputs[0], cond_1)
    self.assertEmpty(cond_3.control_inputs)
    self.assertLen(cond_4.control_inputs, 1)
    self.assertIs(cond_4.control_inputs[0], cond_3)
    # Does not touch any variable so should not have any control inputs.
    self.assertEmpty(stateless_cond.control_inputs)
    fn_output = concrete_fn()
    self.assertEqual(fn_output[0].op.type, "StatefulPartitionedCall")
    self.assertAllEqual(self.evaluate(fn_output), [2.0, 4.0, 6.0])

  @test_util.run_deprecated_v1
  def testFuncCondFuncWithVariable(self):
    v1 = variables.Variable(2.)
    v2 = variables.Variable(4.)

    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def fn_with_cond():

      def update_v1():
        v1.assign(v1)
        return v1

      def update_v2():
        v2.assign(v2)
        return v2

      cond_v2.cond_v2(
          constant_op.constant(True),
          update_v1,
          lambda: constant_op.constant(0.),
          name="cond_1")
      cond_2 = cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: constant_op.constant(0.),
          update_v1,
          name="cond_2")
      cond_v2.cond_v2(
          constant_op.constant(True),
          update_v2,
          lambda: constant_op.constant(0.),
          name="cond_3")

      @def_function.function
      def cond_4_false_branch():
        v2.assign(v2)
        return v2

      cond_4 = cond_v2.cond_v2(
          constant_op.constant(False),
          lambda: constant_op.constant(0.),
          cond_4_false_branch,
          name="cond_4")
      return cond_2, cond_4

    concrete_fn = fn_with_cond.get_concrete_function()
    cond_1 = concrete_fn.graph.get_operation_by_name("cond_1")
    cond_2 = concrete_fn.graph.get_operation_by_name("cond_2")
    cond_3 = concrete_fn.graph.get_operation_by_name("cond_3")
    cond_4 = concrete_fn.graph.get_operation_by_name("cond_4")
    self.assertEqual(cond_1.type, "If")
    self.assertEqual(cond_2.type, "If")
    self.assertEqual(cond_3.type, "If")
    self.assertEqual(cond_4.type, "If")
    self.assertEmpty(cond_1.control_inputs)
    self.assertLen(cond_2.control_inputs, 1)
    self.assertIs(cond_2.control_inputs[0], cond_1)
    self.assertEmpty(cond_3.control_inputs)
    self.assertLen(cond_4.control_inputs, 1)
    self.assertIs(cond_4.control_inputs[0], cond_3)
    _, cond_4_false_graph = cond_v2.get_func_graphs(cond_4)
    cond_4_false_graph_operations = cond_4_false_graph.get_operations()
    self.assertEmpty([
        op for op in cond_4_false_graph_operations
        if op.type == "PartitionedCall"
    ])
    self.assertLen([
        op for op in cond_4_false_graph_operations
        if op.type == "StatefulPartitionedCall"
    ], 1)
    fn_output = concrete_fn()
    self.assertEqual(fn_output[0].op.type, "StatefulPartitionedCall")
    self.assertAllEqual(self.evaluate(fn_output), [2.0, 4.0])

  def testGradientTapeOfCondWithResourceVariableInFunction(self):
    v = variables.Variable(2.)

    @def_function.function
    def fn_with_cond():
      with backprop.GradientTape() as tape:
        pred = constant_op.constant(True, dtype=dtypes.bool)

        def true_fn():
          return math_ops.pow(v, 3)

        def false_fn():
          return v

        cond = cond_v2.cond_v2(pred, true_fn, false_fn, name="cond")
      return tape.gradient(cond, v)

    self.assertAllEqual(fn_with_cond(), 12.0)

  def _CheckIteratedCosGradients(self, func):

    def _grad(f):
      def _grad_function(primal):
        with backprop.GradientTape() as tape:
          tape.watch(primal)
          primal_out = f(primal)
        return tape.gradient(primal_out, primal)
      return _grad_function

    f = func
    one = constant_op.constant(1.)
    for expected in [math_ops.cos,
                     lambda x: -math_ops.sin(x),
                     lambda x: -math_ops.cos(x),
                     math_ops.sin,
                     math_ops.cos]:
      self.assertAllClose(expected(one), def_function.function(f)(one))
      f = _grad(f)

  def testIteratedGradientsCond(self):
    def _func(x):
      return cond_v2.cond_v2(
          constant_op.constant(True),
          lambda: math_ops.cos(array_ops.identity(x)),
          lambda: math_ops.sin(array_ops.identity(x)))
    self._CheckIteratedCosGradients(_func)

  def testIteratedGradientsCase(self):
    def _func(x):
      return cond_v2.indexed_case(
          constant_op.constant(1),
          [lambda: math_ops.sin(array_ops.identity(x)),
           lambda: math_ops.cos(array_ops.identity(x))])
    self._CheckIteratedCosGradients(_func)

  def testLowering(self):
    with ops.Graph().as_default() as g:
      # Disable Loop_optimizer grappler pass for this test because it replaces
      # Switch with Identity when it's part of a dead branch.
      config = config_pb2.ConfigProto()
      config.graph_options.rewrite_options.loop_optimization = (
          rewriter_config_pb2.RewriterConfig.OFF)
      with self.session(graph=g, config=config) as sess:
        cond_output, _ = self._createCond("cond")

        run_options = config_pb2.RunOptions(output_partition_graphs=True)
        run_metadata = config_pb2.RunMetadata()
        sess.run(cond_output, options=run_options, run_metadata=run_metadata)

        # If lowering was enabled, there should be a `Switch` node
        self.assertTrue(
            _has_node_with_op(run_metadata, "Switch"),
            "A `Switch` op should exist if the graph was lowered.")

        # If lowering was enabled, there should be no `If` node
        self.assertFalse(
            _has_node_with_op(run_metadata, "StatelessIf"),
            "An `If` op was found, but it should be lowered.")

  @test_util.run_deprecated_v1
  def testLoweringDisabledInXLA(self):
    with self.session(graph=ops.Graph()) as sess:
      # Build the cond_v2 in an XLA context
      xla_context = control_flow_ops.XLAControlFlowContext()
      xla_context.Enter()
      cond_output, cond_op = self._createCond("cond")
      xla_context.Exit()

      # Check lowering attr is not set.
      with self.assertRaises(ValueError):
        cond_op.get_attr("_lower_using_switch_merge")

      # Check the actual graph that is run.
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      run_metadata = config_pb2.RunMetadata()
      sess.run(cond_output, options=run_options, run_metadata=run_metadata)

      # Lowering disabled in XLA, there should be no `Switch` node
      self.assertFalse(
          _has_node_with_op(run_metadata, "Switch"),
          "A `Switch` op exists, but the graph should not be lowered.")

      if test_util.is_xla_enabled():
        # If XLA is actually enabled then we expect the StatelessIf to have been
        # put inside an XLA cluster.
        self.assertFalse(
            _has_node_with_op(run_metadata, "StatelessIf"),
            ("A `StatelessIf` op was found, but the node should have been " +
             "clustered."))
        self.assertTrue(
            _has_node_with_op(run_metadata, "_XlaCompile"),
            ("An `_XlaCompile` op was not found, but the `StatelessIf` (at " +
             "least) op should have been clustered."))
        self.assertTrue(
            _has_node_with_op(run_metadata, "_XlaRun"),
            ("An `_XlaRun` op was not found, but the `StatelessIf` (at " +
             "least) op should have been clustered."))
      else:
        # Lowering disabled in XLA, there should still be an `If` node
        self.assertTrue(
            _has_node_with_op(run_metadata, "StatelessIf"),
            ("A `StatelessIf` op was not found, but the graph should not be " +
             "lowered."))

  @test_util.run_deprecated_v1
  def testNestedLoweringDisabledInXLA(self):
    # Build the cond_v2 in an XLA context
    xla_context = control_flow_ops.XLAControlFlowContext()
    xla_context.Enter()
    _, cond_op = self._createNestedCond("cond")
    xla_context.Exit()

    # Check lowering attr is not set for either If node.
    with self.assertRaises(ValueError):
      cond_op.get_attr("_lower_using_switch_merge")

    nested_if_ops = []
    for func in ops.get_default_graph()._functions.values():
      nested_if_ops.extend(
          op for op in func.graph.get_operations() if op.type == "StatelessIf")
    self.assertEqual(len(nested_if_ops), 1)
    with self.assertRaises(ValueError):
      nested_if_ops[0].get_attr("_lower_using_switch_merge")

    # TODO(skyewm): check the actual graphs that are run once we have a way to
    # programmatically access those graphs.

  @test_util.run_deprecated_v1
  def testLoweringDisabledWithFastCond(self):
    with self.session(graph=ops.Graph()) as sess:
      # Build the cond_v2 in an XLA context
      cond_output, cond_op = self._createCond("cond", use_fast_cond=True)

      # Check lowering attr is not set.
      lowering_attr = cond_op.get_attr("_lower_using_switch_merge")
      self.assertFalse(
          lowering_attr, f"Lowering attr is not set as expected. {cond_op}"
      )

      # Check the actual graph that is run.
      run_options = config_pb2.RunOptions(output_partition_graphs=True)
      run_metadata = config_pb2.RunMetadata()
      sess.run(cond_output, options=run_options, run_metadata=run_metadata)

      # Lowering disabled for `fast_cond_v2`, there should be no `Switch` node
      self.assertFalse(
          _has_node_with_op(run_metadata, "Switch"),
          "A `Switch` op exists, but the graph should not be lowered.",
      )

      if test_util.is_xla_enabled():
        # If XLA is actually enabled then we expect the StatelessIf to have been
        # put inside an XLA cluster.
        self.assertFalse(
            _has_node_with_op(run_metadata, "StatelessIf"),
            (
                "A `StatelessIf` op was found, but the node should have been "
                + "clustered."
            ),
        )
        self.assertTrue(
            _has_node_with_op(run_metadata, "_XlaCompile"),
            (
                "An `_XlaCompile` op was not found, but the `StatelessIf` (at "
                + "least) op should have been clustered."
            ),
        )
        self.assertTrue(
            _has_node_with_op(run_metadata, "_XlaRun"),
            (
                "An `_XlaRun` op was not found, but the `StatelessIf` (at "
                + "least) op should have been clustered."
            ),
        )
      else:
        # Lowering disabled for `fast_cond_v2`, there should still be a
        # `StatelessIf` node.
        self.assertTrue(
            _has_node_with_op(run_metadata, "StatelessIf"),
            (
                "A `StatelessIf` op was not found, but the graph should not be "
                + "lowered."
                + str(run_metadata)
            ),
        )

  # b/131355614
  @test_util.run_deprecated_v1
  def testNoOptionalsInXla(self):

    @def_function.function
    def func_with_cond():
      pred = constant_op.constant(True, name="pred")
      x = constant_op.constant(1.0, name="x")

      def true_fn():
        intermediate = x + 1
        return intermediate * x

      def false_fn():
        return x + 1

      output = cond_v2.cond_v2(pred, true_fn, false_fn)
      grad = gradients_impl.gradients(output, x)[0]

      forward_if_op = output.op.inputs[0].op
      gradient_if_op = grad.op.inputs[0].op

      def verify_no_optional_ops(op, branch_name):
        branch_function = ops.get_default_graph()._get_function(
            op.get_attr(branch_name).name)
        function_def = branch_function.cached_definition
        for node_def in function_def.node_def:
          self.assertNotIn(node_def.op, _OPTIONAL_OPS)

      verify_no_optional_ops(forward_if_op, "then_branch")
      verify_no_optional_ops(forward_if_op, "else_branch")
      verify_no_optional_ops(gradient_if_op, "then_branch")
      verify_no_optional_ops(gradient_if_op, "else_branch")

      return grad

    xla_context = control_flow_ops.XLAControlFlowContext()
    xla_context.Enter()
    func_with_cond()
    xla_context.Exit()

  @test_util.run_deprecated_v1
  def testLoweringDisabledWithSingleThreadedExecutorContext(self):
    # Single threaded executor does not support partitioned graphs, so we can't
    # run on GPUs (running on GPU requires a mixed CPU/GPU graph).
    with self.session(graph=ops.Graph(), use_gpu=False) as sess:

      @def_function.function
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

    output = tf_cond.cond(
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

    with self.assertRaisesRegex(
        TypeError, "true_fn and false_fn arguments to tf.cond must have the "
        "same number, type, and overall structure of return values."):
      tf_cond.cond(constant_op.constant(False), true_fn, false_fn)

  @test_util.enable_control_flow_v2
  def testCondAndTensorArray(self):
    x = math_ops.range(-5, 5)
    output = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=x.shape[0])

    def loop_body(i, output):

      def if_true():
        return output.write(i, x[i]**2)

      def if_false():
        return output.write(i, x[i])

      output = tf_cond.cond(x[i] > 0, if_true, if_false)
      return i + 1, output

    _, output = while_loop.while_loop(
        lambda i, arr: i < x.shape[0],
        loop_body,
        loop_vars=(constant_op.constant(0), output))
    output_t = output.stack()
    self.assertAllEqual(
        self.evaluate(output_t), [-5, -4, -3, -2, -1, 0, 1, 4, 9, 16])

  @test_util.enable_control_flow_v2
  def testCondAndTensorArrayInFunction(self):

    @def_function.function
    def f():
      x = math_ops.range(-5, 5)
      output = tensor_array_ops.TensorArray(dtype=dtypes.int32, size=x.shape[0])

      def loop_body(i, output):

        def if_true():
          return output.write(i, x[i]**2)

        def if_false():
          return output.write(i, x[i])

        output = tf_cond.cond(x[i] > 0, if_true, if_false)
        return i + 1, output

      _, output = while_loop.while_loop(
          lambda i, arr: i < x.shape[0],
          loop_body,
          loop_vars=(constant_op.constant(0), output))
      return output.stack()

    output_t = f()
    self.assertAllEqual(output_t, [-5, -4, -3, -2, -1, 0, 1, 4, 9, 16])

  @test_util.run_deprecated_v1
  def testForwardPassRewrite(self):
    x = constant_op.constant(1.0, name="x")
    y = constant_op.constant(1.0, name="y")

    def true_fn():
      y_plus_one = y + 1.
      return x * y_plus_one

    output = cond_v2.cond_v2(constant_op.constant(True), true_fn, lambda: x)
    if_op = output.op.inputs[0].op
    self.assertEqual(if_op.type, "StatelessIf")
    # pylint: disable=g-deprecated-assert
    self.assertEqual(len(if_op.outputs), 1)

    gradients_impl.gradients(output, x)
    # if_op should have been rewritten to output `y_plus_one`.
    self.assertEqual(len(if_op.outputs), 2)

    gradients_impl.gradients(output, x)
    # Computing the gradient again shouldn't rewrite if_op again.
    self.assertEqual(len(if_op.outputs), 2)
    # pylint: enable=g-deprecated-assert

  @test_util.run_deprecated_v1
  def testDoNotAccumulateConstants(self):
    x = constant_op.constant(1.0, name="x")
    output = cond_v2.cond_v2(
        constant_op.constant(True), lambda: x * 2.0, lambda: x)
    if_op = output.op.inputs[0].op
    self.assertEqual(if_op.type, "StatelessIf")
    # pylint: disable=g-deprecated-assert
    self.assertEqual(len(if_op.outputs), 1)

    gradients_impl.gradients(output, x)
    # Number of outputs does change because
    # 1. `x` is a loop input so does not need to be accumulated.
    # 2. 2.0 is a constant so it is not accumulated.
    self.assertEqual(len(if_op.outputs), 1)

    gradients_impl.gradients(output, x)
    # Computing the gradient again shouldn't rewrite if_op again.
    self.assertEqual(len(if_op.outputs), 1)
    # pylint: enable=g-deprecated-assert

  def testIsControlFlowGraph(self):
    x = constant_op.constant(1.0, name="x")

    @def_function.function
    def f(c):

      def then_branch():
        i = x + 1
        self.assertTrue(i.graph.is_control_flow_graph)
        return i

      def else_branch():
        i = x + 1
        self.assertTrue(i.graph.is_control_flow_graph)
        return i

      return cond_v2.cond_v2(c, then_branch, else_branch)

    i = f(constant_op.constant(True))
    self.assertEqual(self.evaluate(i), 2.0)

    i = f(constant_op.constant(False))
    self.assertEqual(self.evaluate(i), 2.0)

  def testGradientOfMixedOptionals(self):

    @def_function.function
    def f(c):
      x = constant_op.constant(1., name="x")

      def then_branch():
        return x**2.0, gen_optional_ops.optional_from_value(
            [constant_op.constant(1)]
        )

      def else_branch():
        return x**3.0, gen_optional_ops.optional_from_value(
            [constant_op.constant(1.0)]
        )

      y, _ = cond_v2.cond_v2(c, then_branch, else_branch)
      return gradients_impl.gradients(y, x)
    self.assertAllClose([2.], f(constant_op.constant(True)))


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
        self.assertEqual(self.evaluate(cnd), 7)

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
        self.assertEqual(self.evaluate(cnd), 7)

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
        self.assertEqual(self.evaluate(cnd), 14)

        read_z_collection = ops.get_collection("z")
        self.assertEqual(read_z_collection, [7])


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
          self.assertEqual(self.evaluate(cnd_true), 2)

          cnd_false = cond_v2.cond_v2(
              constant_op.constant(False), true_fn, false_fn)
          self.assertEqual(self.evaluate(cnd_false), 6)

          v4 = variables.Variable([3])
          q4 = data_flow_ops.FIFOQueue(1, dtypes.float32)
        v5 = variables.Variable([4])
        q5 = data_flow_ops.FIFOQueue(1, dtypes.float32)

      self.assertEqual(compat.as_bytes("l1"), container(v4))
      self.assertEqual(compat.as_bytes("l1"), container(q4.queue_ref))
      self.assertEqual(compat.as_bytes(""), container(v5))
      self.assertEqual(compat.as_bytes(""), container(q5.queue_ref))


@test_util.disable_tfrt("b/171412104: This test requires distributed support.")
class CondV2ColocationGroupAndDeviceTest(test.TestCase, parameterized.TestCase):

  def setUp(self):
    context._reset_context()
    super(CondV2ColocationGroupAndDeviceTest, self).setUp()
    cpus = context.context().list_physical_devices("CPU")
    context.context().set_logical_device_configuration(
        cpus[0], [
            context.LogicalDeviceConfiguration(),
            context.LogicalDeviceConfiguration()
        ])
    workers, _ = test_util.create_local_cluster(num_workers=1, num_ps=0)
    remote.connect_to_remote_host(workers[0].target)

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
          self.assertEqual(
              cond_v2.cond_v2(constant_op.constant(True), fn, fn).eval(), 3)

        def fn2():
          c = constant_op.constant(3.0)
          self.assertEqual([b"loc:@a", b"loc:@b"], c.op.colocation_groups())
          return c

        with ops.colocate_with(a.op):
          with ops.colocate_with(b.op):
            self.assertEqual(
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
          self.assertEqual(
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

    def fn():
      cpu_zero_op = test_ops.device_placement_op()
      self.assertEqual("/job:localhost/device:CPU:0", cpu_zero_op.device)
      with ops.device("CPU:1"):
        cpu_one_op = test_ops.device_placement_op()
        self.assertEqual("/job:localhost/device:CPU:1", cpu_one_op.device)
      return cpu_zero_op, cpu_one_op

    @def_function.function
    def _cond_wrapper():
      with ops.device("/job:localhost/device:CPU:0"):
        return cond_v2.cond_v2(constant_op.constant(True), fn, fn)

    zero_expected, one_expected = self.evaluate(_cond_wrapper())
    self.assertIn(compat.as_bytes("CPU:0"), zero_expected)
    self.assertIn(compat.as_bytes("CPU:1"), one_expected)
    self.assertIn(compat.as_bytes("job:localhost"), zero_expected)
    self.assertIn(compat.as_bytes("job:localhost"), one_expected)

    def fn2():
      self.assertEqual("/job:localhost/device:GPU:0",
                       constant_op.constant(3.0).op.device)
      return test_ops.device_placement_op()

    @def_function.function
    def _cond_wrapper2():
      with ops.device("/job:localhost/device:GPU:0"):
        return cond_v2.cond_v2(constant_op.constant(True), fn2, fn2)

    if test_util.is_gpu_available():
      self.assertIn(compat.as_bytes("GPU:0"), self.evaluate(_cond_wrapper2()))
      self.assertIn(
          compat.as_bytes("job:localhost"), self.evaluate(_cond_wrapper2()))
    else:
      self.skipTest("Test requires a GPU to check GPU device placement.")

  @parameterized.named_parameters([
      dict(
          testcase_name="Function",
          functional_op_to_test=lambda fn: def_function.function(fn)()),
      dict(
          testcase_name="Cond",
          functional_op_to_test=
          lambda fn: cond_v2.cond_v2(constant_op.constant(True), fn, fn))
  ])
  def testDeviceBeforeRemote(self, functional_op_to_test):
    context.context().log_device_placement = True

    def _fn():
      local_op = test_ops.device_placement_op()
      with ops.device("/job:worker/CPU:0"):
        worker_op = test_ops.device_placement_op()
      return local_op, worker_op

    @def_function.function
    def _wrapper():
      with ops.device("/job:localhost"):
        return functional_op_to_test(_fn)

    local_expected, worker_expected = self.evaluate(_wrapper())
    self.assertIn(compat.as_bytes("job:localhost"), local_expected)
    self.assertIn(compat.as_bytes("job:worker"), worker_expected)

    del _fn, _wrapper

    # There's nothing special about localhost; if we swap roles (functional op
    # on worker, op on localhost) the inner placement still wins.
    def _fn2():
      local_op = test_ops.device_placement_op()
      with ops.device("/job:localhost/CPU:0"):
        worker_op = test_ops.device_placement_op()
      return local_op, worker_op

    @def_function.function
    def _wrapper2():
      with ops.device("/job:worker"):
        return functional_op_to_test(_fn2)

    worker_expected, local_expected = self.evaluate(_wrapper2())
    self.assertIn(compat.as_bytes("job:worker"), worker_expected)
    self.assertIn(compat.as_bytes("job:localhost"), local_expected)

  def testColocationBeforeCond(self):

    def _fn():
      result = test_ops.device_placement_op()
      self.assertIn("colocation_test_op",
                    result.op.colocation_groups()[0].decode())
      return result

    @def_function.function(autograph=False)
    def _cond_wrapper():
      with ops.device("/device:CPU:0"):
        op_on_cpu_0 = test_ops.device_placement_op(name="colocation_test_op")
      with ops.device("/device:CPU:1"):
        op_on_cpu_1 = test_ops.device_placement_op(name="colocation_test_op_1")
      condition = constant_op.constant(True)
      with ops.colocate_with(op_on_cpu_0.op):
        zero_expected = cond_v2.cond_v2(condition, _fn, _fn)
      with ops.colocate_with(op_on_cpu_1.op):
        one_expected = cond_v2.cond_v2(condition, _fn, _fn)
      return zero_expected, one_expected

    zero_expected, one_expected = self.evaluate(_cond_wrapper())
    self.assertIn(compat.as_bytes("CPU:0"), zero_expected)
    self.assertIn(compat.as_bytes("CPU:1"), one_expected)

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
          self.assertEqual(
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

        self.assertGreaterEqual(len(run_metadata.partition_graphs), 2)


class CaseTest(test.TestCase):

  def testCase(self):

    def branch1(x):
      logging_ops.print_v2("1")
      return x

    def branch2(x):
      return x + 1

    with ops.Graph().as_default():
      x = array_ops.constant(1)
      output = cond_v2.indexed_case(
          array_ops.constant(0), [lambda: branch1(x), lambda: branch2(x)])
      cond_op = output.op.inputs[0].op
      self.assertEqual(cond_op.type, "Case")
      self.assertEqual(1., self.evaluate(output))

  def testStatelessCase(self):

    def branch1(x):
      return x + 1

    def branch2(x):
      return x + 2

    with ops.Graph().as_default():
      x = array_ops.constant(1)
      output = cond_v2.indexed_case(
          array_ops.constant(0), [lambda: branch1(x), lambda: branch2(x)])
      cond_op = output.op.inputs[0].op
      self.assertEqual(cond_op.type, "StatelessCase")
      self.assertEqual(2., self.evaluate(output))


def _cond(pred, true_fn, false_fn, name):
  if _is_old_cond():
    return tf_cond.cond(pred, true_fn, false_fn, name=name)
  else:
    return cond_v2.cond_v2(pred, true_fn, false_fn, name=name)


def _is_old_cond():
  return isinstance(ops.get_default_graph()._get_control_flow_context(),
                    control_flow_ops.CondContext)


def _has_node_with_op(run_metadata, op_type):
  """Whether any node in `run_metadata.partition_graphs` matches `op_type`."""
  for graph in run_metadata.partition_graphs:
    for node in graph.node:
      if node.op == op_type:
        return True
  return False


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
