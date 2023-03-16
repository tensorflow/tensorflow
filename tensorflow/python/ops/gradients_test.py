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
"""Tests for tensorflow.ops.gradients."""

import sys
import warnings

from absl.testing import parameterized
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function as framework_function
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops  # pylint: disable=unused-import
from tensorflow.python.ops import functional_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_grad  # pylint: disable=unused-import
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import unconnected_gradients
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.nn_ops import bias_add
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


class GradientsTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testGradients(self):
    with ops.Graph().as_default():
      inp = constant(1.0, shape=[32, 100], name="in")
      w = constant(1.0, shape=[100, 10], name="w")
      b = constant(1.0, shape=[10], name="b")
      xw = math_ops.matmul(inp, w, name="xw")
      h = bias_add(xw, b, name="h")
      w_grad = gradients.gradients(h, w)[0]
    self.assertEqual("MatMul", w_grad.op.type)
    self.assertEqual(w_grad.op._original_op, xw.op)
    self.assertTrue(w_grad.op.get_attr("transpose_a"))
    self.assertFalse(w_grad.op.get_attr("transpose_b"))

  def testUnusedOutput(self):
    with ops.Graph().as_default():
      w = constant(1.0, shape=[2, 2])
      x = constant(1.0, shape=[2, 2])
      wx = math_ops.matmul(w, x)
      split_wx = array_ops.split(value=wx, num_or_size_splits=2, axis=0)
      c = math_ops.reduce_sum(split_wx[1])
      gw = gradients.gradients(c, [w])[0]
    self.assertEqual("MatMul", gw.op.type)

  def testColocateGradients(self):
    with ops.Graph().as_default() as g:
      w = constant(1.0, shape=[1, 1])
      x = constant(1.0, shape=[1, 2])
      with g.device("/device:GPU:0"):
        wx = math_ops.matmul(w, x)
      gw = gradients.gradients(wx, [w], colocate_gradients_with_ops=True)[0]
    self.assertEqual(gw.op.colocation_groups(), wx.op.colocation_groups())

  def testColocateGradientsWithAggregation(self):
    with ops.Graph().as_default() as g:
      with g.device("/device:GPU:1"):
        w = constant(1.0, shape=[1, 1])
      x = constant(1.0, shape=[1, 2])
      y = constant(1.0, shape=[1, 2])
      wx = math_ops.matmul(w, x)
      wy = math_ops.matmul(w, y)
      with g.device("/device:GPU:0"):
        z = wx + wy

      gw1 = gradients.gradients(z, [w], colocate_gradients_with_ops=True)[0]
      self.assertEqual(gw1.op.colocation_groups(), wx.op.colocation_groups())

      gw2 = gradients.gradients(z, [w], colocate_gradients_with_ops=False)[0]
      self.assertNotEqual(wx.op.colocation_groups(), gw2.op.colocation_groups())

  def testColocateGradientsWithAggregationInMultipleDevices(self):
    with ops.Graph().as_default() as g:
      with g.device("/device:GPU:1"):
        w = constant(1.0, shape=[1, 1])
      x = constant(1.0, shape=[1, 2])
      y = constant(1.0, shape=[1, 2])
      with g.device("/task:1"):
        wx = math_ops.matmul(w, x)
      with g.device("/task:2"):
        wy = math_ops.matmul(w, y)
      with g.device("/device:GPU:0"):
        z = wx + wy

      gw1 = gradients.gradients(z, [w], colocate_gradients_with_ops=True)[0]
      self.assertEqual(gw1.op.colocation_groups(), w.op.colocation_groups())

      gw2 = gradients.gradients(z, [w], colocate_gradients_with_ops=False)[0]
      self.assertNotEqual(w.op.colocation_groups(), gw2.op.colocation_groups())

  def testColocateGradientsWithGateGradients(self):
    if not test_util.is_gpu_available():
      self.skipTest("No GPU available")
    with ops.Graph().as_default() as g:
      with g.device("/device:CPU:0"):
        x = constant(1.0, shape=[1, 1])
        y = constant(1.0, shape=[1, 1])
        s = x + y
      with g.device("/device:GPU:0"):
        z = math_ops.reduce_sum(s)

      gz_x = gradients.gradients(z, [x], colocate_gradients_with_ops=True,
                                 gate_gradients=True)[0]

      # Make sure the placer doesn't complain.
      self.evaluate(gz_x)

  def testBoundaryStop(self):
    # Test that we don't differentiate 'x'. The gradient function for 'x' is
    # set explicitly to None so we will get an exception if the gradient code
    # tries to differentiate 'x'.
    with ops.Graph().as_default():
      c = constant(1.0)
      x = array_ops.identity(c)
      y = x + 1.0
      z = y + 1
      grads = gradients.gradients(z, [x])
      self.assertTrue(all(x is not None for x in grads))

  @test_util.run_v1_only("b/120545219")
  def testBoundaryContinue(self):
    # Test that we differentiate both 'x' and 'y' correctly when x is a
    # predecessor of y.
    with self.cached_session():
      x = constant(1.0)
      y = x * 2.0
      z = y * 3.0
      grads = gradients.gradients(z, [x, y])
      self.assertTrue(all(x is not None for x in grads))
      self.assertEqual(6.0, grads[0].eval())

  @test_util.run_v1_only("b/120545219")
  def testAggregationMethodAccumulateN(self):
    with self.cached_session():
      x = constant(1.0)
      y = x * 2.0
      z = y + y + y + y + y + y + y + y + y + y
      grads = gradients.gradients(
          z, [x, y],
          aggregation_method=gradients.AggregationMethod.
          EXPERIMENTAL_ACCUMULATE_N)
      self.assertTrue(all(x is not None for x in grads))
      self.assertEqual(20.0, grads[0].eval())
      self.assertEqual(10.0, grads[1].eval())

  @test_util.run_v1_only("b/120545219")
  def testAggregationMethodAddN(self):
    with self.cached_session():
      x = constant(1.0)
      y = x * 2.0
      z = y + y + y + y + y + y + y + y + y + y
      grads = gradients.gradients(
          z, [x, y], aggregation_method=gradients.AggregationMethod.ADD_N)
      self.assertTrue(all(x is not None for x in grads))
      self.assertEqual(20.0, grads[0].eval())
      self.assertEqual(10.0, grads[1].eval())

  @test_util.run_v1_only("b/120545219")
  def testAggregationMethodTree(self):
    with self.cached_session():
      x = constant(1.0)
      y = x * 2.0
      z = y + y + y + y + y + y + y + y + y + y
      grads = gradients.gradients(
          z, [x, y],
          aggregation_method=gradients.AggregationMethod.EXPERIMENTAL_TREE)
      self.assertTrue(all(x is not None for x in grads))
      self.assertEqual(20.0, grads[0].eval())
      self.assertEqual(10.0, grads[1].eval())

  def testNoGradientForStringOutputs(self):
    with ops.Graph().as_default():

      def _TestOpGrad(_, float_grad, string_grad):
        """Gradient function for TestStringOutput."""
        self.assertEqual(float_grad.dtype, dtypes.float32)
        self.assertFalse(string_grad)
        return float_grad

      ops.RegisterGradient("TestStringOutput")(_TestOpGrad)

      c = constant(1.0)
      x, _ = test_ops.test_string_output(c)
      z = x * 2.0
      w = z * 3.0
      grads = gradients.gradients(z, [c])
      self.assertIsInstance(grads[0], ops.Tensor)
      grads = gradients.gradients(w, [c])
      self.assertIsInstance(grads[0], ops.Tensor)

  def testNoGradientForStringOutputsWithOpNamespace(self):
    with ops.Graph().as_default():

      def _TestOpGrad(_, float_grad, string_grad):
        """Gradient function for TestStringOutput."""
        self.assertEqual(float_grad.dtype, dtypes.float32)
        self.assertFalse(string_grad)
        return float_grad

      ops.RegisterGradient("Namespace>TestStringOutput")(_TestOpGrad)

      c = constant(1.0)
      x, _ = test_ops.namespace_test_string_output(c)
      z = x * 2.0
      w = z * 3.0
      grads = gradients.gradients(z, [c])
      self.assertIsInstance(grads[0], ops.Tensor)
      grads = gradients.gradients(w, [c])
      self.assertIsInstance(grads[0], ops.Tensor)

  def testSingletonIndexedSlices(self):
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.identity(x)
      dy = indexed_slices.IndexedSlices(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.int32))
      dx, = gradients.gradients(y, x, grad_ys=dy)
      # The IndexedSlices gradient of tf.identity is the identity map.
      with self.cached_session() as sess:
        vdx, vdy = sess.run(
            [dx, dy], feed_dict={x: [1.0], dy.indices: [0], dy.values: [2.0]})
      self.assertEqual(vdx, vdy)

  @test_util.run_v1_only("b/120545219")
  def testNonDifferentiableSwitchInWhileLoop(self):
    with ops.Graph().as_default():
      v = array_ops.placeholder(dtypes.float32, [])

      def _Step(i, a, ta):
        a += math_ops.cast(v, dtypes.int32)
        return (i + 1, a, ta.write(i, a))

      n = 4
      i, _, ta = while_loop.while_loop(
          lambda i, *_: i < n, _Step,
          [0, 0, tensor_array_ops.TensorArray(dtypes.int32, size=n)])
      target = ta.read(i - 1)
      grad, = gradients.gradients(target, v)
      self.assertIsNone(grad)

  def testVariableReadValueGradient(self):
    with ops.Graph().as_default():
      init = constant_op.constant(100.0)
      var = variables.Variable(init)
      gradient = gradients.gradients(var.read_value(), var)
      self.assertIsNotNone(gradient)

  @parameterized.parameters(dtypes.float32, dtypes.float64)
  def testVariableDefaultGrad(self, dtype):
    with ops.Graph().as_default():
      init = constant_op.constant(100.0, dtype=dtype)
      var = variables.Variable(init)
      dummy_const = constant_op.constant(0.0)
      gradient = gradients.gradients(
          dummy_const,
          var,
          unconnected_gradients=unconnected_gradients.UnconnectedGradients.ZERO
      )[0]
      self.assertEqual(gradient.dtype, dtype)
      self.assertIsNotNone(gradient)

  def testVariableAsGraphElementGradient(self):
    with ops.Graph().as_default() as graph:
      init = constant_op.constant(100.0)
      var = variables.Variable(init)
      gradient = gradients.gradients(graph.as_graph_element(var), var)
      self.assertIsNotNone(gradient)

  @test_util.run_v1_only("b/120545219")
  def testVariableRefGradient(self):
    with ops.Graph().as_default():
      init = constant_op.constant(100.0)
      var = variables.VariableV1(init)
      gradient = gradients.gradients(var._ref(), var)
      self.assertIsNotNone(gradient)

  @test_util.run_v1_only("b/120545219")
  def testDependentYs(self):
    with self.cached_session():
      x = constant_op.constant(3.0)
      y = math_ops.square(x)
      y1 = math_ops.square(y)
      y2 = math_ops.square(y1)
      g = gradients.gradients([y, y2], x)
      self.assertAllClose(17502.0, g[0])
      g = gradients.gradients(y + y2, x)
      self.assertAllClose(17502.0, g[0])
      z = array_ops.identity(y)
      z2 = array_ops.identity(y2)
      g = gradients.gradients([z, z2], x)
      self.assertAllClose(17502.0, g[0])

  @test_util.run_v1_only("b/120545219")
  def testPartialDerivatives(self):
    with self.cached_session():
      x = constant_op.constant(1.)
      y = 2 * x
      z = x + y
      totalg = gradients.gradients(z, [x, y])
      self.assertEqual([3.0, 1.0], [g.eval() for g in totalg])
      partialg = gradients.gradients(z, [x, y], stop_gradients=[x, y])
      self.assertEqual([1.0, 1.0], [g.eval() for g in partialg])

  @test_util.run_v1_only("b/120545219")
  def testStopGradients(self):
    def _MakeGraph(rng, stop_gradients=()):
      def _FunctionOf(xs, k=3):
        return ops.convert_to_tensor(
            sum(math_ops.matmul(rng.rand(k, k), x) for x in xs)
            + rng.rand(k, k))

      a = _FunctionOf([])
      if "a" in stop_gradients: a = array_ops.stop_gradient(a)
      b = _FunctionOf([a])
      if "b" in stop_gradients: b = array_ops.stop_gradient(b)
      c = _FunctionOf([a, b])
      if "c" in stop_gradients: c = array_ops.stop_gradient(c)
      d = _FunctionOf([b, c])
      if "d" in stop_gradients: d = array_ops.stop_gradient(d)
      return dict(a=a, b=b, c=c, d=d)

    def _Gradients(ys, xs, **kwargs):
      dydxs = gradients.gradients(ys, xs, **kwargs)
      dydxs = [0. * x if dydx is None else dydx
               for x, dydx in zip(xs, dydxs)]
      return dydxs

    seed = np.random.randint(1000)
    cases = []
    subsets = [""] + "a b c d ab ac ad bc bd cd abc abd acd bcd abcd".split()
    graph = _MakeGraph(np.random.RandomState(seed))
    for constants in subsets:
      graph_with_stops = _MakeGraph(np.random.RandomState(seed), constants)
      for variables_ in subsets:
        # compute the gradient when stopped using tf.stop_gradients
        grad1 = _Gradients([graph_with_stops["d"]],
                           [graph_with_stops[v] for v in variables_])
        # compute the gradient when stopped using the stop_gradients kwarg
        grad2 = _Gradients([graph["d"]],
                           [graph[v] for v in variables_],
                           stop_gradients=[graph[v] for v in constants])
        cases.append(dict(grad1=grad1, grad2=grad2,
                          constants=constants, variables=variables_))

    # evaluate all tensors in one call to session.run for speed
    with self.cached_session() as sess:
      results = sess.run([(case["grad1"], case["grad2"]) for case in cases])

    for (npgrad1, npgrad2), case in zip(results, cases):
      for a, b in zip(npgrad1, npgrad2):
        np.testing.assert_allclose(a, b)

  def testUnconnectedGradientsNoneUnconnectedGradients(self):
    with ops.Graph().as_default():
      x = constant(1.0, shape=[2, 2])
      y = constant(3.0, shape=[3, 1])
      grad = gradients.gradients(
          [y], [x], unconnected_gradients="none")
    self.assertIsNone(grad[0])

  def testUnconnectedGradientsZerosUnconnectedGradients(self):
    with ops.Graph().as_default():
      x = constant(1.0, shape=[2, 2])
      y = constant(3.0, shape=[3, 1])
      grads = gradients.gradients(
          [y], [x], unconnected_gradients="zero")

      self.assertAllEqual([[0.0, 0.0], [0.0, 0.0]], self.evaluate(grads)[0])

  def testUnconnectedGradientsZeroConnectedGradients(self):
    with ops.Graph().as_default():
      x = constant(1.0)
      y = x * 3.0
      grad = gradients.gradients(
          [y], [x], unconnected_gradients="zero")

      self.assertEqual(3.0, self.evaluate(grad)[0])

  def testUnknownUnconnectedGradientsValueGiven(self):
    with ops.Graph().as_default():
      x = constant(1.0)
      y = constant(1.0)
      with self.assertRaisesRegex(
          ValueError, "Unknown value for unconnected_gradients: 'nonsense'"):
        gradients.gradients([y], [x], unconnected_gradients="nonsense")

  @parameterized.parameters(unconnected_gradients.UnconnectedGradients.ZERO,
                            unconnected_gradients.UnconnectedGradients.NONE)
  def testUnconnectedOpWithMultipleOutputs(self, unconnected_gradients_val):
    with ops.Graph().as_default():
      #  a    b
      #  |    |
      # IdentityN
      #  |    |
      #  c    d
      #  |
      # Identity
      #  |
      #  e
      a = constant_op.constant(1.0)
      b = constant_op.constant(1.0)
      c, d = array_ops.identity_n([a, b])
      e = array_ops.identity(c)
      # The aggregated grads for the IdentityN node would look like
      # [Tensor, None]. We expect this None to be converted to zeros.
      output = gradients.gradients(
          e, d, unconnected_gradients=unconnected_gradients_val)
      if (unconnected_gradients_val ==
          unconnected_gradients.UnconnectedGradients.ZERO):
        self.assertIsNotNone(output[0])
      else:
        self.assertIsNone(output[0])

  @parameterized.parameters(unconnected_gradients.UnconnectedGradients.ZERO,
                            unconnected_gradients.UnconnectedGradients.NONE)
  def testUnconnectedOpWithMultipleOutputsStopGradient(
      self, unconnected_gradients_val):
    with ops.Graph().as_default():
      #  a    b
      #  |    |
      # IdentityN
      #  |    |
      #  c    d
      #  |    |
      #  SG   |
      #  |    |
      #   \  /
      #    +
      #    e
      a = constant_op.constant(1.0)
      b = constant_op.constant(1.0)
      c, d = array_ops.identity_n([a, b])
      e = array_ops.stop_gradient(c) + d
      # The aggregated grads for the IdentityN node would look like
      # [None, Tensor]. We expect this None to be converted to zeros.
      output = gradients.gradients(
          e, c, unconnected_gradients=unconnected_gradients_val)
      if (unconnected_gradients_val ==
          unconnected_gradients.UnconnectedGradients.ZERO):
        self.assertIsNotNone(output[0])
      else:
        self.assertIsNone(output[0])


class FunctionGradientsTest(test_util.TensorFlowTestCase):

  @classmethod
  def XSquarePlusB(cls, x, b):
    return x * x + b

  @classmethod
  def XSquarePlusBGradient(cls, x, b, g):
    # Perturb gradients (multiply by 2), so we can test that this was called.
    g *= 2.0
    return g * 2.0 * x, g

  @classmethod
  def _PythonGradient(cls, op, grad):
    # Perturb gradients (multiply by 3), so we can test that this was called.
    grad *= 3.0
    return grad * op.inputs[0] * 2.0, grad

  @classmethod
  def _GetFunc(cls, **kwargs):
    return framework_function.Defun(dtypes.float32, dtypes.float32, **
                                    kwargs)(cls.XSquarePlusB)

  def _GetFuncGradients(self, f, x_value, b_value):
    x = constant_op.constant(x_value, name="x")
    b = constant_op.constant(b_value, name="b")

    y = f(x, b)
    grads = gradients.gradients(y, [x, b])

    return self.evaluate(grads)

  def testFunctionGradientsBasic(self):
    g = ops.Graph()
    with g.as_default():
      f = self._GetFunc()
      # Get gradients (should add SymbolicGradient node for function).
      grads = self._GetFuncGradients(f, [2.0], [1.0])
      self.assertAllEqual([4.0], grads[0])
      self.assertAllEqual([1.0], grads[1])

  def testFunctionGradientsComposition(self):
    with ops.Graph().as_default():
      f = self._GetFunc()
      x = constant_op.constant([2.0], name="x")
      b1 = constant_op.constant([1.0], name="b1")
      b2 = constant_op.constant([1.0], name="b2")

      y = f(f(x, b1), b2)
      # Build gradient graph (should add SymbolicGradient node for function).
      grads = gradients.gradients(y, [x, b1])

      self.assertAllEqual([40.0], self.evaluate(grads)[0])
      self.assertAllEqual([10.0], self.evaluate(grads)[1])

  def testFunctionGradientsWithGradFunc(self):
    g = ops.Graph()
    with g.as_default():
      grad_func = framework_function.Defun(dtypes.float32, dtypes.float32,
                                           dtypes.float32)(
                                               self.XSquarePlusBGradient)
      f = self._GetFunc(grad_func=grad_func)
      # Get gradients (should add SymbolicGradient node for function, which
      # uses the grad_func above, which multiplies all gradients by 2).
      grads = self._GetFuncGradients(f, [2.0], [1.0])
      self.assertAllEqual([4.0 * 2], grads[0])
      self.assertAllEqual([1.0 * 2], grads[1])

  def testFunctionGradientWithRegistration(self):
    g = ops.Graph()
    with g.as_default():
      f = self._GetFunc(python_grad_func=self._PythonGradient)
      # Get gradients, using the python gradient function. It multiplies the
      # gradients by 3.
      grads = self._GetFuncGradients(f, [2.0], [1.0])
      self.assertAllEqual([4.0 * 3], grads[0])
      self.assertAllEqual([1.0 * 3], grads[1])

  def testFunctionGradientWithGradFuncAndRegistration(self):
    g = ops.Graph()
    with g.as_default():
      grad_func = framework_function.Defun(dtypes.float32, dtypes.float32,
                                           dtypes.float32)(
                                               self.XSquarePlusBGradient)
      with self.assertRaisesRegex(ValueError, "Gradient defined twice"):
        f = self._GetFunc(
            grad_func=grad_func, python_grad_func=self._PythonGradient)
        f.add_to_graph(ops.Graph())

  def testGradientWrtCaptured(self):
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")

      @def_function.function
      def Foo():
        y = math_ops.multiply(x, 2.0, name="y")
        g = gradients_impl.gradients(y, x)
        return g[0]

      f = Foo()

      self.assertEqual(self.evaluate(f), 2.0)

  def testGradientOfCaptured(self):
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")
      y = math_ops.multiply(x, 2.0, name="y")

      @framework_function.Defun()
      def Foo():
        g = gradients_impl.gradients(y, x)
        return g[0]

      f = Foo()

      self.assertEqual(self.evaluate(f), 2.0)

  def testCapturedResourceVariable(self):
    with ops.Graph().as_default():
      var = resource_variable_ops.ResourceVariable(1.0, name="var")

      @def_function.function
      def Foo():
        y = math_ops.multiply(var, 2.0, name="y")
        g = gradients_impl.gradients(y, var)
        return g[0]

      f = Foo()

      self.evaluate(variables.global_variables_initializer())
      self.assertEqual(self.evaluate(f), 2.0)

  def testCapturedNested(self):
    with ops.Graph().as_default():
      x1 = constant_op.constant(1.0, name="x1")
      x2 = constant_op.constant(2.0, name="x2")
      x3 = math_ops.multiply(x1, x2, name="x3")

      @def_function.function
      def Outer():
        outer1 = array_ops.identity(x1, name="outer1")

        @def_function.function
        def Inner():
          inner1 = array_ops.identity(outer1, name="inner1")
          inner2 = array_ops.identity(x2, name="inner2")
          inner3 = array_ops.identity(x3, name="inner3")
          return gradients_impl.gradients([inner1, inner2, inner3, x1],
                                          [x1, x2])

        return Inner()

      x1_grad, x2_grad = Outer()

      # 1.0 + None + 2.0 + 1.0 = 4.0
      self.assertEqual(self.evaluate(x1_grad), 4.0)
      # None + 1.0 + 1.0 + None = 2.0
      self.assertEqual(self.evaluate(x2_grad), 2.0)

  def testCapturedFromFunction(self):
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")

      @def_function.function
      def Outer():
        y = math_ops.multiply(x, 2.0, name="y")

        @def_function.function
        def Inner():
          z = math_ops.multiply(y, 3.0, name="z")
          g = gradients_impl.gradients(z, y)
          return g[0]

        return Inner()

      z_grad = Outer()

      self.assertEqual(self.evaluate(z_grad), 3.0)

  def testCapturedEagerTensors(self):
    # Test that we can handle captured eager tensors unrelated to the gradient
    # computation (i.e. we need to ignore them).
    # TODO(skyewm): make it an error if you try to take the gradient wrt a
    # captured EagerTensor
    with context.eager_mode():
      c = constant_op.constant(2.0, name="c")

      @def_function.function
      def Foo():
        x = constant_op.constant(10.0, name="x")
        y = math_ops.multiply(x, c, name="y")
        # Regression test for b/122564611.
        z = math_ops.multiply(c, y, name="z")
        g = gradients_impl.gradients(z, x)
        return g[0]

      self.assertEqual(Foo().numpy(), 4.0)


class StopGradientTest(test_util.TensorFlowTestCase):

  def testStopGradient(self):
    with ops.Graph().as_default():
      inp = constant(1.0, shape=[100, 32], name="in")
      out = array_ops.stop_gradient(inp)
      igrad = gradients.gradients(out, inp)[0]
    assert igrad is None


class PreventGradientTest(test_util.TensorFlowTestCase):

  def testPreventGradient(self):
    with ops.Graph().as_default():
      inp = constant(1.0, shape=[100, 32], name="in")
      out = array_ops.prevent_gradient(inp)
      with self.assertRaisesRegex(LookupError, "explicitly disabled"):
        _ = gradients.gradients(out, inp)


class HessianVectorProductTest(test_util.TensorFlowTestCase):

  @test_util.run_v1_only("b/120545219")
  def testHessianVectorProduct(self):
    # Manually compute the Hessian explicitly for a low-dimensional problem
    # and check that HessianVectorProduct matches multiplication by the
    # explicit Hessian.
    # Specifically, the Hessian of f(x) = x^T A x is
    # H = A + A^T.
    # We expect HessianVectorProduct(f(x), x, v) to be H v.
    m = 4
    rng = np.random.RandomState([1, 2, 3])
    mat_value = rng.randn(m, m).astype("float32")
    v_value = rng.randn(m, 1).astype("float32")
    x_value = rng.randn(m, 1).astype("float32")
    hess_value = mat_value + mat_value.T
    hess_v_value = np.dot(hess_value, v_value)
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        mat = constant_op.constant(mat_value)
        v = constant_op.constant(v_value)
        x = constant_op.constant(x_value)
        mat_x = math_ops.matmul(mat, x, name="Ax")
        x_mat_x = math_ops.matmul(array_ops.transpose(x), mat_x, name="xAx")
        hess_v = gradients_impl._hessian_vector_product(x_mat_x, [x], [v])[0]
        hess_v_actual = self.evaluate(hess_v)
      self.assertAllClose(hess_v_value, hess_v_actual)


class HessianTest(test_util.TensorFlowTestCase):

  @test_util.run_v1_only("b/120545219")
  def testHessian1D(self):
    # Manually compute the Hessian explicitly for a low-dimensional problem
    # and check that `hessian` matches. Specifically, the Hessian of
    # f(x) = x^T A x is H = A + A^T.
    m = 4
    rng = np.random.RandomState([1, 2, 3])
    mat_value = rng.randn(m, m).astype("float32")
    x_value = rng.randn(m).astype("float32")
    hess_value = mat_value + mat_value.T
    with self.session():
      mat = constant_op.constant(mat_value)
      x = constant_op.constant(x_value)
      x_mat_x = math_ops.reduce_sum(x[:, None] * mat * x[None, :])
      hess = gradients.hessians(x_mat_x, x)[0]
      hess_actual = self.evaluate(hess)
    self.assertAllClose(hess_value, hess_actual)

  @test_util.run_v1_only("b/120545219")
  def testHessian1D_multi(self):
    # Test the computation of the hessian with respect to multiple tensors
    m = 4
    n = 3
    rng = np.random.RandomState([1, 2, 3])
    mat_values = [rng.randn(m, m).astype("float32") for _ in range(n)]
    x_values = [rng.randn(m).astype("float32") for _ in range(n)]
    hess_values = [mat_value + mat_value.T for mat_value in mat_values]
    with self.session():
      mats = [constant_op.constant(mat_value) for mat_value in mat_values]
      xs = [constant_op.constant(x_value) for x_value in x_values]
      xs_mats_xs = [
          math_ops.reduce_sum(x[:, None] * mat * x[None, :])
          for x, mat in zip(xs, mats)
      ]
      hessians = gradients.hessians(xs_mats_xs, xs)
      hessians_actual = [hess.eval() for hess in hessians]
    for hess_value, hess_actual in zip(hess_values, hessians_actual):
      self.assertAllClose(hess_value, hess_actual)

  @test_util.run_v1_only("b/120545219")
  def testHessianInvalidDimension(self):
    for shape in [(10, 10), None]:
      with self.cached_session():
        x = array_ops.placeholder(dtypes.float32, shape)
        # Expect a ValueError because the dimensions are wrong
        with self.assertRaises(ValueError):
          gradients.hessians(x, x)

  @test_util.run_v1_only("b/120545219")
  def testHessian2D_square_matrix(self):
    # Manually compute the Hessian explicitly for a low-dimensional problem
    # and check that `hessian` matches. Specifically, the Hessian of
    # f(x) = 1/2 * x^T * x is H = constant (block identity matrix)
    m = 3
    rng = np.random.RandomState([1, 2, 3])
    x_value = rng.randn(m, m).astype("float32")
    with self.session():
      x = constant_op.constant(x_value)
      x_square = math_ops.reduce_sum(
          math_ops.matmul(array_ops.transpose(x), x) * 0.5
      )
      hess = gradients.hessians(x_square, x)[0]
      hess_actual = self.evaluate(hess)
    hess_value = np.bmat([
        [elem*np.ones((m, m)) for elem in vec]
        for vec in np.eye(m)
    ]).astype("float32")
    self.assertAllEqual((m, m, m, m), hess_actual.shape)
    self.assertAllClose(hess_value, hess_actual.reshape((m * m, m * m)))

  @test_util.run_v1_only("b/120545219")
  def testHessian2D_non_square_matrix(self):
    m = 3
    n = 4
    rng = np.random.RandomState([1, 2, 3])
    x_value = rng.randn(m, n).astype("float32")
    with self.session():
      x = constant_op.constant(x_value)
      x_square = math_ops.reduce_sum(
          math_ops.matmul(array_ops.transpose(x), x) * 0.5
      )
      hess = gradients.hessians(x_square, x)[0]
      hess_actual = self.evaluate(hess)
    hess_value = np.bmat([
        [elem*np.ones((n, n)) for elem in vec]
        for vec in np.eye(m)
    ]).astype("float32")
    self.assertAllEqual((m, n, m, n), hess_actual.shape)
    self.assertAllClose(hess_value, hess_actual.reshape((m * n, m * n)))


class IndexedSlicesToTensorTest(test_util.TensorFlowTestCase):

  @test_util.run_v1_only("b/120545219")
  def testIndexedSlicesToTensor(self):
    with self.cached_session():
      np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
      c = constant_op.constant(np_val)
      c_sparse = math_ops._as_indexed_slices(c)
      self.assertAllEqual(np_val.shape, c_sparse.dense_shape)
      c_dense = math_ops.multiply(c_sparse, 1.0)
      self.assertAllClose(np_val, self.evaluate(c_dense))

  @test_util.run_v1_only("b/120545219")
  def testIndexedSlicesToTensorList(self):
    with self.cached_session():
      numpy_list = []
      dense_list = []
      sparse_list = []
      for _ in range(3):
        np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
        c = constant_op.constant(np_val)
        c_sparse = math_ops._as_indexed_slices(c)
        numpy_list.append(np_val)
        dense_list.append(c)
        sparse_list.append(c_sparse)
      packed_dense = array_ops_stack.stack(dense_list)
      packed_sparse = array_ops_stack.stack(sparse_list)
      self.assertAllClose(packed_dense, self.evaluate(packed_sparse))

  @test_util.run_v1_only("b/120545219")
  def testInt64Indices(self):
    with self.cached_session():
      np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
      c = constant_op.constant(np_val)
      c_sparse = math_ops._as_indexed_slices(c)
      c_sparse = indexed_slices.IndexedSlices(
          c_sparse.values,
          math_ops.cast(c_sparse.indices, dtypes.int64), c_sparse.dense_shape)
      self.assertAllEqual(np_val.shape, c_sparse.dense_shape)
      c_dense = math_ops.multiply(c_sparse, 1.0)
      self.assertAllClose(np_val, self.evaluate(c_dense))

  @test_util.run_v1_only("b/120545219")
  def testWarnings(self):
    # TODO(gunan) Reenable after this issue is fixed:
    # https://github.com/google/protobuf/issues/2812
    if sys.version_info >= (3, 5):
      self.skipTest("Skipped test for Python 3.5+")

    # Smaller than the threshold: no warning.
    c_sparse = indexed_slices.IndexedSlices(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32), constant([4, 4, 4, 4]))
    with warnings.catch_warnings(record=True) as w:
      math_ops.multiply(c_sparse, 1.0)
    self.assertEqual(0, len(w))

    # Greater than or equal to the threshold: warning.
    c_sparse = indexed_slices.IndexedSlices(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32), constant([100, 100, 100, 100]))
    # "always" filter prevents the warning from being suppressed if it was
    # already triggered in a different test.
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
      math_ops.multiply(c_sparse, 1.0)
    self.assertEqual(1, len(w))
    self.assertIn(
        "with 100000000 elements. This may consume a large amount of memory.",
        str(w[0].message))

    # Unknown dense shape: warning.
    c_sparse = indexed_slices.IndexedSlices(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32),
        array_ops.placeholder(dtypes.int32))
    with warnings.catch_warnings(record=True) as w:
      math_ops.multiply(c_sparse, 1.0)
    self.assertEqual(1, len(w))
    self.assertIn(
        "of unknown shape. This may consume a large amount of memory.",
        str(w[0].message))


class OnlyRealGradientsTest(test_util.TensorFlowTestCase):

  @test_util.run_v1_only("b/120545219")
  def testRealOnly(self):
    x = constant_op.constant(7+3j, dtype=dtypes.complex64)
    y = math_ops.square(x)
    with self.assertRaisesRegex(
        TypeError, r"Gradients of complex tensors .* must set grad_ys "
        r"\(y\.dtype = complex64\)"):
      gradients.gradients(y, x)


class ResourceCondTest(test_util.TensorFlowTestCase):

  @test_util.run_v1_only("b/120545219")
  def testBasic(self):
    gamma = resource_variable_ops.ResourceVariable(
        np.random.random((3,)),
        dtype="float32", name="gamma")

    inputs = array_ops.ones(shape=(3,), dtype="float32")

    def TestFn():
      output = inputs + gamma
      return output

    training = array_ops.placeholder_with_default(True, shape=())
    output = control_flow_ops.cond(
        training, TestFn, lambda: inputs)

    loss = output

    grads = gradients.gradients(
        loss, [gamma])
    self.assertNotIn(None, grads)


class GetDependentVariablesTest(test_util.TensorFlowTestCase):

  def testNoVariables(self):
    with ops.Graph().as_default():
      func = lambda x: array_ops.identity(x) + 5.0
      input_t = constant_op.constant(2.0)
      result_t = func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])

      # There are no variables.
      self.assertEqual(dependent_vars, [])

  def testVariablesOutside(self):
    with ops.Graph().as_default():
      init = constant_op.constant(100.0)
      var = variables.Variable(init)

      # The variable is closed over. It should be found.
      func = lambda x: array_ops.identity(x) + 5.0 + var

      input_t = constant_op.constant(2.0)
      result_t = func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(dependent_vars, [var])

  def testVariableSamePrefix(self):
    with ops.Graph().as_default():
      var_name = "my_variable"
      v_z = variable_scope.get_variable(var_name, shape=())
      v_o = variable_scope.get_variable(var_name + "_ones", shape=())

      # The variable is closed over. It should be found.
      func = lambda x: array_ops.identity(x) + 5.0 + v_z + v_o

      input_t = constant_op.constant(2.0)
      result_t = func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(set(dependent_vars), set([v_o, v_z]))

  def testVariablesOutsideButDSeparated(self):
    with ops.Graph().as_default():
      init = constant_op.constant(100.0)
      var = variables.Variable(init)

      # The variable is d-separated by the inputs. It should not be found.
      input_t = array_ops.identity(var) * 5.0

      func = lambda x: array_ops.identity(x) + 5.0
      result_t = func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(dependent_vars, [])

  def testVariablesOutsideAndNonDifferentiable(self):
    with ops.Graph().as_default():
      init = constant_op.constant(100.0, shape=(5,))
      var = variables.Variable(init, shape=(5,))

      def _Func(x):
        # non-differentiable dependency on var.
        # the variable should not be found.
        y = array_ops.ones_like(var)
        return array_ops.identity(x) + 5.0 + y

      input_t = constant_op.constant(2.0)
      result_t = _Func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(dependent_vars, [])

  def testGetVariableByName(self):
    with context.graph_mode():
      init = constant_op.constant(100.0)
      var = variable_scope.variable(init, name="a/replica_1")
      if isinstance(var, variables.RefVariable):
        var._variable = array_ops.identity(var, name="a")
      else:
        var._handle = array_ops.identity(var, name="a")
      var2 = custom_gradient.get_variable_by_name("a")
      self.assertEqual(var2.name, var.name)

  def testVariablesOutsideAndNonTrainable(self):
    with ops.Graph().as_default():
      init = constant_op.constant(100.0, shape=(5,))

      # Both variables are used in the function but only the trainable one
      # should be found.
      var_trainable = variables.Variable(init, shape=(5,))
      var_nontrainable = variables.Variable(init, shape=(5,), trainable=False)

      def _Func(x):
        del x
        return var_trainable + var_nontrainable

      input_t = constant_op.constant(2.0)
      result_t = _Func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(dependent_vars, [var_trainable])

  def testVariablesOutsideAndCustomGradient(self):
    with ops.Graph().as_default():
      init = constant_op.constant(100.0, shape=(5,))
      var = variables.Variable(init, shape=(5,))

      @custom_gradient.custom_gradient
      def _MyOnesLike(x):
        """Dummy version of ones_like which defines a gradient."""

        output = array_ops.ones_like(x)

        def _Grad(dy):
          return array_ops.identity(dy)

        return output, _Grad

      def _Func(x):
        # non-differentiable operation with custom gradient.
        # The variable should be found.
        y = _MyOnesLike(var)
        return array_ops.identity(x) + 5.0 + y

      input_t = constant_op.constant(2.0)
      result_t = _Func(input_t)
      dependent_vars = custom_gradient._get_dependent_variables(
          [input_t], [result_t])
      self.assertEqual(dependent_vars, [var])


class CustomGradientTest(test_util.TensorFlowTestCase, parameterized.TestCase):

  def testCustomGradientTrivial(self):

    @custom_gradient.custom_gradient
    def MyIdentity(x):

      def Grad(dy):
        return [3 * dy]

      return x, Grad

    with ops.Graph().as_default():
      x = constant(3.)
      y = MyIdentity(MyIdentity(x))
      dy = gradients.gradients(y, x)[0]
      with session.Session():
        self.assertEqual(9., self.evaluate(dy))

  def testCustomGradient(self):

    @custom_gradient.custom_gradient
    def MyMultiply(x1, x2):
      result = x1 * x2

      def Grad(dy):
        # Switched the ordering here.
        return [dy * x1, dy * x2]

      return result, Grad

    with ops.Graph().as_default():
      x1 = constant(3.)
      x2 = constant(5.)
      y = MyMultiply(x1, x2)
      dy = gradients.gradients(y, [x1, x2])

      self.assertAllEqual([3., 5.], self.evaluate(dy))

  def testCustomGradientClass(self):

    class Model:

      @custom_gradient.custom_gradient
      def Multiply(self, x1, x2):
        result = x1 * x2
        grad = lambda dy: (dy * x1, dy * x2)
        return result, grad

    with ops.Graph().as_default():
      x1 = constant(3.)
      x2 = constant(5.)
      m = Model()
      y = m.Multiply(x1, x2)
      dy = gradients.gradients(y, [x1, x2])
      self.assertAllEqual([3., 5.], self.evaluate(dy))

  def testCustomGradientErrors(self):

    @custom_gradient.custom_gradient
    def F(x):

      def Grad(_):
        raise RuntimeError("x")

      return x, Grad

    with ops.Graph().as_default():
      x = constant(1.0)
      y = F(x)
      with self.assertRaises(RuntimeError):
        gradients.gradients(y, x)

  def testCustomGradientWithVariables(self):

    @custom_gradient.custom_gradient
    def F(x):
      out = core_layers.dense(x, 3, use_bias=False)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
        grads = gradients.gradients(out, [x, variables[0]], grad_ys=out_grad)
        return grads[0], [array_ops.ones((4, 3))]

      return out, Grad

    with ops.Graph().as_default():
      x = array_ops.ones((2, 4))
      with variable_scope.variable_scope("f", use_resource=True) as vs:
        y = F(x)
        all_vars = vs.global_variables()
        assert len(all_vars) == 1
      grads = gradients.gradients(y, [x, all_vars[0]])
      for g in grads:
        self.assertIsNotNone(g)

      self.evaluate(variables.global_variables_initializer())
      dw = self.evaluate(math_ops.reduce_sum(grads[1]))
      self.assertEqual(12., dw)

  @parameterized.named_parameters([
      dict(
          testcase_name="Eager",
          decorator=lambda f: f),
      dict(
          testcase_name="Function",
          decorator=def_function.function),
  ])
  def testCustomGradientRaggedTensor(self, decorator):
    with context.eager_mode():
      @custom_gradient.custom_gradient
      def F(x):
        out = x * x

        def Grad(*grad):
          return 3 * grad[0]

        return out, Grad

      rt = ragged_factory_ops.constant([[1., 2.], [3.]])
      with backprop.GradientTape() as tape:
        tape.watch(rt.values)
        out = decorator(F)(rt)
        result = tape.gradient(out, rt)

      self.assertIsInstance(out, ragged_tensor.RaggedTensor)
      self.assertAllEqual(out, [[1., 4.], [9.]])
      self.assertIsInstance(result, ragged_tensor.RaggedTensor)
      self.assertAllEqual(result, [[3., 3.], [3.]])

  @parameterized.named_parameters([
      dict(
          testcase_name="Eager",
          decorator=lambda f: f),
      dict(
          testcase_name="Function",
          decorator=def_function.function),
  ])
  def testCustomGradientMultipleRaggedTensors(self, decorator):
    with context.eager_mode():
      @custom_gradient.custom_gradient
      def F(x, y):
        out = (x * x, 2 * y)

        def Grad(*grad):
          return (3 * grad[0], 4 * grad[1])

        return out, Grad

      rt1 = ragged_factory_ops.constant([[1., 2.], [3.]])
      rt2 = ragged_factory_ops.constant([[4.], [5., 6.]])
      with backprop.GradientTape() as tape:
        tape.watch((rt1, rt2))
        out1, out2 = decorator(F)(rt1, rt2)
        grad1, grad2 = tape.gradient((out1, out2), (rt1, rt2))

      self.assertIsInstance(out1, ragged_tensor.RaggedTensor)
      self.assertAllEqual(out1, [[1., 4.], [9.]])
      self.assertIsInstance(out2, ragged_tensor.RaggedTensor)
      self.assertAllEqual(out2, [[8.], [10., 12.]])
      self.assertIsInstance(grad1, ragged_tensor.RaggedTensor)
      self.assertAllEqual(grad1, [[3., 3.], [3.]])
      self.assertIsInstance(grad2, ragged_tensor.RaggedTensor)
      self.assertAllEqual(grad2, [[4.], [4., 4.]])

  @test_util.enable_quantized_dtypes_training
  def testCustomGradientQuantizedDtypeTraining(self):
    with context.eager_mode():
      @custom_gradient.custom_gradient
      def F(x):
        out = x

        def Grad(*grad):
          return grad

        return out, Grad

      x = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.qint8)
      with backprop.GradientTape() as tape:
        tape.watch(x)
        out = F(x)
        result = tape.gradient(out, x)

      self.assertAllEqual(out, [[1, 2], [3, 4]])
      self.assertAllEqual(result, [[1, 1], [1, 1]])
      self.assertEqual(result.dtype, dtypes.qint8)

  def testCustomGradientWithCapture(self):
    with ops.Graph().as_default():
      x = constant(3.)

      @framework_function.Defun(dtypes.float32)
      def F(y):

        @custom_gradient.custom_gradient
        def MyMultiply(x1, x2):
          result = x1 * x2

          def Grad(dy):
            # Switched the ordering here.
            return [dy * x1, dy * x2]

          return result, Grad

        res = MyMultiply(x, y)
        return gradients.gradients(res, [y])

      y = constant(5.)
      dy = F(y)
      self.assertAllEqual(5., self.evaluate(dy))

  def testCustomGradientWithVariablesNoFalsePositives(self):

    @custom_gradient.custom_gradient
    def F(x):
      out = core_layers.dense(x, 3, use_bias=False)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
        grads = gradients.gradients(out, [x, variables[0]], grad_ys=out_grad)
        return grads[0], [array_ops.ones((3, 3))]

      return out, Grad

    with ops.Graph().as_default():
      with variable_scope.variable_scope("f", use_resource=True) as vs:
        a = array_ops.ones((2, 4))

        # Variabes in these layers shouldn't be picked up by the decorator.
        b = core_layers.dense(a, 3, use_bias=False)
        c = core_layers.dense(b, 3, use_bias=False)
        x = core_layers.dense(b, 3, use_bias=False) + c

        # Only the variables used in F.
        y = F(x)

        all_vars = vs.global_variables()
        assert len(all_vars) == 4
      grads = gradients.gradients(y, [x] + all_vars)
      _, var_grads = grads[0], grads[1:]
      for g in grads:
        self.assertIsNotNone(g)

      self.evaluate(variables.global_variables_initializer())
      dw = self.evaluate(math_ops.reduce_sum(var_grads[-1]))
      self.assertEqual(9., dw)

  def testCustomGradientWithVariablesEager(self):
    with context.eager_mode():
      layer = core_layers.Dense(4, use_bias=False)

      @custom_gradient.custom_gradient
      def F(x):
        out = layer(x)

        def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
          del out_grad
          self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
          return (array_ops.ones((3, 2)),
                  [array_ops.ones((2, 4))])

        return out, Grad

      x = array_ops.ones((3, 2)) + 2.
      with backprop.GradientTape() as tape:
        tape.watch(x)
        y = F(x)
      w, = layer.variables
      dx, dw = tape.gradient(y, [x, w])
      self.assertEqual(6., math_ops.reduce_sum(dx).numpy())
      self.assertEqual(8., math_ops.reduce_sum(dw).numpy())

  @test_util.run_v1_only("b/120545219")
  def testCustomGradientErrorsWithNonResourceVariables(self):

    def F(x, use_resource=False):
      with variable_scope.variable_scope("f", use_resource=use_resource):
        out = core_layers.dense(x, 4, use_bias=False)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        del out_grad
        self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
        return (array_ops.ones((3, 2)), [array_ops.ones((2, 4))])

      return out, Grad

    @custom_gradient.custom_gradient
    def FResource(x):
      return F(x, use_resource=True)

    @custom_gradient.custom_gradient
    def FNonResource(x):
      return F(x, use_resource=False)

    x = array_ops.ones((3, 2)) + 2.

    # Wrapping scope has use_resource=True but inner scope sets to False. Fails.
    with variable_scope.variable_scope("vs1", use_resource=True):
      with self.assertRaisesWithPredicateMatch(TypeError,
                                               "must be `ResourceVariable`s"):
        FNonResource(x)

    # Wrapping scope has use_resource=False but inner scope sets to True.
    # Passes.
    with variable_scope.variable_scope("vs2", use_resource=False):
      FResource(x)

  @parameterized.parameters(True, False)
  def testCustomGradientVariablesKwonlyArgs(self, anonymous_varargs):
    with context.eager_mode():
      x_captured = variables.Variable(3.)  # Used by FuncMult
      @custom_gradient.custom_gradient
      def FuncMult(x):
        def ActualGrad(dy, variables):  # pylint: disable=redefined-outer-name
          self.assertLen(variables, 1)
          self.assertIs(variables[0], x_captured)
          x_captured_grad = 5. * x * dy
          return (4. * x_captured * dy, [x_captured_grad])
        # Define the returned GradMult, using varargs; "variables" is kwonlyarg
        if anonymous_varargs:
          def GradMult(dy, *, variables=None):  # pylint: disable=redefined-outer-name
            return ActualGrad(dy, variables)
        else:
          def GradMult(*dys, variables=None):  # pylint: disable=redefined-outer-name
            return ActualGrad(dys[0], variables)

        return x * x_captured, GradMult

      x = variables.Variable(6.)
      with backprop.GradientTape(persistent=True) as g:
        y = FuncMult(x)
      self.assertAllEqual(g.gradient(y, x), 4. * 3.)

  def testWithNumpyInputs(self):
    with context.eager_mode():

      @custom_gradient.custom_gradient
      def F(x):
        out = x

        def Grad(_):
          return (None, None)

        return out, Grad

      x = np.ones((3, 2), dtype=np.float32)
      # Smoke test to ensure numpy inputs are accepted
      F(x)

  @test_util.run_v1_only("b/120545219")
  def testRVGradientsDynamicCond(self):
    with self.cached_session():
      alpha = resource_variable_ops.ResourceVariable(
          np.random.random((1,)),
          dtype="float32")

      conditional = array_ops.placeholder_with_default(True, shape=())
      output = control_flow_ops.cond(
          conditional, lambda: alpha * 2, lambda: alpha * 3)

      g, = gradients_impl.gradients(output, alpha)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllEqual(g, [2.0])
      self.assertAllEqual(g.eval(feed_dict={conditional: False}), [3.0])

  def testRecursiveCustomGradient(self):
    @custom_gradient.custom_gradient
    def F(x):
      out = core_layers.dense(x, 3, use_bias=False)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
        grads = gradients.gradients(out, [x, variables[0]], grad_ys=out_grad)
        return grads[0], [array_ops.ones((4, 3))]

      return out, Grad

    @custom_gradient.custom_gradient
    def DoubleF(x):
      out = F(x)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        self.assertEqual(1, len(variables))  # pylint: disable=g-generic-assert
        grads = gradients.gradients(out, [x, variables[0]], grad_ys=out_grad)
        return grads[0], [array_ops.ones((4, 3))]

      return out, Grad
    with ops.Graph().as_default():
      x = array_ops.ones((2, 4))
      with variable_scope.variable_scope("f", use_resource=True) as vs:
        y = DoubleF(x)
        all_vars = vs.global_variables()
        assert len(all_vars) == 1
      grads = gradients.gradients(y, [x, all_vars[0]])
      for g in grads:
        self.assertIsNotNone(g)

      self.evaluate(variables.global_variables_initializer())
      dw = self.evaluate(math_ops.reduce_sum(grads[1]))
      self.assertEqual(12., dw)

  @parameterized.named_parameters(
      [(("_%s_%s" % (x_struct, y_struct)).replace(" ", "").replace("None", ""),  # pylint: disable=g-complex-comprehension
        x_struct, y_struct)
       for y_struct in [[None, ()], (None, (), [], (None, ((), None)))]
       for x_struct in [(None, ()), (((), ()), [None, None], [], (None, ()))]
      ])
  @test_util.run_in_graph_and_eager_modes
  def testCustomGradientStructuralInputOutput(self, x_struct, y_struct):
    """Tests that custom_gradient can handle structured inputs/outputs."""
    def Zeros(x):
      return nest.map_structure(lambda _: array_ops.zeros([], "float32"), x)
    def GetStruct(x):
      return nest.map_structure(lambda _: None, x)

    def MakeVjp(f, *x):
      with backprop.GradientTape(persistent=True) as tape:
        tape.watch(nest.flatten(x))
        y = f(*x)
      def Vjp(dy):
        return tape.gradient(y, x, output_gradients=dy)
      return y, Vjp

    @custom_gradient.custom_gradient
    def F(*x):
      self.assertEqual(x_struct, GetStruct(x))
      def Vjp(*dy):
        self.assertEqual(len(nest.flatten(y_struct)),
                         len(nest.flatten(dy)))
        return nest.flatten(Zeros(x_struct))
      return Zeros(y_struct), Vjp

    x, dy = Zeros([x_struct, y_struct])
    y, vjp = MakeVjp(F, *x)
    dx = vjp(dy)
    self.assertEqual(x_struct, GetStruct(dx))
    self.assertEqual(y_struct, GetStruct(y))


class TensorListGradientsTest(test_util.TensorFlowTestCase):

  def testDefaultGradYs(self):
    with ops.Graph().as_default():
      tl = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=ops.convert_to_tensor([], dtype=dtypes.int32))
      a = constant(1.0)
      tl = list_ops.tensor_list_push_back(tl, a)

      grad_tl = list_ops.empty_tensor_list(
          element_dtype=dtypes.float32,
          element_shape=ops.convert_to_tensor([], dtype=dtypes.int32))
      grad_tl = list_ops.tensor_list_push_back(tl, constant(5.0))

      grad = gradients.gradients(tl, a, grad_ys=grad_tl)[0]

      self.assertEqual(self.evaluate(grad), 5.)


class VariablesGradientTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):

  def _TestFnVariablesGradient(self, inputs, test_fn, vars_to_grad):
    """Returns gradients of `test_model` with respect to `vars_to_grad`."""

    test_fn_re = custom_gradient.recompute_grad(test_fn)

    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(vars_to_grad)
      out_re = test_fn_re(inputs, vars_to_grad)
      out = test_fn(inputs, vars_to_grad)

    grads_re = tape.gradient(out_re, vars_to_grad)
    grads = tape.gradient(out, vars_to_grad)

    return grads_re, grads

  def _grad(self, f, argnums=0):
    """Return a function which computes the gradient of `f`."""

    def F(*params):
      with backprop.GradientTape() as tape:
        tape.watch(params)
        outputs = f(*params)
      return tape.gradient(
          outputs,
          params[argnums],
          unconnected_gradients=unconnected_gradients.UnconnectedGradients.ZERO)

    return F

  def _test_gradients(self, f, inputs, order, delta=1e-3, rtol=1e-2, atol=1e-6):
    """Tests backward jacobians of `f`'s [0, `order`)-order gradients."""
    if order < 1:
      raise ValueError(
          "`order` should be a positive integer, got '{}'.".format(order))
    if order > 1:
      self._test_gradients(
          f=self._grad(f),
          inputs=inputs,
          order=order - 1,
          delta=delta,
          rtol=rtol,
          atol=atol)
    sym_jac_back, num_jac = gradient_checker_v2.compute_gradient(
        f, inputs, delta=delta)
    self.assertAllClose(num_jac, sym_jac_back, rtol=rtol, atol=atol)

  def testRecomputeGradWrapped(self):

    def f(x):  # pylint: disable=invalid-name
      return 2 * x

    g = custom_gradient.recompute_grad(f)
    self.assertIs(g.__wrapped__, f)

  def testRecomputeGradZeroSizeInput(self):

    def F(x):
      return 2 * x

    x = array_ops.constant(())
    grads_re = self._grad(custom_gradient.recompute_grad(F))(x)
    grads = self._grad(F)(x)
    self.assertAllClose(grads_re, grads)

    f_graph = def_function.function(
        F, input_signature=[tensor_spec.TensorSpec(None)])
    grads_re = self._grad(custom_gradient.recompute_grad(f_graph))(x)
    grads = self._grad(f_graph)(x)
    self.assertAllClose(grads_re, grads)

  def testRecomputeGradDifferentDtypesInputs(self):

    def F(x1, x2):
      return 2 * x1, 2 * x2

    x1 = array_ops.constant(1, dtype=dtypes.int32)
    x2 = array_ops.constant(1., dtype=dtypes.float32)
    grads_re = self._grad(custom_gradient.recompute_grad(F))(x1, x2)
    grads = self._grad(F)(x1, x2)
    self.assertAllClose(grads_re, grads)

    f_graph = def_function.function(
        F,
        input_signature=[
            tensor_spec.TensorSpec(None, dtype=dtypes.int32),
            tensor_spec.TensorSpec(None, dtype=dtypes.float32),
        ])
    grads_re = self._grad(custom_gradient.recompute_grad(f_graph))(x1, x2)
    grads = self._grad(f_graph)(x1, x2)
    self.assertAllClose(grads_re, grads)

  @test_util.run_v2_only
  def testCustomGradientRecomputeGradHigherOrder(self):

    @custom_gradient.recompute_grad
    def F(x):
      return math_ops.reduce_prod(math_ops.tanh(x)**2)

    self._test_gradients(F, [constant_op.constant([1.])], order=3)

  @test_util.run_in_graph_and_eager_modes
  def testFnRecompute(self):
    """Checks that recompute_grad works grads of function args."""

    def TestFn(inputs, input_vars):
      return inputs * input_vars

    def TestFnSeq(inputs, input_vars):
      return (inputs * input_vars, inputs * input_vars * 2.0)

    with variable_scope.variable_scope("test", use_resource=True):
      test_var = variable_scope.get_variable(
          name="test_var",
          shape=10,
          trainable=True,
      )
      self.evaluate(test_var.assign(np.ones([10])))
      test_input = constant(np.ones((10, 10), dtype=np.float32))

      grads_re, grads = self._TestFnVariablesGradient(test_input, TestFn,
                                                      test_input)

      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)

      grads_re, grads = self._TestFnVariablesGradient(test_input, TestFn,
                                                      test_var)
      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)

      # Regression test for wrapping sequence outputting functions.
      grads_re, grads = self._TestFnVariablesGradient(test_input, TestFnSeq,
                                                      test_input)
      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)

      grads_re, grads = self._TestFnVariablesGradient(test_input, TestFnSeq,
                                                      test_var)
      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testFnRecomputeWithScopeGradient(self, use_tape):
    """Checks that recompute_grad works with var scope and GradientTape."""

    def TestFn(input_t):
      with variable_scope.variable_scope("inner_scope"):
        test_var = variable_scope.get_variable(
            name="test_var",
            shape=10,
            trainable=True,
        )
        return input_t * test_var

    test_input_t = constant(np.zeros((10, 10), dtype=np.float32))

    with variable_scope.variable_scope(
        "output_scope", reuse=variable_scope.AUTO_REUSE, use_resource=True):
      test_fn_re = custom_gradient.recompute_grad(TestFn)

      with test_util.AbstractGradientTape(
          use_tape=use_tape, persistent=True) as tape:
        out_re = test_fn_re(test_input_t)
        out = TestFn(test_input_t)

    self.evaluate(variables.global_variables_initializer())
    grads_re = tape.gradient(out_re, variables.trainable_variables())
    grads = tape.gradient(out, variables.trainable_variables())

    grads_re = self.evaluate(grads_re)
    grads = self.evaluate(grads)
    for g, g_re in zip(grads, grads_re):
      self.assertAllClose(g, g_re)

  @test_util.run_in_graph_and_eager_modes
  def testFnRecomputeSameTensor(self):
    """Check recompute_grad when wrapped f called as f(x, x) - b/147369366."""

    def TestFnMul(x, y):
      return x * y

    def TestFnSingleVar(x, y):
      # pylint: disable=unused-argument
      return x

    with variable_scope.variable_scope("test", use_resource=True):
      x = array_ops.ones((10))

      grads_re, grads = self._TestFnVariablesGradient(x, TestFnMul,
                                                      x)
      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)

      grads_re, grads = self._TestFnVariablesGradient(x, TestFnSingleVar,
                                                      x)
      grads_re = self.evaluate(grads_re)
      grads = self.evaluate(grads)
      for g, g_re in zip(grads, grads_re):
        self.assertAllClose(g, g_re)


class GradPassThroughTest(test_util.TensorFlowTestCase):

  @test_util.run_v1_only("b/120545219")
  def test_gradients_v1(self):
    x = variable_scope.get_variable(
        name="x", shape=(), initializer=init_ops.constant_initializer(1.0),
        use_resource=True)
    z = variable_scope.get_variable(
        name="z", shape=(), initializer=init_ops.constant_initializer(3.0),
        use_resource=True)

    # Verify that assign op is not differentiable
    y = state_ops.assign(x, z**2)
    grads = gradients.gradients(y, z)
    self.assertIsNone(grads[0])

    # Verify that when the (non differentiable) assign op is wrapped with
    # grad_pass_through, gradients are correctly forwarded to the inputs.
    # Form an input as quadratic function of variable z and check that the
    # gradient of output wrt to z is correct.
    y = custom_gradient.grad_pass_through(
        lambda v: state_ops.assign(x, v))(z**2)
    grads = gradients.gradients(y, z)

    with self.cached_session():
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose(grads[0], 6.0)

    # Verify that variables involved in the wrapped op do not receive gradients.
    y = custom_gradient.grad_pass_through(lambda v: x * v)(z)
    grads = gradients.gradients(y, x)
    self.assertIsNone(grads[0])

  @test_util.run_v2_only
  def test_gradients_v2(self):
    x = variables.Variable(1.0, name="x")
    z = variables.Variable(3.0, name="z")

    # Verify that assign op is not differentiable
    with backprop.GradientTape() as tape:
      y = x.assign(z**2)
    grads = tape.gradient(y, z)
    self.assertIsNone(grads)

    # Verify that when the (non differentiable) assign op is wrapped with
    # grad_pass_through, gradients are correctly forwarded to the inputs.
    # Form an input as quadratic function of variable z and check that the
    # gradient of output wrt to z is correct.
    with backprop.GradientTape() as tape:
      y = custom_gradient.grad_pass_through(x.assign)(z**2)
    grads = tape.gradient(y, z)
    self.assertAllClose(grads, 6.0)

    # Verify that variables involved in the wrapped op do not receive gradients.
    with backprop.GradientTape() as tape:
      y = custom_gradient.grad_pass_through(lambda v: x * v)(z)
    grads = tape.gradient(y, x)
    self.assertIsNone(grads)


if __name__ == "__main__":
  googletest.main()
