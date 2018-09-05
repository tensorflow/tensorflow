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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import warnings

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.layers import core as core_layers
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops  # pylint: disable=unused-import
from tensorflow.python.ops import functional_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gradients
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.ops.nn_ops import bias_add
from tensorflow.python.platform import googletest


class GradientsTest(test_util.TensorFlowTestCase):

  def testGradients(self):
    with ops.Graph().as_default():
      inp = constant(1.0, shape=[32, 100], name="in")
      w = constant(1.0, shape=[100, 10], name="w")
      b = constant(1.0, shape=[10], name="b")
      xw = math_ops.matmul(inp, w, name="xw")
      h = bias_add(xw, b, name="h")
      w_grad = gradients.gradients(h, w)[0]
    self.assertEquals("MatMul", w_grad.op.type)
    self.assertEquals(w_grad.op._original_op, xw.op)
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
    self.assertEquals("MatMul", gw.op.type)

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
      self.assertTrue(wx.op.colocation_groups() != gw2.op.colocation_groups())

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
      self.assertTrue(w.op.colocation_groups() != gw2.op.colocation_groups())

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
      with session.Session():
        # Make sure the placer doesn't complain.
        gz_x.eval()

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
        self.assertEquals(float_grad.dtype, dtypes.float32)
        self.assertFalse(string_grad)
        return float_grad

      ops.RegisterGradient("TestStringOutput")(_TestOpGrad)

      c = constant(1.0)
      x, _ = test_ops.test_string_output(c)
      z = x * 2.0
      w = z * 3.0
      grads = gradients.gradients(z, [c])
      self.assertTrue(isinstance(grads[0], ops.Tensor))
      grads = gradients.gradients(w, [c])
      self.assertTrue(isinstance(grads[0], ops.Tensor))

  def testSingletonIndexedSlices(self):
    with ops.Graph().as_default():
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.identity(x)
      dy = ops.IndexedSlices(
          array_ops.placeholder(dtypes.float32),
          array_ops.placeholder(dtypes.int32))
      dx, = gradients.gradients(y, x, grad_ys=dy)
      # The IndexedSlices gradient of tf.identity is the identity map.
      with self.cached_session() as sess:
        vdx, vdy = sess.run(
            [dx, dy], feed_dict={x: [1.0], dy.indices: [0], dy.values: [2.0]})
      self.assertEqual(vdx, vdy)

  def testNonDifferentiableSwitchInWhileLoop(self):
    with ops.Graph().as_default():
      v = array_ops.placeholder(dtypes.float32, [])

      def _Step(i, a, ta):
        a += math_ops.cast(v, dtypes.int32)
        return (i + 1, a, ta.write(i, a))

      n = 4
      i, _, ta = control_flow_ops.while_loop(
          lambda i, *_: i < n,
          _Step, [0, 0, tensor_array_ops.TensorArray(
              dtypes.int32, size=n)])
      target = ta.read(i - 1)
      grad, = gradients.gradients(target, v)
      self.assertIsNone(grad)

  def testVariableReadValueGradient(self):
    with ops.Graph().as_default():
      init = constant_op.constant(100.0)
      var = variables.Variable(init)
      gradient = gradients.gradients(var.read_value(), var)
      self.assertIsNotNone(gradient)

  def testVariableAsGraphElementGradient(self):
    with ops.Graph().as_default() as graph:
      init = constant_op.constant(100.0)
      var = variables.Variable(init)
      gradient = gradients.gradients(graph.as_graph_element(var), var)
      self.assertIsNotNone(gradient)

  def testVariableRefGradient(self):
    with ops.Graph().as_default():
      init = constant_op.constant(100.0)
      var = variables.Variable(init)
      gradient = gradients.gradients(var._ref(), var)
      self.assertIsNotNone(gradient)

  def testDependentYs(self):
    with self.cached_session():
      x = constant_op.constant(3.0)
      y = math_ops.square(x)
      y1 = math_ops.square(y)
      y2 = math_ops.square(y1)
      g = gradients.gradients([y, y2], x)
      self.assertAllClose(17502.0, g[0].eval())
      g = gradients.gradients(y + y2, x)
      self.assertAllClose(17502.0, g[0].eval())
      z = array_ops.identity(y)
      z2 = array_ops.identity(y2)
      g = gradients.gradients([z, z2], x)
      self.assertAllClose(17502.0, g[0].eval())

  def testPartialDerivatives(self):
    with self.cached_session():
      x = constant_op.constant(1.)
      y = 2 * x
      z = x + y
      totalg = gradients.gradients(z, [x, y])
      self.assertEqual([3.0, 1.0], [g.eval() for g in totalg])
      partialg = gradients.gradients(z, [x, y], stop_gradients=[x, y])
      self.assertEqual([1.0, 1.0], [g.eval() for g in partialg])

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
    return function.Defun(dtypes.float32, dtypes.float32, **
                          kwargs)(cls.XSquarePlusB)

  def _GetFuncGradients(self, f, x_value, b_value):
    x = constant_op.constant(x_value, name="x")
    b = constant_op.constant(b_value, name="b")

    y = f(x, b)
    grads = gradients.gradients(y, [x, b])
    with self.cached_session() as sess:
      return sess.run(grads)

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

      with self.cached_session() as sess:
        self.assertAllEqual([40.0], sess.run(grads)[0])
        self.assertAllEqual([10.0], sess.run(grads)[1])

  def testFunctionGradientsWithGradFunc(self):
    g = ops.Graph()
    with g.as_default():
      grad_func = function.Defun(dtypes.float32, dtypes.float32,
                                 dtypes.float32)(self.XSquarePlusBGradient)
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
      grad_func = function.Defun(dtypes.float32, dtypes.float32,
                                 dtypes.float32)(self.XSquarePlusBGradient)
      with self.assertRaisesRegexp(ValueError, "Gradient defined twice"):
        f = self._GetFunc(
            grad_func=grad_func, python_grad_func=self._PythonGradient)
        f.add_to_graph(ops.Graph())

  def testGradientWrtCaptured(self):
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")

      @function.Defun()
      def Foo():
        y = math_ops.multiply(x, 2.0, name="y")
        g = gradients_impl.gradients(y, x)
        return g[0]

      f = Foo()
      with self.cached_session() as sess:
        self.assertEqual(sess.run(f), 2.0)

  def testGradientOfCaptured(self):
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")
      y = math_ops.multiply(x, 2.0, name="y")

      @function.Defun()
      def Foo():
        g = gradients_impl.gradients(y, x)
        return g[0]

      f = Foo()
      with self.cached_session() as sess:
        self.assertEqual(sess.run(f), 2.0)

  def testCapturedResourceVariable(self):
    with ops.Graph().as_default():
      var = resource_variable_ops.ResourceVariable(1.0, name="var")

      @function.Defun()
      def Foo():
        y = math_ops.multiply(var, 2.0, name="y")
        g = gradients_impl.gradients(y, var)
        return g[0]

      f = Foo()
      with self.cached_session() as sess:
        sess.run(variables.global_variables_initializer())
        self.assertEqual(sess.run(f), 2.0)

  def testCapturedNested(self):
    with ops.Graph().as_default():
      x1 = constant_op.constant(1.0, name="x1")
      x2 = constant_op.constant(2.0, name="x2")
      x3 = math_ops.multiply(x1, x2, name="x3")

      @function.Defun()
      def Outer():
        outer1 = array_ops.identity(x1, name="outer1")

        @function.Defun()
        def Inner():
          inner1 = array_ops.identity(outer1, name="inner1")
          inner2 = array_ops.identity(x2, name="inner2")
          inner3 = array_ops.identity(x3, name="inner3")
          return gradients_impl.gradients([inner1, inner2, inner3, x1],
                                          [x1, x2])

        return Inner()

      x1_grad, x2_grad = Outer()
      with self.cached_session() as sess:
        # 1.0 + None + 2.0 + 1.0 = 4.0
        self.assertEqual(sess.run(x1_grad), 4.0)
        # None + 1.0 + 1.0 + None = 2.0
        self.assertEqual(sess.run(x2_grad), 2.0)

  def testCapturedFromFunction(self):
    with ops.Graph().as_default():
      x = constant_op.constant(1.0, name="x")

      @function.Defun()
      def Outer():
        y = math_ops.multiply(x, 2.0, name="y")

        @function.Defun()
        def Inner():
          z = math_ops.multiply(y, 3.0, name="z")
          g = gradients_impl.gradients(z, y)
          return g[0]

        return Inner()

      z_grad = Outer()
      with self.cached_session() as sess:
        self.assertEqual(sess.run(z_grad), 3.0)


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
      with self.assertRaisesRegexp(LookupError, "explicitly disabled"):
        _ = gradients.gradients(out, inp)


class HessianVectorProductTest(test_util.TensorFlowTestCase):

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
      with self.test_session(use_gpu=use_gpu):
        mat = constant_op.constant(mat_value)
        v = constant_op.constant(v_value)
        x = constant_op.constant(x_value)
        mat_x = math_ops.matmul(mat, x, name="Ax")
        x_mat_x = math_ops.matmul(array_ops.transpose(x), mat_x, name="xAx")
        hess_v = gradients_impl._hessian_vector_product(x_mat_x, [x], [v])[0]
        hess_v_actual = hess_v.eval()
      self.assertAllClose(hess_v_value, hess_v_actual)


class HessianTest(test_util.TensorFlowTestCase):

  def testHessian1D(self):
    # Manually compute the Hessian explicitly for a low-dimensional problem
    # and check that `hessian` matches. Specifically, the Hessian of
    # f(x) = x^T A x is H = A + A^T.
    m = 4
    rng = np.random.RandomState([1, 2, 3])
    mat_value = rng.randn(m, m).astype("float32")
    x_value = rng.randn(m).astype("float32")
    hess_value = mat_value + mat_value.T
    with self.test_session(use_gpu=True):
      mat = constant_op.constant(mat_value)
      x = constant_op.constant(x_value)
      x_mat_x = math_ops.reduce_sum(x[:, None] * mat * x[None, :])
      hess = gradients.hessians(x_mat_x, x)[0]
      hess_actual = hess.eval()
    self.assertAllClose(hess_value, hess_actual)

  def testHessian1D_multi(self):
    # Test the computation of the hessian with respect to multiple tensors
    m = 4
    n = 3
    rng = np.random.RandomState([1, 2, 3])
    mat_values = [rng.randn(m, m).astype("float32") for _ in range(n)]
    x_values = [rng.randn(m).astype("float32") for _ in range(n)]
    hess_values = [mat_value + mat_value.T for mat_value in mat_values]
    with self.test_session(use_gpu=True):
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

  def testHessianInvalidDimension(self):
    for shape in [(10, 10), None]:
      with self.test_session(use_gpu=True):
        x = array_ops.placeholder(dtypes.float32, shape)
        # Expect a ValueError because the dimensions are wrong
        with self.assertRaises(ValueError):
          gradients.hessians(x, x)

  def testHessian2D_square_matrix(self):
    # Manually compute the Hessian explicitly for a low-dimensional problem
    # and check that `hessian` matches. Specifically, the Hessian of
    # f(x) = 1/2 * x^T * x is H = constant (block identity matrix)
    m = 3
    rng = np.random.RandomState([1, 2, 3])
    x_value = rng.randn(m, m).astype("float32")
    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_value)
      x_square = math_ops.reduce_sum(
          math_ops.matmul(array_ops.transpose(x), x) * 0.5
      )
      hess = gradients.hessians(x_square, x)[0]
      hess_actual = hess.eval()
    hess_value = np.bmat([
        [elem*np.ones((m, m)) for elem in vec]
        for vec in np.eye(m)
    ]).astype("float32")
    self.assertAllEqual((m, m, m, m), hess_actual.shape)
    self.assertAllClose(hess_value, hess_actual.reshape((m * m, m * m)))

  def testHessian2D_non_square_matrix(self):
    m = 3
    n = 4
    rng = np.random.RandomState([1, 2, 3])
    x_value = rng.randn(m, n).astype("float32")
    with self.test_session(use_gpu=True):
      x = constant_op.constant(x_value)
      x_square = math_ops.reduce_sum(
          math_ops.matmul(array_ops.transpose(x), x) * 0.5
      )
      hess = gradients.hessians(x_square, x)[0]
      hess_actual = hess.eval()
    hess_value = np.bmat([
        [elem*np.ones((n, n)) for elem in vec]
        for vec in np.eye(m)
    ]).astype("float32")
    self.assertAllEqual((m, n, m, n), hess_actual.shape)
    self.assertAllClose(hess_value, hess_actual.reshape((m * n, m * n)))


class IndexedSlicesToTensorTest(test_util.TensorFlowTestCase):

  def testIndexedSlicesToTensor(self):
    with self.cached_session():
      np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
      c = constant_op.constant(np_val)
      c_sparse = math_ops._as_indexed_slices(c)
      self.assertAllEqual(np_val.shape, c_sparse.dense_shape.eval())
      c_dense = math_ops.multiply(c_sparse, 1.0)
      self.assertAllClose(np_val, c_dense.eval())

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
      packed_dense = array_ops.stack(dense_list)
      packed_sparse = array_ops.stack(sparse_list)
      self.assertAllClose(packed_dense.eval(), packed_sparse.eval())

  def testInt64Indices(self):
    with self.cached_session():
      np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
      c = constant_op.constant(np_val)
      c_sparse = math_ops._as_indexed_slices(c)
      c_sparse = ops.IndexedSlices(
          c_sparse.values,
          math_ops.cast(c_sparse.indices, dtypes.int64), c_sparse.dense_shape)
      self.assertAllEqual(np_val.shape, c_sparse.dense_shape.eval())
      c_dense = math_ops.multiply(c_sparse, 1.0)
      self.assertAllClose(np_val, c_dense.eval())

  def testWarnings(self):
    # TODO(gunan) Reenable after this issue is fixed:
    # https://github.com/google/protobuf/issues/2812
    if sys.version_info >= (3, 5):
      self.skipTest("Skipped test for Python 3.5+")

    # Smaller than the threshold: no warning.
    c_sparse = ops.IndexedSlices(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32), constant([4, 4, 4, 4]))
    with warnings.catch_warnings(record=True) as w:
      math_ops.multiply(c_sparse, 1.0)
    self.assertEqual(0, len(w))

    # Greater than or equal to the threshold: warning.
    c_sparse = ops.IndexedSlices(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32), constant([100, 100, 100, 100]))
    # "always" filter prevents the warning from being suppressed if it was
    # already triggered in a different test.
    warnings.simplefilter("always")
    with warnings.catch_warnings(record=True) as w:
      math_ops.multiply(c_sparse, 1.0)
    self.assertEqual(1, len(w))
    self.assertTrue(
        "with 100000000 elements. This may consume a large amount of memory." in
        str(w[0].message))

    # Unknown dense shape: warning.
    c_sparse = ops.IndexedSlices(
        array_ops.placeholder(dtypes.float32),
        array_ops.placeholder(dtypes.int32),
        array_ops.placeholder(dtypes.int32))
    with warnings.catch_warnings(record=True) as w:
      math_ops.multiply(c_sparse, 1.0)
    self.assertEqual(1, len(w))
    self.assertTrue(
        "of unknown shape. This may consume a large amount of memory." in
        str(w[0].message))


class OnlyRealGradientsTest(test_util.TensorFlowTestCase):

  def testRealOnly(self):
    x = constant_op.constant(7+3j, dtype=dtypes.complex64)
    y = math_ops.square(x)
    with self.assertRaisesRegexp(
        TypeError,
        r"Gradients of complex tensors must set grad_ys "
        r"\(y\.dtype = tf\.complex64\)"):
      gradients.gradients(y, x)


class ResourceCondTest(test_util.TensorFlowTestCase):

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
    self.assertTrue(None not in grads)


class CustomGradientTest(test_util.TensorFlowTestCase):

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
        self.assertEqual(9., dy.eval())

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
      with session.Session() as sess:
        self.assertAllEqual([3., 5.], sess.run(dy))

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
        self.assertEqual(1, len(variables))
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
        self.assertTrue(g is not None)
      with session.Session() as sess:
        sess.run(variables.global_variables_initializer())
        dw = sess.run(math_ops.reduce_sum(grads[1]))
        self.assertEqual(12., dw)

  def testCustomGradientWithVariablesEager(self):
    with context.eager_mode():
      layer = core_layers.Dense(4, use_bias=False)

      @custom_gradient.custom_gradient
      def F(x):
        out = layer(x)

        def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
          del out_grad
          self.assertEqual(1, len(variables))
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

  def testCustomGradientErrorsWithNonResourceVariables(self):

    def F(x, use_resource=False):
      with variable_scope.variable_scope("f", use_resource=use_resource):
        out = core_layers.dense(x, 4, use_bias=False)

      def Grad(out_grad, variables=None):  # pylint: disable=redefined-outer-name
        del out_grad
        self.assertEqual(1, len(variables))
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

  def testRVGradientsDynamicCond(self):
    with self.cached_session():
      alpha = resource_variable_ops.ResourceVariable(
          np.random.random((1,)),
          dtype="float32")

      conditional = array_ops.placeholder_with_default(True, shape=())
      output = control_flow_ops.cond(
          conditional, lambda: alpha * 2, lambda: alpha * 3)

      g, = gradients_impl.gradients(output, alpha)
      variables.global_variables_initializer().run()
      self.assertAllEqual(g.eval(), [2.0])
      self.assertAllEqual(g.eval(feed_dict={conditional: False}), [3.0])


class AggregateIndexedSlicesGradientsTest(test_util.TensorFlowTestCase):

  def _assert_indexed_slices_equal(self, left, right):
    self.assertAllEqual(
        self.evaluate(ops.convert_to_tensor(left)),
        self.evaluate(ops.convert_to_tensor(right)))

  def testNoGradients(self):
    self.assertIsNone(gradients_impl._AggregateIndexedSlicesGradients([]))

  def testOneGradient(self):
    t = math_ops._as_indexed_slices(constant_op.constant(
        [[1., 2.], [0, 0], [3., 4.]]))
    result = gradients_impl._AggregateIndexedSlicesGradients([t])
    self._assert_indexed_slices_equal(t, result)

  def testMultipleGradients(self):
    t0 = math_ops._as_indexed_slices(constant_op.constant(
        [[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(constant_op.constant(
        [[0., 0.], [5, 6], [7., 8.]]))
    total = constant_op.constant(
        [[1., 2.], [5, 6], [10., 12.]])
    result = gradients_impl._AggregateIndexedSlicesGradients([t0, t1])
    self._assert_indexed_slices_equal(total, result)

  def testMultipleGradientsWithNones(self):
    t0 = math_ops._as_indexed_slices(constant_op.constant(
        [[1., 2.], [0, 0], [3., 4.]]))
    t1 = math_ops._as_indexed_slices(constant_op.constant(
        [[0., 0.], [5, 6], [7., 8.]]))
    t3 = None
    total = constant_op.constant(
        [[1., 2.], [5, 6], [10., 12.]])
    result = gradients_impl._AggregateIndexedSlicesGradients([t0, t1, t3])
    self._assert_indexed_slices_equal(total, result)

  def testMixedTensorAndIndexedSlices(self):
    t0 = math_ops._as_indexed_slices(constant_op.constant(
        [[1., 2.], [0, 0], [3., 4.]]))
    t1 = constant_op.constant(
        [[0., 0.], [5, 6], [7., 8.]])
    total = constant_op.constant(
        [[1., 2.], [5, 6], [10., 12.]])
    result = gradients_impl._AggregateIndexedSlicesGradients([t0, t1])
    self._assert_indexed_slices_equal(total, result)


if __name__ == "__main__":
  googletest.main()
