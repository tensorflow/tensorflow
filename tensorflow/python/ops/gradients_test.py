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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.constant_op import constant
from tensorflow.python.ops import array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_grad  # pylint: disable=unused-import
from tensorflow.python.ops import data_flow_ops  # pylint: disable=unused-import
from tensorflow.python.ops import functional_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gradients
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import state_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.nn_ops import bias_add
from tensorflow.python.platform import googletest


def _OpsBetween(graph, to_ops, from_ops):
  """Build the list of operations between two lists of Operations.

  Args:
    graph: a Graph.
    to_ops: list of Operations.
    from_ops: list of Operations.

  Returns:
    The list of operations between "from_ops" and "to_ops", sorted by
    decreasing operation id. This list contains all elements of to_ops.

    TODO(touts): Think about returning an empty list if from_ops are not
    reachable from to_ops.  Presently it returns to_ops in that case.
  """
  # List of booleans, indexed by operation id, indicating if
  # an op is reached from the output of "input_ops".
  reached_ops = [False] * (graph._last_id + 1)
  # We only care to reach up to "output_ops" so we mark the
  # output ops as reached to avoid recursing past them.
  for op in to_ops:
    reached_ops[op._id] = True
  gradients_impl._MarkReachedOps(from_ops, reached_ops)
  between_ops = gradients_impl._GatherInputs(to_ops, reached_ops)
  between_ops.sort(key=lambda x: -x._id)
  return between_ops


class GradientsTest(test_util.TensorFlowTestCase):

  def _OpNames(self, op_list):
    return ["%s/%d" % (str(op.name), op._id) for op in op_list]

  def _assertOpListEqual(self, ops1, ops2):
    self.assertEquals(self._OpNames(ops1), self._OpNames(ops2))

  def testOpsBetweenSimple(self):
    with ops.Graph().as_default() as g:
      t1 = constant(1.0)
      t2 = constant(2.0)
      t3 = array_ops.stack([t1, t2])
    # Full graph
    self._assertOpListEqual([t3.op, t2.op, t1.op],
                            _OpsBetween(g, [t3.op], [t1.op, t2.op]))
    # Only t1, t3.
    self._assertOpListEqual([t3.op, t1.op], _OpsBetween(g, [t3.op], [t1.op]))

  def testOpsBetweenUnreachable(self):
    with ops.Graph().as_default() as g:
      t1 = constant(1.0)
      t2 = constant(2.0)
      _ = array_ops.stack([t1, t2])
      t4 = constant(1.0)
      t5 = constant(2.0)
      t6 = array_ops.stack([t4, t5])
    # Elements of to_ops are always listed.
    self._assertOpListEqual([t6.op], _OpsBetween(g, [t6.op], [t1.op]))

  def testOpsBetweenCut(self):
    with ops.Graph().as_default() as g:
      t1 = constant(1.0)
      t2 = constant(2.0)
      t3 = array_ops.stack([t1, t2])
      t4 = constant([1.0])
      t5 = array_ops.concat([t4, t3], 0)
      t6 = constant([2.0])
      t7 = array_ops.concat([t5, t6], 0)
    self._assertOpListEqual([t7.op, t5.op, t4.op],
                            _OpsBetween(g, [t7.op], [t4.op]))

  def testOpsBetweenCycle(self):
    with ops.Graph().as_default() as g:
      t1 = constant(1.0)
      t2 = constant(2.0)
      t3 = array_ops.stack([t1, t2])
      t4 = array_ops.concat([t3, t3, t3], 0)
      t5 = constant([1.0])
      t6 = array_ops.concat([t4, t5], 0)
      t7 = array_ops.concat([t6, t3], 0)
    self._assertOpListEqual([t6.op, t4.op, t3.op],
                            _OpsBetween(g, [t6.op], [t3.op]))
    self._assertOpListEqual([t7.op, t6.op, t5.op, t4.op, t3.op, t1.op],
                            _OpsBetween(g, [t7.op], [t1.op, t5.op]))
    self._assertOpListEqual([t6.op, t5.op, t4.op, t3.op, t2.op],
                            _OpsBetween(g, [t6.op], [t2.op, t5.op]))

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
    with self.test_session():
      x = constant(1.0)
      y = x * 2.0
      z = y * 3.0
      grads = gradients.gradients(z, [x, y])
      self.assertTrue(all(x is not None for x in grads))
      self.assertEqual(6.0, grads[0].eval())

  def testAggregationMethodAccumulateN(self):
    with self.test_session():
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
    with self.test_session():
      x = constant(1.0)
      y = x * 2.0
      z = y + y + y + y + y + y + y + y + y + y
      grads = gradients.gradients(
          z, [x, y], aggregation_method=gradients.AggregationMethod.ADD_N)
      self.assertTrue(all(x is not None for x in grads))
      self.assertEqual(20.0, grads[0].eval())
      self.assertEqual(10.0, grads[1].eval())

  def testAggregationMethodTree(self):
    with self.test_session():
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
      # The gradient of tf.identity should pass the value through unchanged.
      # A previous version of the code did this only for tf.Tensor, not
      # tf.IndexedSlices.
      self.assertEqual(dx, dy)

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
    with self.test_session():
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
    with self.test_session() as sess:
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

      with self.test_session() as sess:
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


class IndexedSlicesToTensorTest(test_util.TensorFlowTestCase):

  def testIndexedSlicesToTensor(self):
    with self.test_session():
      np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
      c = constant_op.constant(np_val)
      c_sparse = math_ops._as_indexed_slices(c)
      self.assertAllEqual(np_val.shape, c_sparse.dense_shape.eval())
      c_dense = math_ops.multiply(c_sparse, 1.0)
      self.assertAllClose(np_val, c_dense.eval())

  def testIndexedSlicesToTensorList(self):
    with self.test_session():
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
    with self.test_session():
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
    if sys.version_info >= (3, 6):
      self.skipTest("Skipped test for Python 3.6+")

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


if __name__ == "__main__":
  googletest.main()
