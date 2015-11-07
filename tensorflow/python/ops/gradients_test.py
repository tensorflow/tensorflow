"""Tests for tensorflow.ops.gradients."""
import warnings

import tensorflow.python.platform

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import types
# pylint: disable=unused-import
from tensorflow.python.ops import array_grad
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import data_flow_grad
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_grad
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import state_grad
# pylint: enable=unused-import
from tensorflow.python.ops.constant_op import constant
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
  gradients._MarkReachedOps(from_ops, reached_ops)
  between_ops = gradients._GatherInputs(to_ops, reached_ops)
  between_ops.sort(lambda x, y: y._id - x._id)
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
      t3 = array_ops.pack([t1, t2])
    # Full graph
    self._assertOpListEqual([t3.op, t2.op, t1.op],
                            _OpsBetween(g, [t3.op], [t1.op, t2.op]))
    # Only t1, t3.
    self._assertOpListEqual([t3.op, t1.op],
                            _OpsBetween(g, [t3.op], [t1.op]))

  def testOpsBetweenUnreachable(self):
    with ops.Graph().as_default() as g:
      t1 = constant(1.0)
      t2 = constant(2.0)
      _ = array_ops.pack([t1, t2])
      t4 = constant(1.0)
      t5 = constant(2.0)
      t6 = array_ops.pack([t4, t5])
    # Elements of to_ops are always listed.
    self._assertOpListEqual([t6.op], _OpsBetween(g, [t6.op], [t1.op]))

  def testOpsBetweenCut(self):
    with ops.Graph().as_default() as g:
      t1 = constant(1.0)
      t2 = constant(2.0)
      t3 = array_ops.pack([t1, t2])
      t4 = constant([1.0])
      t5 = array_ops.concat(0, [t4, t3])
      t6 = constant([2.0])
      t7 = array_ops.concat(0, [t5, t6])
    self._assertOpListEqual([t7.op, t5.op, t4.op],
                            _OpsBetween(g, [t7.op], [t4.op]))

  def testOpsBetweenCycle(self):
    with ops.Graph().as_default() as g:
      t1 = constant(1.0)
      t2 = constant(2.0)
      t3 = array_ops.pack([t1, t2])
      t4 = array_ops.concat(0, [t3, t3, t3])
      t5 = constant([1.0])
      t6 = array_ops.concat(0, [t4, t5])
      t7 = array_ops.concat(0, [t6, t3])
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
      split_wx = array_ops.split(0, 2, wx)
      c = math_ops.reduce_sum(split_wx[1])
      gw = gradients.gradients(c, [w])[0]
    self.assertEquals("MatMul", gw.op.type)

  def testColocateGradients(self):
    with ops.Graph().as_default() as g:
      w = constant(1.0, shape=[1, 1])
      x = constant(1.0, shape=[1, 2])
      with g.device("/gpu:0"):
        wx = math_ops.matmul(w, x)
      gw = gradients.gradients(wx, [w], colocate_gradients_with_ops=True)[0]
    self.assertEquals("/gpu:0", gw.device)

  def testColocateGradientsWithAggregation(self):
    with ops.Graph().as_default() as g:
      with g.device("/gpu:1"):
        w = constant(1.0, shape=[1, 1])
      x = constant(1.0, shape=[1, 2])
      y = constant(1.0, shape=[1, 2])
      wx = math_ops.matmul(w, x)
      wy = math_ops.matmul(w, y)
      with g.device("/gpu:0"):
        z = wx + wy
      gw1 = gradients.gradients(z, [w], colocate_gradients_with_ops=True)[0]
      self.assertEquals("/gpu:1", gw1.device)
      gw2 = gradients.gradients(z, [w], colocate_gradients_with_ops=False)[0]
      self.assertEquals(None, gw2.device)

  def testBoundaryStop(self):
    # Test that we don't differentiate 'x'. The gradient function for 'x' is
    # set explicitly to None so we will get an exception if the gradient code
    # tries to differentiate 'x'.
    with ops.Graph().as_default() as g:
      c = constant(1.0)
      x = array_ops.identity(c)
      y = x + 1.0
      z = y + 1
      grads = gradients.gradients(z, [x])
      self.assertTrue(all([x for x in grads]))

  def testBoundaryContinue(self):
    # Test that we differentiate both 'x' and 'y' correctly when x is a
    # predecessor of y.
    with self.test_session():
      x = constant(1.0)
      y = x * 2.0
      z = y * 3.0
      grads = gradients.gradients(z, [x, y])
      self.assertTrue(all([x for x in grads]))
      self.assertEqual(6.0, grads[0].eval())

  def testAggregationMethodAccumulateN(self):
    with self.test_session():
      x = constant(1.0)
      y = x * 2.0
      z = y + y + y + y + y + y + y + y + y + y
      grads = gradients.gradients(
          z,
          [x, y],
          aggregation_method=
          gradients.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
      self.assertTrue(all([x for x in grads]))
      self.assertEqual(20.0, grads[0].eval())
      self.assertEqual(10.0, grads[1].eval())

  def testAggregationMethodAddN(self):
    with self.test_session():
      x = constant(1.0)
      y = x * 2.0
      z = y + y + y + y + y + y + y + y + y + y
      grads = gradients.gradients(
          z,
          [x, y],
          aggregation_method=gradients.AggregationMethod.ADD_N)
      self.assertTrue(all([x for x in grads]))
      self.assertEqual(20.0, grads[0].eval())
      self.assertEqual(10.0, grads[1].eval())

  def testAggregationMethodTree(self):
    with self.test_session():
      x = constant(1.0)
      y = x * 2.0
      z = y + y + y + y + y + y + y + y + y + y
      grads = gradients.gradients(
          z,
          [x, y],
          aggregation_method=gradients.AggregationMethod.EXPERIMENTAL_TREE)
      self.assertTrue(all([x for x in grads]))
      self.assertEqual(20.0, grads[0].eval())
      self.assertEqual(10.0, grads[1].eval())

  def testNoGradientForStringOutputs(self):
    with ops.Graph().as_default() as g:
      @ops.RegisterGradient("TestOp")
      def _TestOpGrad(op, float_grad, string_grad):
        """Gradient function for TestOp."""
        self.assertEquals(float_grad.dtype, types.float32)
        self.assertFalse(string_grad)
        return float_grad
      ops.RegisterShape("TestOp")(None)

      c = constant(1.0)
      x, y = g.create_op("TestOp", [c], [types.float32, types.string]).outputs
      z = x * 2.0
      w = z * 3.0
      grads = gradients.gradients(z, [c])
      self.assertTrue(isinstance(grads[0], ops.Tensor))


class StopGradientTest(test_util.TensorFlowTestCase):

  def testStopGradient(self):
    with ops.Graph().as_default():
      inp = constant(1.0, shape=[100, 32], name="in")
      out = array_ops.stop_gradient(inp)
      igrad = gradients.gradients(out, inp)[0]
    assert igrad is None


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
        hess_v = gradients._hessian_vector_product(x_mat_x, [x], [v])[0]
        hess_v_actual = hess_v.eval()
      self.assertAllClose(hess_v_value, hess_v_actual)


class IndexedSlicesToTensorTest(test_util.TensorFlowTestCase):

  def testIndexedSlicesToTensor(self):
    with self.test_session():
      np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
      c = constant_op.constant(np_val)
      c_sparse = math_ops._as_indexed_slices(c)
      self.assertAllEqual(np_val.shape, c_sparse.dense_shape.eval())
      c_dense = math_ops.mul(c_sparse, 1.0)
      self.assertAllClose(np_val, c_dense.eval())

  def testInt64Indices(self):
    with self.test_session():
      np_val = np.random.rand(4, 4, 4, 4).astype(np.float32)
      c = constant_op.constant(np_val)
      c_sparse = math_ops._as_indexed_slices(c)
      c_sparse = ops.IndexedSlices(
          c_sparse.values, math_ops.cast(c_sparse.indices, types.int64),
          c_sparse.dense_shape)
      self.assertAllEqual(np_val.shape, c_sparse.dense_shape.eval())
      c_dense = math_ops.mul(c_sparse, 1.0)
      self.assertAllClose(np_val, c_dense.eval())

  def testWarnings(self):
    # Smaller than the threshold: no warning.
    c_sparse = ops.IndexedSlices(array_ops.placeholder(types.float32),
                                 array_ops.placeholder(types.int32),
                                 constant([4, 4, 4, 4]))
    with warnings.catch_warnings(record=True) as w:
      math_ops.mul(c_sparse, 1.0)
    self.assertEqual(0, len(w))

    # Greater than or equal to the threshold: warning.
    c_sparse = ops.IndexedSlices(array_ops.placeholder(types.float32),
                                 array_ops.placeholder(types.int32),
                                 constant([100, 100, 100, 100]))
    with warnings.catch_warnings(record=True) as w:
      math_ops.mul(c_sparse, 1.0)
    self.assertEqual(1, len(w))
    self.assertTrue(
        "with 100000000 elements. This may consume a large amount of memory."
        in str(w[0].message))

    # Unknown dense shape: warning.
    c_sparse = ops.IndexedSlices(array_ops.placeholder(types.float32),
                                 array_ops.placeholder(types.int32),
                                 array_ops.placeholder(types.int32))
    with warnings.catch_warnings(record=True) as w:
      math_ops.mul(c_sparse, 1.0)
    self.assertEqual(1, len(w))
    self.assertTrue(
        "of unknown shape. This may consume a large amount of memory."
        in str(w[0].message))


if __name__ == "__main__":
  googletest.main()
