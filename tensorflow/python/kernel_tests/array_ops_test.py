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

"""Tests for array_ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops


class BatchMatrixTransposeTest(test_util.TensorFlowTestCase):

  def testNonBatchMatrix(self):
    matrix = [[1, 2, 3], [4, 5, 6]]  # Shape (2, 3)
    expected_transposed = [[1, 4], [2, 5], [3, 6]]  # Shape (3, 2)
    with self.test_session():
      transposed = tf.matrix_transpose(matrix)
      self.assertEqual((3, 2), transposed.get_shape())
      self.assertAllEqual(expected_transposed, transposed.eval())

  def testBatchMatrix(self):
    matrix_0 = [[1, 2, 3], [4, 5, 6]]
    matrix_0_t = [[1, 4], [2, 5], [3, 6]]
    matrix_1 = [[11, 22, 33], [44, 55, 66]]
    matrix_1_t = [[11, 44], [22, 55], [33, 66]]
    batch_matrix = [matrix_0, matrix_1]  # Shape (2, 2, 3)
    expected_transposed = [matrix_0_t, matrix_1_t]  # Shape (2, 3, 2)
    with self.test_session():
      transposed = tf.matrix_transpose(batch_matrix)
      self.assertEqual((2, 3, 2), transposed.get_shape())
      self.assertAllEqual(expected_transposed, transposed.eval())

  def testNonBatchMatrixDynamicallyDefined(self):
    matrix = [[1, 2, 3], [4, 5, 6]]  # Shape (2, 3)
    expected_transposed = [[1, 4], [2, 5], [3, 6]]  # Shape (3, 2)
    with self.test_session():
      matrix_ph = tf.placeholder(tf.int32)
      transposed = tf.matrix_transpose(matrix_ph)
      self.assertAllEqual(
          expected_transposed,
          transposed.eval(feed_dict={matrix_ph: matrix}))

  def testBatchMatrixDynamicallyDefined(self):
    matrix_0 = [[1, 2, 3], [4, 5, 6]]
    matrix_0_t = [[1, 4], [2, 5], [3, 6]]
    matrix_1 = [[11, 22, 33], [44, 55, 66]]
    matrix_1_t = [[11, 44], [22, 55], [33, 66]]
    batch_matrix = [matrix_0, matrix_1]  # Shape (2, 2, 3)
    expected_transposed = [matrix_0_t, matrix_1_t]  # Shape (2, 3, 2)
    with self.test_session():
      batch_matrix_ph = tf.placeholder(tf.int32)
      transposed = tf.matrix_transpose(batch_matrix_ph)
      self.assertAllEqual(
          expected_transposed,
          transposed.eval(feed_dict={batch_matrix_ph: batch_matrix}))

  def testTensorWithStaticRankLessThanTwoRaisesBecauseNotAMatrix(self):
    vector = [1, 2, 3]
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "should be a "):
        tf.matrix_transpose(vector)


class BooleanMaskTest(test_util.TensorFlowTestCase):

  def CheckVersusNumpy(self, ndims_mask, arr_shape, make_mask=None):
    """Check equivalence between boolean_mask and numpy masking."""
    if make_mask is None:
      make_mask = lambda shape: np.random.randint(0, 2, size=shape).astype(bool)
    arr = np.random.rand(*arr_shape)
    mask = make_mask(arr_shape[: ndims_mask])
    masked_arr = arr[mask]
    with self.test_session():
      masked_tensor = array_ops.boolean_mask(arr, mask)
      np.testing.assert_allclose(
          masked_arr,
          masked_tensor.eval(),
          err_msg="masked_arr:\n%s\n\nmasked_tensor:\n%s" % (
              masked_arr, masked_tensor.eval()))
      masked_tensor.get_shape().assert_is_compatible_with(masked_arr.shape)
      self.assertSequenceEqual(
          masked_tensor.get_shape()[1:].as_list(),
          masked_arr.shape[1:],
          msg="shape information lost %s -> %s" % (
              masked_arr.shape, masked_tensor.get_shape()))

  def testOneDimensionalMask(self):
    # Do 1d separately because it's the only easy one to debug!
    ndims_mask = 1
    for ndims_arr in range(ndims_mask, ndims_mask + 3):
      for _ in range(3):
        arr_shape = np.random.randint(1, 5, size=ndims_arr)
        self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testMultiDimensionalMask(self):
    for ndims_mask in range(1, 4):
      for ndims_arr in range(ndims_mask, ndims_mask + 3):
        for _ in range(3):
          arr_shape = np.random.randint(1, 5, size=ndims_arr)
          self.CheckVersusNumpy(ndims_mask, arr_shape)

  def testEmptyOutput(self):
    make_mask = lambda shape: np.zeros(shape, dtype=bool)
    for ndims_mask in range(1, 4):
      for ndims_arr in range(ndims_mask, ndims_mask + 3):
        for _ in range(3):
          arr_shape = np.random.randint(1, 5, size=ndims_arr)
          self.CheckVersusNumpy(ndims_mask, arr_shape, make_mask=make_mask)

  def testWorksWithDimensionsEqualToNoneDuringGraphBuild(self):
    # The rank of the mask tensor must be specified. This is explained
    # in the docstring as well.
    with self.test_session() as sess:
      ph_tensor = array_ops.placeholder(dtypes.int32, shape=None)
      ph_mask = array_ops.placeholder(dtypes.bool, shape=[None])

      arr = np.array([[1, 2], [3, 4]])
      mask = np.array([False, True])

      masked_tensor = sess.run(
          array_ops.boolean_mask(ph_tensor, ph_mask),
          feed_dict={ph_tensor: arr, ph_mask: mask})
      np.testing.assert_allclose(masked_tensor, arr[mask])

  def testMaskDimensionsSetToNoneRaises(self):
    # The rank of the mask tensor must be specified. This is explained
    # in the docstring as well.
    with self.test_session():
      tensor = array_ops.placeholder(dtypes.int32, shape=[None, 2])
      mask = array_ops.placeholder(dtypes.bool, shape=None)
      with self.assertRaisesRegexp(ValueError, "dimensions must be specified"):
        array_ops.boolean_mask(tensor, mask)

  def testMaskHasMoreDimsThanTensorRaises(self):
    mask = [[True, True], [False, False]]
    tensor = [1, 2, 3, 4]
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "incompatible"):
        array_ops.boolean_mask(tensor, mask).eval()

  def testMaskIsScalarRaises(self):
    mask = True
    tensor = 1
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "mask.*scalar"):
        array_ops.boolean_mask(tensor, mask).eval()

  def testMaskShapeDifferentThanFirstPartOfTensorShapeRaises(self):
    mask = [True, True, True]
    tensor = [[1, 2], [3, 4]]
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "incompatible"):
        array_ops.boolean_mask(tensor, mask).eval()


class OperatorShapeTest(test_util.TensorFlowTestCase):

  def testExpandScalar(self):
    scalar = "hello"
    scalar_expanded = array_ops.expand_dims(scalar, [0])
    self.assertEqual(scalar_expanded.get_shape(), (1,))

  def testSqueeze(self):
    scalar = "hello"
    scalar_squeezed = array_ops.squeeze(scalar, ())
    self.assertEqual(scalar_squeezed.get_shape(), ())


class ReverseTest(test_util.TensorFlowTestCase):

  def testReverse0DimAuto(self):
    x_np = 4
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = array_ops.reverse(x_np, []).eval()
        self.assertAllEqual(x_tf, x_np)

  def _reverse1DimAuto(self, np_dtype):
    x_np = np.array([1, 2, 3, 4, 5], dtype=np_dtype)

    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        x_tf = array_ops.reverse(x_np, [True]).eval()
        self.assertAllEqual(x_tf, np.asarray(x_np)[::-1])

  def testReverse1DimAuto(self):
    for dtype in [np.uint8, np.int8, np.int32, np.int64, np.bool, np.float16,
                  np.float32, np.float64, np.complex64, np.complex128]:
      self._reverse1DimAuto(dtype)

  def testUnknownDims(self):
    data_t = tf.placeholder(tf.float32)
    dims_known_t = tf.placeholder(tf.bool, shape=[3])
    reverse_known_t = tf.reverse(data_t, dims_known_t)
    self.assertEqual(3, reverse_known_t.get_shape().ndims)

    dims_unknown_t = tf.placeholder(tf.bool)
    reverse_unknown_t = tf.reverse(data_t, dims_unknown_t)
    self.assertIs(None, reverse_unknown_t.get_shape().ndims)

    data_2d_t = tf.placeholder(tf.float32, shape=[None, None])
    dims_2d_t = tf.placeholder(tf.bool, shape=[2])
    reverse_2d_t = tf.reverse(data_2d_t, dims_2d_t)
    self.assertEqual(2, reverse_2d_t.get_shape().ndims)

    dims_3d_t = tf.placeholder(tf.bool, shape=[3])
    with self.assertRaisesRegexp(ValueError, "must be rank 3"):
      tf.reverse(data_2d_t, dims_3d_t)


class MeshgridTest(test_util.TensorFlowTestCase):

  def _compare(self, n, np_dtype, use_gpu):
    inputs = []
    for i in range(n):
      x = np.linspace(-10, 10, 5).astype(np_dtype)
      if np_dtype in (np.complex64, np.complex128):
        x += 1j
      inputs.append(x)

    numpy_out = np.meshgrid(*inputs)
    with self.test_session(use_gpu=use_gpu):
      tf_out = array_ops.meshgrid(*inputs)
      for X, _X in zip(numpy_out, tf_out):
        self.assertAllEqual(X, _X.eval())

  def testCompare(self):
    for t in (np.float16, np.float32, np.float64, np.int32, np.int64,
            np.complex64, np.complex128):
      # Don't test the one-dimensional case, as
      # old numpy versions don't support it
      self._compare(2, t, False)
      self._compare(3, t, False)
      self._compare(4, t, False)
      self._compare(5, t, False)

    # Test for inputs with rank not equal to 1
    x = [[1, 1], [1, 1]]
    with self.assertRaisesRegexp(errors.InvalidArgumentError,
                                 "needs to have rank 1"):
      with self.test_session():
        X, _ = array_ops.meshgrid(x, x)
        X.eval()


class StridedSliceChecker(object):
  """Check a given tensor against the numpy result."""

  REF_TENSOR = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
  REF_TENSOR_ALIGNED = np.arange(1, 97, dtype=np.float32).reshape(3, 4, 8)

  def __init__(self, test, x, tensor_type=tf.int32, check_type_infer=True):
    self.test = test
    self.x = tf.cast(tf.constant(x, dtype=tf.float32), dtype=tensor_type)
    self.x_np = np.array(x)
    self.check_type_infer = check_type_infer

  def __getitem__(self, spec):
    op = self.x.__getitem__(spec)
    if not isinstance(spec, (list, tuple)):
      spec = [spec]

    tensor = op.eval()

    # Make a numpy spec that pre-evals the tensors
    np_specs = []

    def eval_if_tensor(x):
      try:
        return x.eval()
      except AttributeError:
        return x

    for s in spec:
      if isinstance(s, slice):
        start = eval_if_tensor(s.start)
        stop = eval_if_tensor(s.stop)
        step = eval_if_tensor(s.step)
        np_specs.append(slice(start, stop, step))
      else:
        np_specs.append(eval_if_tensor(s))

    self.test.assertAllEqual(self.x_np[tuple(np_specs)], tensor)
    if self.check_type_infer:
      self.test.assertAllEqual(tensor.shape, op.get_shape())
    return tensor


class StridedSliceTest(test_util.TensorFlowTestCase):
  """Test the strided slice operation with variants of slices."""

  def test_basic_slice(self):
    for tensor_type in [tf.int32, tf.int64, tf.int16, tf.int8, tf.float32,
                        tf.float64]:
      for use_gpu in [False, True]:
        with self.test_session(use_gpu=use_gpu):
          checker = StridedSliceChecker(
              self, StridedSliceChecker.REF_TENSOR, tensor_type=tensor_type)
          _ = checker[:, :, :]
          # Various ways of representing identity slice
          _ = checker[:, :, :]
          _ = checker[::, ::, ::]
          _ = checker[::1, ::1, ::1]
          # Not zero slice
          _ = checker[::1, ::5, ::2]
          # Reverse in each dimension independently
          _ = checker[::-1, :, :]
          _ = checker[:, ::-1, :]
          _ = checker[:, :, ::-1]
          ## negative index tests i.e. n-2 in first component
          _ = checker[-2::-1, :, ::1]
          # negative index tests i.e. n-2 in first component, non-unit stride
          _ = checker[-2::-1, :, ::2]

          # Check rank-0 examples
          checker2 = StridedSliceChecker(self, 5, tensor_type=tf.int32)
          _ = checker2[None]
          _ = checker2[...]
          _ = checker2[tuple()]

  def testDegenerateSlices(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        checker = StridedSliceChecker(self, StridedSliceChecker.REF_TENSOR)
        # degenerate by offering a forward interval with a negative stride
        _ = checker[0:-1:-1, :, :]
        # degenerate with a reverse interval with a positive stride
        _ = checker[-1:0, :, :]
        # empty interval in every dimension
        _ = checker[-1:0, 2:2, 2:3:-1]

  def testEllipsis(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        raw = [[[[[1, 2], [3, 4], [5, 6]]], [[[7, 8], [9, 10], [11, 12]]]]]
        checker = StridedSliceChecker(self, raw)

        _ = checker[0:]
        # implicit ellipsis
        _ = checker[0:, ...]
        # ellipsis alone
        _ = checker[...]
        # ellipsis at end
        _ = checker[0:1, ...]
        # ellipsis at begin
        _ = checker[..., 0:1]
        # ellipsis at middle
        _ = checker[0:1, ..., 0:1]
        # multiple ellipses not allowed
        with self.assertRaisesRegexp(ValueError, "Multiple ellipses"):
          _ = checker[..., :, ...].eval()

  def testShrink(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
                [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
        checker = StridedSliceChecker(self, raw)
        _ = checker[:, :, :, :, 3]
        _ = checker[..., 3]
        _ = checker[:, 0]
        _ = checker[:, :, 0]

  def testTensorIndexing(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
                [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
        checker = StridedSliceChecker(self, raw, check_type_infer=False)
        bar = tf.constant(2)
        bar2 = tf.constant(3)
        _ = checker[..., bar:bar2]
        _ = checker[..., bar]
        with self.assertRaisesRegexp(TypeError,
                                     "DataType float32 for attr 'Index'"):
          _ = checker[..., 3.0]
        _ = checker[..., 3]

  def testExpand(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        raw = [[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
                [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]]
        checker = StridedSliceChecker(self, raw)
        # new axis (followed by implicit ellipsis)
        _ = checker[np.newaxis]
        # newaxis after ellipsis
        _ = checker[..., np.newaxis]
        # newaxis in between ellipsis and explicit range
        _ = checker[..., np.newaxis, :]
        _ = checker[:, ..., np.newaxis, :, :]
        # Reverse final dimension with new axis
        _ = checker[:, :, np.newaxis, :, 2::-1]
        # Ellipsis in middle of two newaxis
        _ = checker[np.newaxis, ..., np.newaxis]

  def testExpandVariable(self):
    for use_gpu in False, True:
      with self.test_session(use_gpu=use_gpu):
        x = tf.Variable(7, dtype=tf.int32)
        x.initializer.run()
        y = x[None].eval()
        self.assertEqual(y.shape, (1,))
        self.assertAllEqual(y, (7,))

  def testOptimizedCases(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        checker = StridedSliceChecker(self,
                                      StridedSliceChecker.REF_TENSOR_ALIGNED)
        # Identity
        _ = checker[:]
        # Identity
        _ = checker[...]
        # Identity
        _ = checker[np.newaxis, ..., np.newaxis]
        # First axis slice
        _ = checker[1:]
        # First axis slice
        _ = checker[np.newaxis, 1:]


class StridedSliceShapeChecker(object):

  def __init__(self, x):
    self.x = x

  def __getitem__(self, spec):
    op = self.x.__getitem__(spec)
    return op.get_shape()


class StridedSliceShapeTest(test_util.TensorFlowTestCase):
  """Test the shape inference of StridedSliceShapes."""

  def testUnknown(self):
    with self.test_session(use_gpu=False):
      uncertain_tensor = tf.placeholder(tf.float32)
      a = StridedSliceShapeChecker(uncertain_tensor)
      a_slice_shape = a[...]
      self.assertAllEqual(a_slice_shape.ndims, None)

  def tensorShapeEqual(self, x, y):
    self.assertTrue(x is not None and y is not None or x is None and y is None)
    self.assertEqual(x.as_list(), y.as_list())

  def testTensorShapeUncertain(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu):
        uncertain_tensor = tf.placeholder(tf.float32, shape=(5, None, 7))
        a = StridedSliceShapeChecker(uncertain_tensor)
        self.tensorShapeEqual(a[3:5], tensor_shape.TensorShape([2, None, 7]))
        self.tensorShapeEqual(a[3:5, :, 4], tensor_shape.TensorShape([2, None]))
        self.tensorShapeEqual(a[3:5, 3:4, 4],
                              tensor_shape.TensorShape([2, None]))
        self.tensorShapeEqual(a[3:5, :, 5:10],
                              tensor_shape.TensorShape([2, None, 2]))
        self.tensorShapeEqual(a[3:5, :, 50:3],
                              tensor_shape.TensorShape([2, None, 0]))
        self.tensorShapeEqual(a[3:5, :, tf.newaxis, 50:3,],
                              tensor_shape.TensorShape([2, None, 1, 0]))
        self.tensorShapeEqual(a[1:5:2, :, tf.newaxis, 50:3,],
                              tensor_shape.TensorShape([2, None, 1, 0]))
        self.tensorShapeEqual(a[:5:3, :, tf.newaxis, 50:3,],
                              tensor_shape.TensorShape([2, None, 1, 0]))
        self.tensorShapeEqual(a[:2:3, :, tf.newaxis, 50:3,],
                              tensor_shape.TensorShape([1, None, 1, 0]))
        self.tensorShapeEqual(a[::-1, :, tf.newaxis, ::-2],
                              tensor_shape.TensorShape([5, None, 1, 4]))


class GradSliceChecker(object):
  """Tests that we can compute a gradient for var^2."""

  def __init__(self, test, sess, var, varnp):
    self.test = test
    self.sess = sess
    self.val = var * var
    self.var = var
    self.varnp = varnp

  def __getitem__(self, spec):
    slice_var = self.var[spec]
    slice_val = self.val[spec]

    # compute analytic 2nd derivative
    analytic_grad2 = 2 * slice_val

    dy = tf.Variable(tf.ones(shape=slice_var.get_shape(), dtype=tf.int32))
    assign = dy.assign(slice_var)
    slice_val_grad, = tf.gradients(slice_val, self.var, grad_ys=dy)
    slice_val_grad2, = tf.gradients(slice_val_grad, dy, grad_ys=self.var)
    self.sess.run(assign)
    slice_val_grad_evaled, slice_val_grad2_evaled = (
        self.sess.run([slice_val_grad, slice_val_grad2]))
    analytic_grad2_evaled = analytic_grad2.eval()
    self.test.assertAllEqual(slice_val_grad2_evaled, analytic_grad2_evaled)

    # compute analytic gradient for slice
    np_val_grad = (2 * self.varnp * self.varnp)
    np_sliceval_grad = np.zeros(self.var.get_shape())
    np_sliceval_grad[spec] = np_val_grad[spec]
    # verify gradient
    self.test.assertAllEqual(slice_val_grad_evaled, np_sliceval_grad)


class StridedSliceGradTest(test_util.TensorFlowTestCase):
  """Test that strided slice's custom gradient produces correct gradients."""

  def testGradient(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu) as sess:
        var = tf.Variable(tf.reshape(tf.range(1, 97, 1), shape=(6, 4, 4)))
        init = tf.initialize_all_variables()
        sess.run(init)

        grad = GradSliceChecker(self, sess, var,
                                np.array(range(1, 97, 1)).reshape((6, 4, 4)))
        _ = grad[2:6:2, 1:3, 1:3]
        _ = grad[3:0:-2, 1:3, 1:3]
        _ = grad[3:0:-2, tf.newaxis, 1:3, 2, tf.newaxis]
        _ = grad[3:0:-2, 1:3, 2]
        _ = grad[:, -1, :]
        _ = grad[:, -2, :]
        with self.assertRaisesRegexp(ValueError, "out of bounds"):
          _ = grad[:, -200, :]
        with self.assertRaisesRegexp(ValueError, "out of bounds"):
          _ = grad[:, 200, :]

  def testGradientZero(self):
    for use_gpu in [False, True]:
      with self.test_session(use_gpu=use_gpu) as sess:
        var = tf.Variable(8)
        init = tf.initialize_all_variables()
        sess.run(init)
        grad = GradSliceChecker(self, sess, var,
                                np.array(8))
        _ = grad[tuple()]


class StridedSliceGradTypeTest(test_util.TensorFlowTestCase):
  """Test varied index types and host located memory."""

  def testHostVsDevice(self):
    with self.test_session(use_gpu=True) as sess:
      var2 = tf.Variable(
          tf.reshape(
              tf.cast(tf.range(1, 5, 1), tf.float32), shape=(4, 1, 1)))
      varshape = tf.Variable([6, 4, 4], dtype=tf.int32)
      sess.run(tf.initialize_all_variables())
      begin = tf.constant([0, 0, 0])
      end = tf.constant([4, 1, 1])
      strides = tf.constant([1, 1, 1])
      foo = array_ops.strided_slice_grad(varshape, begin, end, strides, var2)
      sess.run(foo)

  def testInt64Shape(self):
    with self.test_session(use_gpu=True) as sess:
      original_dy = tf.reshape(
          tf.cast(tf.range(1, 5, 1), tf.float32), shape=(4, 1, 1))
      original_shape = tf.constant([6, 4, 4], dtype=tf.int64)
      sess.run(tf.initialize_all_variables())
      begin = tf.constant([0, 0, 0], dtype=tf.int64)
      end = tf.constant([4, 1, 1], dtype=tf.int64)
      strides = tf.constant([1, 1, 1], dtype=tf.int64)
      dx = array_ops.strided_slice_grad(original_shape, begin, end, strides,
                                        original_dy)
      sess.run(dx)

  def testMixedIndexTypes(self):
    with self.test_session(use_gpu=True) as sess:
      original_dy = tf.reshape(
          tf.cast(tf.range(1, 5, 1), tf.float32), shape=(4, 1, 1))
      original_shape = tf.constant([6, 4, 4], dtype=tf.int64)
      sess.run(tf.initialize_all_variables())
      begin = tf.constant([0, 0, 0], dtype=tf.int32)
      end = tf.constant([4, 1, 1], dtype=tf.int64)
      strides = tf.constant([1, 1, 1], dtype=tf.int64)
      with self.assertRaisesRegexp(
          TypeError, "Input 'begin' of 'StridedSliceGrad' Op has type int32"
          " that does not match type int64 of argument 'shape'"):
        dx = array_ops.strided_slice_grad(original_shape, begin, end, strides,
                                          original_dy)
        sess.run(dx)


class BenchmarkSlice(object):

  def __init__(self, tensor):
    self.tensor = tensor

  def __getitem__(self, x):
    return self.tensor[x]


class StridedSliceBenchmark(tf.test.Benchmark):
  """Benchmark new strided slice operation on non-trivial case."""

  def run_and_time(self, slice_op):
    tf.initialize_all_variables().run()
    for _ in range(10):
      _ = slice_op.eval()
    iters = 1000
    t0 = time.time()
    for _ in range(iters):
      slice_op.eval()
    t1 = time.time()
    self.report_benchmark(iters=iters, wall_time=(t1 - t0) / 1000.0)

  def make_variable(self):
    n = 256
    shape = (n, n, n)
    items = n**3
    var = tf.Variable(
        tf.reshape(
            tf.linspace(1., float(items), items), shape),
        dtype=tf.float32)
    return var

  def benchmark_strided_slice_skip(self):
    with tf.Session():
      var = self.make_variable()
      helper = BenchmarkSlice(var)
      slice_op = helper[::2, ::1, ::2]
      self.run_and_time(slice_op)

  def benchmark_strided_slice_easy(self):
    with tf.Session():
      var = self.make_variable()
      helper = BenchmarkSlice(var)
      slice_op = helper[3::1, 3::1, 3::1]
      self.run_and_time(slice_op)

  def benchmark_slice_easy(self):
    with tf.Session():
      var = self.make_variable()
      slice_op = var[3::1, 3::1, 3::1]
      self.run_and_time(slice_op)


class StridedSliceAssignChecker(object):

  def __init__(self, test, x, tensor_type=tf.int32):
    self.tensor_type = tensor_type
    self.test = test
    self.x = tf.cast(tf.constant(x, dtype=tf.float32), dtype=tensor_type)
    self.x_np = np.array(x)

  def __setitem__(self, index, value):
    with self.test.test_session() as sess:
      var = tf.Variable(self.x)
      sess.run(tf.initialize_variables([var]))
      val = sess.run(var[index].assign(
          tf.constant(
              value, dtype=self.tensor_type)))
      valnp = np.copy(self.x_np)
      valnp[index] = np.array(value)
      self.test.assertAllEqual(val, valnp)


class SliceAssignTest(test_util.TensorFlowTestCase):

  def testInvalidSlice(self):
    with self.test_session() as sess:
      foo = tf.constant([1, 2, 3])
      with self.assertRaisesRegexp(ValueError, "Sliced assignment"
                                   " is only supported for variables"):
        bar = foo[:2].assign(tf.constant([1, 2]))
        sess.run(bar)

  def testSliceAssign(self):
    checker = StridedSliceAssignChecker(self, [[1, 2, 3], [4, 5, 6]])
    # Check if equal
    checker[:] = [[10, 20, 30], [40, 50, 60]]
    # Check trivial (1,1) shape tensor
    checker[1:2, 1:2] = [[666]]
    # shrinks shape changes
    checker[1:2, 1] = [666]
    checker[1, 1:2] = [666]
    checker[1, 1] = 666
    # newaxis shape changes
    checker[:, None, :] = [[[10, 20, 30]], [[40, 50, 50]]]
    # shrink and newaxis
    checker[None, None, 0, 0:1] = [[[999]]]
    # Non unit strides
    checker[::1, ::-2] = [[33, 333], [44, 444]]
    # degenerate interval
    checker[8:10, 0] = []
    checker[8:10, 8:10] = [[]]
    # Assign vector to scalar (rank-0) using newaxis
    checker2 = StridedSliceAssignChecker(self, 2225)
    checker2[()] = 6  # no indices
    checker2[...] = 6  # ellipsis
    checker2[None] = [6]  # new axis

  def testUninitialized(self):
    with self.assertRaisesRegexp(
        errors.FailedPreconditionError,
        "Attempting to use uninitialized value Variable"):
      with self.test_session() as sess:
        v = tf.Variable([1, 2])
        sess.run(v[:].assign([1, 2]))


class ShapeSizeRankTest(test_util.TensorFlowTestCase):

  def testDenseShape(self):
    with self.test_session():
      t_value = [[0, 42], [24, 0]]
      self.assertAllEqual((2, 2), tf.shape(t_value).eval())
      self.assertEqual(4, tf.size(t_value).eval())
      self.assertEqual(2, tf.rank(t_value).eval())

      t = tf.constant(t_value)
      self.assertAllEqual((2, 2), tf.shape(t).eval())
      self.assertEqual(4, tf.size(t).eval())
      self.assertEqual(2, tf.rank(t).eval())

  def testSparseShape(self):
    with self.test_session():
      sp_value = tf.SparseTensorValue(
          indices=((0, 1), (1, 0)),
          values=(42, 24),
          shape=(2, 2))
      self.assertAllEqual((2, 2), tf.shape(sp_value).eval())
      self.assertEqual(4, tf.size(sp_value).eval())
      self.assertEqual(2, tf.rank(sp_value).eval())

      sp = tf.SparseTensor.from_value(sp_value)
      self.assertAllEqual((2, 2), tf.shape(sp).eval())
      self.assertEqual(4, tf.size(sp).eval())
      self.assertEqual(2, tf.rank(sp).eval())


class SequenceMaskTest(test_util.TensorFlowTestCase):

  def testExceptions(self):
    with self.test_session():
      with self.assertRaisesRegexp(ValueError, "lengths must be 1D"):
        tf.sequence_mask([[10, 20]], [10, 20])
      with self.assertRaisesRegexp(ValueError, "maxlen must be scalar"):
        tf.sequence_mask([10, 20], [10, 20])

  def testNormal(self):
    with self.test_session():
      res = tf.sequence_mask(tf.constant([1, 3, 2]), 5)
      self.assertAllEqual(res.get_shape(), [3, 5])
      self.assertAllEqual(res.eval(), [[True, False, False, False, False],
                                       [True, True, True, False, False],
                                       [True, True, False, False, False]])

      # test dtype and default maxlen:
      res = tf.sequence_mask(tf.constant([0, 1, 4]), dtype=tf.float32)
      self.assertAllEqual(res.get_shape().as_list(), [3, None])
      self.assertAllEqual(res.eval(), [[0.0, 0.0, 0.0, 0.0],
                                       [1.0, 0.0, 0.0, 0.0],
                                       [1.0, 1.0, 1.0, 1.0]])

if __name__ == "__main__":
  tf.test.main()
