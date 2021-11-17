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
"""Functional tests for segment reduction ops."""

import itertools

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class SegmentReductionHelper(test.TestCase):

  def _input(self, input_shape, dtype=dtypes_lib.int32):
    num_elem = 1
    for x in input_shape:
      num_elem *= x
    values = np.arange(1, num_elem + 1)
    np_values = values.reshape(input_shape).astype(dtype.as_numpy_dtype)
    # Add a non-zero imaginary component to complex types.
    if dtype.is_complex:
      np_values -= 1j * np_values
    return constant_op.constant(
        np_values, shape=input_shape, dtype=dtype), np_values

  def _segmentReduce(self, indices, x, op1, op2=None, num_segments=None,
                     initial_value=0):
    if not x.size:
      return np.array([])
    indices = np.asarray(indices)
    if num_segments is None:
      num_segments = indices[-1] + 1
    output = [None] * num_segments
    slice_shape = x.shape[indices.ndim:]
    x_flat = x.reshape((indices.size,) + slice_shape)
    for i, index in enumerate(indices.ravel()):
      if (output[index] is not None) and op1 == np.max:
        for j in range(0, output[index].shape[0]):
          output[index][j] = op1([output[index][j], x_flat[i][j]])
      elif output[index] is not None:
        output[index] = op1(output[index], x_flat[i])
      else:
        output[index] = x_flat[i]
    # zero initialize values that are still uncalculated.
    initial_value_slice = np.ones(slice_shape) * initial_value
    output = [o if o is not None else initial_value_slice for o in output]
    if op2 is not None:
      output = [op2(o) for o in output]
    output = [o.reshape(slice_shape) for o in output]
    return np.array(output)

  def _mean_cum_op(self, x, y):
    return (x[0] + y, x[1] + 1) if isinstance(x, tuple) else (x + y, 2)

  def _mean_reduce_op(self, x):
    return x[0] / x[1] if isinstance(x, tuple) else x

  def _sqrt_n_reduce_op(self, x):
    return x[0] / np.sqrt(x[1]) if isinstance(x, tuple) else x


class SegmentReductionOpTest(SegmentReductionHelper):

  def testValues(self):
    dtypes = [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int64,
        dtypes_lib.int32, dtypes_lib.complex64, dtypes_lib.complex128
    ]

    # Each item is np_op1, np_op2, tf_op
    ops_list = [(np.add, None, math_ops.segment_sum),
                (self._mean_cum_op, self._mean_reduce_op,
                 math_ops.segment_mean),
                (np.ndarray.__mul__, None, math_ops.segment_prod),
                (np.minimum, None, math_ops.segment_min),
                (np.maximum, None, math_ops.segment_max)]

    # A subset of ops has been enabled for complex numbers
    complex_ops_list = [(np.add, None, math_ops.segment_sum),
                        (np.ndarray.__mul__, None, math_ops.segment_prod),
                        (self._mean_cum_op, self._mean_reduce_op,
                         math_ops.segment_mean)]

    n = 10
    # Note that the GPU implem has different paths for different inner sizes.
    for shape in [[n, 1], [n, 2], [n, 3], [n, 32]]:
      indices = [i // 3 for i in range(n)]
      for dtype in dtypes:
        if dtype in (dtypes_lib.complex64, dtypes_lib.complex128):
          curr_ops_list = complex_ops_list
        else:
          curr_ops_list = ops_list
        for use_gpu in [True, False]:
          with self.cached_session(use_gpu=use_gpu):
            tf_x, np_x = self._input(shape, dtype=dtype)
            for np_op1, np_op2, tf_op in curr_ops_list:
              initial_value = 1 if tf_op is math_ops.segment_prod else 0
              np_ans = self._segmentReduce(
                  indices, np_x, np_op1, np_op2, initial_value=initial_value)
              s = tf_op(data=tf_x, segment_ids=indices)
              tf_ans = self.evaluate(s)
              self.assertAllClose(np_ans, tf_ans)
              # NOTE(mrry): The static shape inference that computes
              # `tf_ans.shape` can only infer that sizes from dimension 1
              # onwards, because the size of dimension 0 is data-dependent
              # and may therefore vary dynamically.
              self.assertAllEqual(np_ans.shape[1:], tf_ans.shape[1:])

  @test_util.run_deprecated_v1
  def testSegmentIdsShape(self):
    shape = [4, 4]
    tf_x, _ = self._input(shape)
    indices = constant_op.constant([0, 1, 2, 2], shape=[2, 2])
    with self.assertRaises(ValueError):
      math_ops.segment_sum(data=tf_x, segment_ids=indices)

  @test_util.run_deprecated_v1
  def testSegmentIdsSize(self):
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape)
        indices = [0, 1]
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        with self.assertRaisesOpError("segment_ids should be the same size"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsValid(self):
    # This is a baseline for the following SegmentIdsInvalid* tests.
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 0, 1]
        result = math_ops.segment_sum(data=tf_x, segment_ids=indices).eval()
        self.assertAllEqual([[15, 18, 21, 24], [13, 14, 15, 16]], result)

  def testSegmentIdsGreaterThanZero(self):
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float32)
        indices = [1, 1, 2, 2]
        np_ans = self._segmentReduce(indices, np_x, np.add)
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  def testSegmentIdsHole(self):
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 3, 3]
        np_ans = self._segmentReduce(indices, np_x, np.add)
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid1(self):
    shape = [4, 4]
    with self.cached_session():
      tf_x, _ = self._input(shape)
      indices = [-1, -1, 0, 0]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError(
          r"Segment id -1 out of range \[0, 1\), possibly because "
          "'segment_ids' input is not sorted."):
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid2(self):
    shape = [4, 4]
    with self.cached_session():
      tf_x, _ = self._input(shape)
      indices = [0, 1, 0, 1]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment ids are not increasing"):
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid3(self):
    shape = [4, 4]
    with self.cached_session():
      tf_x, _ = self._input(shape)
      indices = [0, 1, 2, 0]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError(
          r"Segment id 1 out of range \[0, 1\), possibly "
          "because 'segment_ids' input is not sorted."):
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid4(self):
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 0, -1]
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentIdsInvalid5(self):
    shape = [4, 4]
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        tf_x, _ = self._input(shape, dtype=dtypes_lib.float32)
        indices = [0, 0, 0, -2]
        s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradient(self):
    shape = [4, 4]
    indices = [0, 1, 2, 2]
    for tf_op in [
        math_ops.segment_sum, math_ops.segment_mean, math_ops.segment_min,
        math_ops.segment_max
    ]:
      with self.cached_session():
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float64)
        s = tf_op(data=tf_x, segment_ids=indices)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [3, 4],
            x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n)

  def testDataInvalid(self):
    # Test case for GitHub issue 40653.
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        with self.assertRaisesRegex(
            (ValueError, errors_impl.InvalidArgumentError),
            "must be at least rank 1"):
          s = math_ops.segment_mean(
              data=np.uint16(10), segment_ids=np.array([]).astype("int64"))
          self.evaluate(s)

  def testInvalidIds(self):
    # Test case for GitHub issue 46888.
    for op in [
        math_ops.segment_max,
        math_ops.segment_min,
        math_ops.segment_mean,
        math_ops.segment_sum,
        math_ops.segment_prod,
    ]:
      with self.cached_session(use_gpu=False):
        with self.assertRaises((ValueError, errors_impl.InternalError)):
          s = op(data=np.ones((1, 10, 1)), segment_ids=[1676240524292489355])
          self.evaluate(s)


class UnsortedSegmentTest(SegmentReductionHelper):

  def __init__(self, methodName='runTest'):
    # Each item is np_op1, np_op2, tf_op, initial_value functor
    self.ops_list = [(np.add, None,
                      math_ops.unsorted_segment_sum, lambda t: 0),
                     (self._mean_cum_op, self._mean_reduce_op,
                      math_ops.unsorted_segment_mean, lambda t: 0),
                     (self._mean_cum_op, self._sqrt_n_reduce_op,
                      math_ops.unsorted_segment_sqrt_n, lambda t: 0),
                     (np.ndarray.__mul__, None,
                      math_ops.unsorted_segment_prod, lambda t: 1),
                     (np.minimum, None,
                      math_ops.unsorted_segment_min, lambda t: t.max),
                     (np.maximum, None,
                      math_ops.unsorted_segment_max, lambda t: t.min)]

    # A subset of ops has been enabled for complex numbers
    self.complex_ops_list = [(np.add, None,
                              math_ops.unsorted_segment_sum, lambda t: 0),
                             (np.ndarray.__mul__, None,
                              math_ops.unsorted_segment_prod, lambda t: 1)]
    self.differentiable_dtypes = [dtypes_lib.float16, dtypes_lib.float32,
                                  dtypes_lib.float64]
    self.all_dtypes = (self.differentiable_dtypes +
                       [dtypes_lib.bfloat16,
                        dtypes_lib.int64, dtypes_lib.int32,
                        dtypes_lib.complex64, dtypes_lib.complex128])
    super(UnsortedSegmentTest, self).__init__(methodName=methodName)

  def testValues(self):
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      # Note that the GPU implem has different paths for different inner sizes.
      for inner_size in [1, 2, 3, 32]:
        shape = indices.shape + (inner_size,)
        for dtype in self.all_dtypes:
          ops_list = (
              self.complex_ops_list if dtype.is_complex else self.ops_list)
          tf_x, np_x = self._input(shape, dtype=dtype)
          for use_gpu in [True, False]:
            with self.cached_session():
              for np_op1, np_op2, tf_op, init_op in ops_list:
                # sqrt_n doesn't support integers
                if (np_op2 == self._sqrt_n_reduce_op and dtype.is_integer):
                  continue
                # todo(philjd): enable this test once real_div supports bfloat16
                if (np_op2 in [self._sqrt_n_reduce_op, self._mean_reduce_op] and
                    dtype == dtypes_lib.bfloat16):
                  continue
                np_ans = self._segmentReduce(
                    indices,
                    np_x,
                    np_op1,
                    np_op2,
                    num_segments=num_segments,
                    initial_value=init_op(dtype))
                s = tf_op(tf_x, segment_ids=indices, num_segments=num_segments)
                tf_ans = self.evaluate(s)
                if dtype is dtypes_lib.bfloat16:
                  tf_ans = tf_ans.astype(np.float32)
                self.assertAllCloseAccordingToType(np_ans, tf_ans)
                self.assertShapeEqual(np_ans, s)

  def testNumSegmentsTypes(self):
    dtypes = [dtypes_lib.int32, dtypes_lib.int64]
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in dtypes:
        with self.cached_session():
          tf_x, np_x = self._input(shape)
          num_segments_constant = constant_op.constant(
              num_segments, dtype=dtype)
          np_ans = self._segmentReduce(
              indices, np_x, np.add, op2=None, num_segments=num_segments)
          s = math_ops.unsorted_segment_sum(
              data=tf_x,
              segment_ids=indices,
              num_segments=num_segments_constant)
          tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, s)

  @test_util.run_deprecated_v1
  def testGradientsTFGradients(self):
    num_cols = 2
    indices_flat = np.array([0, 4, 0, -1, 3, -1, 4, 7, 7, 3])
    num_segments = max(indices_flat) + 3
    for dtype in self.differentiable_dtypes:
      ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
      for indices in indices_flat, indices_flat.reshape(5, 2):
        shape = indices.shape + (num_cols,)
        # test CPU and GPU as tf.gather behaves differently on each device
        for use_gpu in [False, True]:
          with self.cached_session(use_gpu=use_gpu):
            for _, _, tf_op, _ in ops_list:
              tf_x, np_x = self._input(shape, dtype=dtype)
              s = tf_op(tf_x, indices, num_segments)
              jacob_t, jacob_n = gradient_checker.compute_gradient(
                  tf_x,
                  shape,
                  s, [num_segments, num_cols],
                  x_init_value=np_x,
                  delta=1.)
              self.assertAllCloseAccordingToType(jacob_t, jacob_n,
                                                 half_atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def testGradientsGradientTape(self):
    num_cols = 2
    indices_flat = np.array([0, 4, 0, -1, 3, -1, 4, 7, 7, 3])
    num_segments = max(indices_flat) + 3
    for dtype in self.differentiable_dtypes:
      ops_list = self.complex_ops_list if dtype.is_complex else self.ops_list
      for indices in indices_flat, indices_flat.reshape(5, 2):
        shape = indices.shape + (num_cols,)
        # test CPU and GPU as tf.gather behaves differently on each device
        for use_gpu in [test_util.use_gpu, test_util.force_cpu]:
          with use_gpu():
            for _, _, tf_op, _ in ops_list:
              _, np_x = self._input(shape, dtype=dtype)
              # pylint: disable=cell-var-from-loop
              def f(x):
                return tf_op(x, indices, num_segments)
              gradient_tape_jacob_t, jacob_n = (
                  gradient_checker_v2.compute_gradient(
                      f, [np_x], delta=1.))
              # pylint: enable=cell-var-from-loop
              self.assertAllCloseAccordingToType(jacob_n, gradient_tape_jacob_t,
                                                 half_atol=1e-2)

  @test_util.run_deprecated_v1
  def testProdGrad(self):
    # additional test for the prod gradient to ensure correct handling of zeros
    values = np.array([0, 0, 1, 0, 2, 2, 3, 3, 3], dtype=np.float32)
    indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int32)
    indices_neg = np.array([-1, 0, 0, -1, 1, 1, -1, 2, 2], dtype=np.int32)
    values_tf = constant_op.constant(values)
    # ground truth partial derivatives
    gradients_indices = np.zeros((9, 3), dtype=np.float32)
    gradients_indices_neg = np.zeros((9, 3), dtype=np.float32)
    # the derivative w.r.t. to the other segments is zero, so here we only
    # explicitly set the grad values for the corresponding segment
    gradients_indices[range(9), indices] = [0, 0, 0, 4, 0, 0, 9, 9, 9]
    gradients_indices_neg[range(9), indices_neg] = [0, 1, 0, 0, 2, 2, 0, 3, 3]
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        for ind, grad_gt in [(indices, gradients_indices),
                             (indices_neg, gradients_indices_neg)]:
          s = math_ops.unsorted_segment_prod(values_tf,
                                             constant_op.constant(ind), 3)
          jacob_t, jacob_n = gradient_checker.compute_gradient(
              values_tf, (9,), s, (3,), x_init_value=values, delta=1)
          self.assertAllClose(jacob_t, jacob_n)
          self.assertAllClose(jacob_t, grad_gt)

  @test_util.run_deprecated_v1
  def testGradientMatchesSegmentSum(self):
    # Strategy: compute the gradient for UnsortedSegmentSum and SegmentSum
    # and compare the outputs, which should be identical.
    # NB: for this test to work, indices must be valid for SegmentSum, namely
    # it must be sorted, the indices must be contiguous, and num_segments
    # must be max(indices) + 1.
    indices = [0, 0, 1, 1, 1, 2, 3, 4, 5]
    n = len(indices)
    num_cols = 2
    shape = [n, num_cols]
    num_segments = max(indices) + 1
    for dtype in self.differentiable_dtypes:
      with self.cached_session():
        tf_x, np_x = self._input(shape, dtype=dtype)
        # Results from UnsortedSegmentSum
        unsorted_s = math_ops.unsorted_segment_sum(
            data=tf_x, segment_ids=indices, num_segments=num_segments)
        unsorted_jacob_t, unsorted_jacob_n = (
            gradient_checker.compute_gradient(tf_x, shape, unsorted_s,
                                              [num_segments, num_cols],
                                              x_init_value=np_x, delta=1))

        # Results from SegmentSum
        sorted_s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
        sorted_jacob_t, sorted_jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            sorted_s, [num_segments, num_cols],
            x_init_value=np_x,
            delta=1)
      self.assertAllClose(unsorted_jacob_t, sorted_jacob_t)
      self.assertAllClose(unsorted_jacob_n, sorted_jacob_n)

  @test_util.run_deprecated_v1
  def testBadIndices(self):
    # Note: GPU kernel does not return the out-of-range error needed for this
    # test, so this test is marked as cpu-only.
    # Note: With PR #13055 a negative index will be ignored silently.
    with self.session(use_gpu=False):
      for bad in [[2]], [[7]]:
        unsorted = math_ops.unsorted_segment_sum([[17]], bad, num_segments=2)
        with self.assertRaisesOpError(
            r"segment_ids\[0,0\] = %d is out of range \[0, 2\)" % bad[0][0]):
          self.evaluate(unsorted)

  @test_util.run_deprecated_v1
  def testEmptySecondDimension(self):
    dtypes = [np.float16, np.float32, np.float64, np.int64, np.int32,
              np.complex64, np.complex128]
    with self.session():
      for dtype in dtypes:
        for itype in (np.int32, np.int64):
          data = np.zeros((2, 0), dtype=dtype)
          segment_ids = np.array([0, 1], dtype=itype)
          unsorted = math_ops.unsorted_segment_sum(data, segment_ids, 2)
          self.assertAllEqual(unsorted, np.zeros((2, 0), dtype=dtype))

  def testDropNegatives(self):
    # Note: the test is done by replacing segment_ids with 8 to -1
    # for index  and replace values generated by numpy with 0.
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in self.all_dtypes:
        with self.session():
          tf_x, np_x = self._input(shape, dtype=dtype)
          np_ans = self._segmentReduce(
              indices, np_x, np.add, op2=None, num_segments=num_segments)
          # Replace np_ans[8] with 0 for the value
          np_ans[8:] = 0
          # Replace 8 with -1 in indices
          np.place(indices, indices == 8, [-1])
          s = math_ops.unsorted_segment_sum(
              data=tf_x, segment_ids=indices, num_segments=num_segments)
          tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, s)


class SparseSegmentReductionHelper(SegmentReductionHelper):

  def _sparse_input(self, input_shape, num_indices, dtype=dtypes_lib.int32):
    a, b = super(SparseSegmentReductionHelper, self)._input(input_shape, dtype)
    indices = np.random.randint(0, input_shape[0], num_indices).astype(np.int32)
    return (constant_op.constant(
        indices, dtype=dtypes_lib.int32), indices, a, b)

  def _sparseSegmentReduce(self,
                           x,
                           indices,
                           segment_indices,
                           op1,
                           op2=None,
                           num_segments=None):
    return self._segmentReduce(
        segment_indices, x[indices], op1, op2, num_segments=num_segments)

  def _sparseSegmentReduceGrad(self, ygrad, indices, segment_ids, output_dim0,
                               mode):
    assert mode in ("sum", "mean", "sqrtn")
    if mode != "sum":
      weights = np.zeros(ygrad.shape[0], ygrad.dtype)
      for segment in segment_ids:
        weights[segment] += 1
      weights = 1. / weights if mode == "mean" else 1. / np.sqrt(weights)
    xgrad = np.zeros([output_dim0, ygrad.shape[1]], ygrad.dtype)
    for segment, index in zip(segment_ids, indices):
      if mode == "sum":
        xgrad[index] += ygrad[segment]
      else:
        xgrad[index] += ygrad[segment] * weights[segment]
    return xgrad


class SparseSegmentReductionOpTest(SparseSegmentReductionHelper):

  def testValues(self):
    dtypes = [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int64,
        dtypes_lib.int32
    ]

    index_dtypes = [dtypes_lib.int32, dtypes_lib.int64]
    segment_ids_dtypes = [dtypes_lib.int32, dtypes_lib.int64]

    mean_dtypes = [dtypes_lib.float32, dtypes_lib.float64]

    # Each item is np_op1, np_op2, tf_op
    ops_list = [(np.add, None, math_ops.sparse_segment_sum),
                (self._mean_cum_op, self._mean_reduce_op,
                 math_ops.sparse_segment_mean)]

    n = 400
    # Note that the GPU implem has different paths for different inner sizes.
    for inner_size in [1, 2, 3, 32]:
      shape = [n, inner_size]
      segment_indices = []
      for i in range(20):
        for _ in range(i + 1):
          segment_indices.append(i)
      num_indices = len(segment_indices)
      for dtype in dtypes:
        for index_dtype in index_dtypes:
          for segment_ids_dtype in segment_ids_dtypes:
            with self.cached_session():
              tf_indices, np_indices, tf_x, np_x = self._sparse_input(
                  shape, num_indices, dtype=dtype)
              for np_op1, np_op2, tf_op in ops_list:
                if (tf_op == math_ops.sparse_segment_mean and
                    dtype not in mean_dtypes):
                  continue
                np_ans = self._sparseSegmentReduce(np_x, np_indices,
                                                   segment_indices, np_op1,
                                                   np_op2)
                s = tf_op(
                    data=tf_x,
                    indices=math_ops.cast(tf_indices, index_dtype),
                    segment_ids=math_ops.cast(segment_indices,
                                              segment_ids_dtype))
                tf_ans = self.evaluate(s)
                self.assertAllClose(np_ans, tf_ans)
                # NOTE(mrry): The static shape inference that computes
                # `tf_ans.shape` can only infer that sizes from dimension 1
                # onwards, because the size of dimension 0 is data-dependent
                # and may therefore vary dynamically.
                self.assertAllEqual(np_ans.shape[1:], tf_ans.shape[1:])

  def testSegmentIdsHole(self):
    tf_x, np_x = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [(np.add, None, math_ops.sparse_segment_sum), (
        self._mean_cum_op, self._mean_reduce_op, math_ops.sparse_segment_mean)]
    segment_indices = [0, 2, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.session():
      for np_op1, np_op2, tf_op in ops_list:
        np_ans = self._sparseSegmentReduce(np_x, tf_indices, segment_indices,
                                           np_op1, np_op2)
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  def testWithNumSegments(self):
    tf_x, np_x = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [(np.add, None, math_ops.sparse_segment_sum_with_num_segments),
                (self._mean_cum_op, self._mean_reduce_op,
                 math_ops.sparse_segment_mean_with_num_segments)]
    segment_indices = [0, 2, 2, 2]
    tf_indices = [8, 3, 0, 9]
    num_segments = 5
    with self.session():
      for np_op1, np_op2, tf_op in ops_list:
        np_ans = self._sparseSegmentReduce(
            np_x,
            tf_indices,
            segment_indices,
            np_op1,
            np_op2,
            num_segments=num_segments)
        s = tf_op(
            data=tf_x,
            indices=tf_indices,
            segment_ids=segment_indices,
            num_segments=num_segments)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  def testWithEmptySegments(self):
    tf_x = constant_op.constant([], shape=[0, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_with_num_segments,
        math_ops.sparse_segment_mean_with_num_segments
    ]
    segment_indices = []
    tf_indices = []
    num_segments = 5
    with self.session():
      for tf_op in ops_list:
        s = tf_op(
            data=tf_x,
            indices=tf_indices,
            segment_ids=segment_indices,
            num_segments=num_segments)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np.zeros([5, 4]), tf_ans)

  @test_util.run_in_graph_and_eager_modes
  def testSegmentScalarIdiRaisesInvalidArgumentError(self):
    """Test for github #46897."""
    ops_list = [
        math_ops.sparse_segment_sum,
        math_ops.sparse_segment_mean,
        math_ops.sparse_segment_sqrt_n,
    ]
    for op in ops_list:
      with self.assertRaisesRegex(
          (ValueError, errors_impl.InvalidArgumentError),
          "Shape must be at least rank 1"):
        op(data=1.0, indices=[0], segment_ids=[3])

  def testSegmentIdsGreaterThanZero(self):
    tf_x, np_x = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [(np.add, None, math_ops.sparse_segment_sum), (
        self._mean_cum_op, self._mean_reduce_op, math_ops.sparse_segment_mean)]
    segment_indices = [1, 2, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.session():
      for np_op1, np_op2, tf_op in ops_list:
        np_ans = self._sparseSegmentReduce(np_x, tf_indices, segment_indices,
                                           np_op1, np_op2)
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        tf_ans = self.evaluate(s)
        self.assertAllClose(np_ans, tf_ans)

  def testValid(self):
    # Baseline for the test*Invalid* methods below.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.session():
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testIndicesInvalid1(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, -1, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"indices\[1\] == -1 out of range \[0, 10\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testIndicesInvalid2(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 10]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"indices\[3\] == 10 out of range \[0, 10\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid2(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 0, 1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids are not increasing"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid3(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 0]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"Segment id 1 out of range \[0, 1\), possibly because "
            "'segment_ids' input is not sorted"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid4(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [-1, 0, 1, 1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"Segment id -1 out of range \[0, 2\), possibly because "
            "'segment_ids' input is not sorted"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid6(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 0, 0, -1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentsInvalid7(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 0, 0, -2]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          self.evaluate(s)

  def testSegmentWithNumSegmentsValid(self):
    # Baseline for the test*WithNumSegmentsInvalid* methods below.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_with_num_segments,
        math_ops.sparse_segment_mean_with_num_segments,
    ]
    num_segments = 5
    segment_indices = [0, 1, 3, 3]
    tf_indices = [8, 3, 0, 9]
    with self.session():
      for tf_op in ops_list:
        s = tf_op(
            data=tf_x,
            indices=tf_indices,
            segment_ids=segment_indices,
            num_segments=num_segments)
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentWithNumSegmentsInvalid1(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_with_num_segments,
        math_ops.sparse_segment_mean_with_num_segments,
    ]
    num_segments = 5
    segment_indices = [0, 1, 3, 5]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(
            data=tf_x,
            indices=tf_indices,
            segment_ids=segment_indices,
            num_segments=num_segments)
        with self.assertRaisesOpError("segment ids must be < num_segments"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testSegmentWithNumSegmentsInvalid2(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_with_num_segments,
        math_ops.sparse_segment_mean_with_num_segments,
    ]
    num_segments = -2
    segment_indices = [0, 1, 3, 3]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        with self.assertRaisesRegex(
            ValueError, "Cannot specify a negative value for num_segments"):
          tf_op(
              data=tf_x,
              indices=tf_indices,
              segment_ids=segment_indices,
              num_segments=num_segments)

  @test_util.run_deprecated_v1
  def testGradient(self):
    shape = [10, 4]

    segment_indices = [0, 1, 2, 2]
    num_indices = len(segment_indices)
    for tf_op in [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]:
      with self.cached_session():
        tf_indices, _, tf_x, np_x = self._sparse_input(
            shape, num_indices, dtype=dtypes_lib.float64)
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [3, 4],
            x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n)

  @test_util.run_deprecated_v1
  def testGradientWithEmptySegmentsAtEnd(self):
    shape = [10, 4]

    num_segments = 5
    segment_indices = [0, 1, 2, 2]
    num_indices = len(segment_indices)
    for tf_op in [
        math_ops.sparse_segment_sum_with_num_segments,
        math_ops.sparse_segment_mean_with_num_segments,
    ]:
      with self.cached_session():
        tf_indices, _, tf_x, np_x = self._sparse_input(
            shape, num_indices, dtype=dtypes_lib.float64)
        s = tf_op(
            data=tf_x,
            indices=tf_indices,
            segment_ids=segment_indices,
            num_segments=num_segments)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [5, 4],
            x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n)

  def testGradientExplicit(self):
    # Note that the GPU implem has different paths for different inner sizes.
    for inner_size in (1, 2, 3, 32):
      with self.session():
        tf_ygrad, np_ygrad = self._input([3, inner_size],
                                         dtype=dtypes_lib.float32)
        segment_ids = [0, 1, 2, 2, 2]
        indices = [8, 3, 0, 9, 3]
        output_dim0 = 10
        ops_list = [
            (math_ops.sparse_segment_sum_grad, "sum"),
            (math_ops.sparse_segment_mean_grad, "mean"),
            (math_ops.sparse_segment_sqrt_n_grad, "sqrtn"),
        ]
        for tf_op, mode in ops_list:
          np_xgrad = self._sparseSegmentReduceGrad(np_ygrad, indices,
                                                   segment_ids, output_dim0,
                                                   mode)
          tf_xgrad = tf_op(tf_ygrad, indices, segment_ids, output_dim0)
          self.assertAllClose(tf_xgrad, np_xgrad)

  def testGradientExplicitSingleOutput(self):
    # The GPU implem has a special case when there is a single output.
    for inner_size in (1, 2, 3, 32):
      with self.session():
        tf_ygrad, np_ygrad = self._input([3, inner_size],
                                         dtype=dtypes_lib.float32)
        segment_ids = [0, 1, 2, 2, 2]
        indices = [0, 0, 0, 0, 0]
        output_dim0 = 1
        ops_list = [
            (math_ops.sparse_segment_sum_grad, "sum"),
            (math_ops.sparse_segment_mean_grad, "mean"),
            (math_ops.sparse_segment_sqrt_n_grad, "sqrtn"),
        ]
        for tf_op, mode in ops_list:
          np_xgrad = self._sparseSegmentReduceGrad(np_ygrad, indices,
                                                   segment_ids, output_dim0,
                                                   mode)
          tf_xgrad = tf_op(tf_ygrad, indices, segment_ids, output_dim0)
          self.assertAllClose(tf_xgrad, np_xgrad)

  def testGradientValid(self):
    # Baseline for the testGradient*Invalid* methods below.
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientIndicesInvalid1(self):
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 10]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Index 10 out of range \[0, 10\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientIndicesInvalid2(self):
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, -1, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Index -1 out of range \[0, 10\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientSegmentsInvalid1(self):
    tf_x, _ = self._input(
        [3, 4], dtype=dtypes_lib.float32)  # expecting 3 segments
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 1, 4]  # 5 segments
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError("Invalid number of segments"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientSegmentsInvalid2(self):
    tf_x, _ = self._input([1, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 0]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id 1 out of range \[0, 1\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientSegmentsInvalid3(self):
    tf_x, _ = self._input([2, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [-1, 0, 1, 1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id -1 out of range \[0, 2\)"):
          self.evaluate(s)

  @test_util.run_deprecated_v1
  def testGradientSegmentsInvalid4(self):
    tf_x, _ = self._input([0, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_sum_grad, math_ops.sparse_segment_mean_grad,
        math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, -1]
    tf_indices = [8, 3, 0, 9]
    with self.session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id 0 out of range \[0, 0\)"):
          self.evaluate(s)


class SegmentReductionOpBenchmark(test.Benchmark):
  outer_dim_options = [2**x for x in range(9, 14, 2)]
  ratio_options = [2**x for x in range(1, 6, 2)]
  inner_dim_options = [2**x for x in range(9, 14, 2)]
  # randomly generated sizes with less alignments
  inner_dim_options += [
      1120, 1215, 1856, 1302, 1329, 1531, 1313, 1672, 1851, 1584
  ]
  dtype_options = [np.float32, np.float64]
  options = (outer_dim_options, ratio_options, inner_dim_options, dtype_options)
  # pylint: disable=g-long-lambda
  op_functors = [lambda vc, vs, seg_ids:
                 ("sorted", math_ops.segment_sum(vc, vs)),
                 lambda vc, vs, seg_ids:
                 ("unsorted",
                  math_ops.unsorted_segment_sum(vc, vs, seg_ids[-1]+1))]
  # pylint: enable=g-long-lambda
  repeat = 10

  def _npTypeToStr(self, t):
    if t == np.float32:
      return "fp32"
    if t == np.float64:
      return "fp64"

  def _runGraph(self, op_functor, outer_dim, ratio, inner_dim, dtype):
    output_outer_dim = int(outer_dim / ratio)
    const = np.random.randint(5, size=(outer_dim, inner_dim))
    seg_ids = np.sort(np.random.randint(output_outer_dim, size=outer_dim))
    vs = variables.Variable(seg_ids.astype(np.int32))
    with ops.device("/gpu:0"):
      vc = variables.Variable(const.astype(dtype))
    name, op = op_functor(vc, vs, seg_ids)
    with session.Session() as sess:
      self.evaluate(variables.global_variables_initializer())
      r = self.run_op_benchmark(
          sess,
          op,
          min_iters=self.repeat,
          name="_".join(
              map(str,
                  [name, outer_dim, ratio, inner_dim,
                   self._npTypeToStr(dtype)])))
    return name, r["wall_time"]

  def benchmarkSegmentSumGPU(self):
    if not test.is_gpu_available(cuda_only=True):
      return
    for outer_dim, ratio, inner_dim, dtype in itertools.product(*self.options):
      op_functor = self.op_functors[0]
      with ops.Graph().as_default():
        self._runGraph(op_functor, outer_dim, ratio, inner_dim, dtype)

  def benchmarkUnsortedSegmentSumGPU(self):
    if not test.is_gpu_available(cuda_only=True):
      return
    for outer_dim, ratio, inner_dim, dtype in itertools.product(*self.options):
      op_functor = self.op_functors[1]
      with ops.Graph().as_default():
        self._runGraph(op_functor, outer_dim, ratio, inner_dim, dtype)


if __name__ == "__main__":
  test.main()
