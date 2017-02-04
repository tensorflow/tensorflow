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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SegmentReductionHelper(test.TestCase):

  def _input(self, input_shape, dtype=dtypes_lib.int32):
    num_elem = 1
    for x in input_shape:
      num_elem *= x
    values = np.arange(1, num_elem + 1)
    np_values = values.reshape(input_shape).astype(dtype.as_numpy_dtype)
    return constant_op.constant(
        values, shape=input_shape, dtype=dtype), np_values

  def _segmentReduce(self, indices, x, op1, op2=None, num_out_rows=None):
    if not x.size:
      return np.array([])
    indices = np.asarray(indices)
    if num_out_rows is None:
      num_out_rows = indices[-1] + 1
    output = [None] * num_out_rows
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
    # zero initialize values that are still uncalcuated.
    # output = [o if o is not None else np.zeros(slice_shape) for o in output]
    if not op1 == np.max:
      output = [o if o is not None else np.zeros(slice_shape) for o in output]
    else:
      zeroslice = np.zeros(slice_shape)
      zeroslice.fill(dtype.min)
      output = [o if o is not None else zeroslice for o in output]
    if op2 is not None:
      output = [op2(o) for o in output]
    output = [o.reshape(slice_shape) for o in output]
    return np.array(output)

  def _assertAllClose(self, indices, np_x, tf_x):
    for i in set(np.asarray(indices).ravel()):
      self.assertAllClose(np_x[i], tf_x[i])

  def _mean_cum_op(self, x, y):
    return (x[0] + y, x[1] + 1) if isinstance(x, tuple) else (x + y, 2)

  def _mean_reduce_op(self, x):
    return x[0] / x[1] if isinstance(x, tuple) else x


class SegmentReductionOpTest(SegmentReductionHelper):

  def testValues(self):
    dtypes = [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int64,
        dtypes_lib.int32, dtypes_lib.complex64, dtypes_lib.complex128
    ]

    # Each item is np_op1, np_op2, tf_op
    ops_list = [(np.add, None, math_ops.segment_sum), (self._mean_cum_op,
                                                       self._mean_reduce_op,
                                                       math_ops.segment_mean),
                (np.ndarray.__mul__, None, math_ops.segment_prod),
                (np.minimum, None, math_ops.segment_min),
                (np.maximum, None, math_ops.segment_max)]

    # A subset of ops has been enabled for complex numbers
    complex_ops_list = [(np.add, None, math_ops.segment_sum),
                        (np.ndarray.__mul__, None, math_ops.segment_prod)]

    n = 10
    shape = [n, 2]
    indices = [i // 3 for i in range(n)]
    for dtype in dtypes:
      if dtype in (dtypes_lib.complex64, dtypes_lib.complex128):
        curr_ops_list = complex_ops_list
      else:
        curr_ops_list = ops_list

      with self.test_session(use_gpu=False):
        tf_x, np_x = self._input(shape, dtype=dtype)
        for np_op1, np_op2, tf_op in curr_ops_list:
          np_ans = self._segmentReduce(indices, np_x, np_op1, np_op2)
          s = tf_op(data=tf_x, segment_ids=indices)
          tf_ans = s.eval()
          self._assertAllClose(indices, np_ans, tf_ans)
          # NOTE(mrry): The static shape inference that computes
          # `tf_ans.shape` can only infer that sizes from dimension 1
          # onwards, because the size of dimension 0 is data-dependent
          # and may therefore vary dynamically.
          self.assertAllEqual(np_ans.shape[1:], tf_ans.shape[1:])

  def testSegmentIdsShape(self):
    shape = [4, 4]
    tf_x, _ = self._input(shape)
    indices = constant_op.constant([0, 1, 2, 2], shape=[2, 2])
    with self.assertRaises(ValueError):
      math_ops.segment_sum(data=tf_x, segment_ids=indices)

  def testSegmentIdsSize(self):
    shape = [4, 4]
    with self.test_session():
      tf_x, _ = self._input(shape)
      indices = [0, 1]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment_ids should be the same size"):
        s.eval()

  def testSegmentIdsValid(self):
    # This is a baseline for the following SegmentIdsInvalid* tests.
    shape = [4, 4]
    with self.test_session():
      tf_x, _ = self._input(shape)
      indices = [0, 0, 0, 1]
      result = math_ops.segment_sum(data=tf_x, segment_ids=indices).eval()
      self.assertAllEqual([[15, 18, 21, 24], [13, 14, 15, 16]], result)

  def testSegmentIdsInvalid1(self):
    shape = [4, 4]
    with self.test_session():
      tf_x, _ = self._input(shape)
      indices = [-1, -1, 0, 0]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment ids do not start at 0"):
        s.eval()

  def testSegmentIdsInvalid2(self):
    shape = [4, 4]
    with self.test_session():
      tf_x, _ = self._input(shape)
      indices = [1, 1, 2, 2]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment ids do not start at 0"):
        s.eval()

  def testSegmentIdsInvalid3(self):
    shape = [4, 4]
    with self.test_session():
      tf_x, _ = self._input(shape)
      indices = [0, 0, 2, 2]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment ids are not increasing by 1"):
        s.eval()

  def testSegmentIdsInvalid4(self):
    shape = [4, 4]
    with self.test_session():
      tf_x, _ = self._input(shape)
      indices = [0, 1, 0, 1]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment ids are not increasing by 1"):
        s.eval()

  def testSegmentIdsInvalid5(self):
    shape = [4, 4]
    with self.test_session():
      tf_x, _ = self._input(shape)
      indices = [0, 1, 2, 0]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError(
          r"Segment id 1 out of range \[0, 1\), probably "
          "because 'segment_ids' input is not sorted."):
        s.eval()

  def testSegmentIdsInvalid6(self):
    shape = [4, 4]
    with self.test_session():
      tf_x, _ = self._input(shape)
      indices = [0, 0, 0, -1]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment ids must be >= 0"):
        s.eval()

  def testSegmentIdsInvalid7(self):
    shape = [4, 4]
    with self.test_session():
      tf_x, _ = self._input(shape)
      indices = [0, 0, 0, -2]
      s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment ids must be >= 0"):
        s.eval()

  def testGradient(self):
    shape = [4, 4]
    indices = [0, 1, 2, 2]
    for tf_op in [
        math_ops.segment_sum, math_ops.segment_mean, math_ops.segment_min,
        math_ops.segment_max
    ]:
      with self.test_session():
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float64)
        s = tf_op(data=tf_x, segment_ids=indices)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [3, 4],
            x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)


class UnsortedSegmentSumTest(SegmentReductionHelper):
  use_gpu = False

  def testValues(self):
    dtypes = [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int64,
        dtypes_lib.int32, dtypes_lib.complex64, dtypes_lib.complex128
    ]
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in dtypes:
        with self.test_session(use_gpu=self.use_gpu):
          tf_x, np_x = self._input(shape, dtype=dtype)
          np_ans = self._segmentReduce(
              indices, np_x, np.add, op2=None, num_out_rows=num_segments)
          s = math_ops.unsorted_segment_sum(
              data=tf_x, segment_ids=indices, num_segments=num_segments)
          tf_ans = s.eval()
        self._assertAllClose(indices, np_ans, tf_ans)
        self.assertShapeEqual(np_ans, s)

  def testGradientSegmentSum(self):
    num_cols = 2
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = max(indices_flat) + 3
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (num_cols,)
      with self.test_session(use_gpu=self.use_gpu):
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float64)
        s = math_ops.unsorted_segment_sum(
            data=tf_x, segment_ids=indices, num_segments=num_segments)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [num_segments, num_cols],
            x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)

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
    with self.test_session(use_gpu=self.use_gpu):
      tf_x, np_x = self._input(shape, dtype=dtypes_lib.float64)
      # Results from UnsortedSegmentSum
      unsorted_s = math_ops.unsorted_segment_sum(
          data=tf_x, segment_ids=indices, num_segments=num_segments)
      (unsorted_jacob_t, unsorted_jacob_n) = gradient_checker.compute_gradient(
          tf_x,
          shape,
          unsorted_s, [num_segments, num_cols],
          x_init_value=np_x.astype(np.double),
          delta=1)
      # Results from SegmentSum
      sorted_s = math_ops.segment_sum(data=tf_x, segment_ids=indices)
      sorted_jacob_t, sorted_jacob_n = gradient_checker.compute_gradient(
          tf_x,
          shape,
          sorted_s, [num_segments, num_cols],
          x_init_value=np_x.astype(np.double),
          delta=1)
    self.assertAllClose(unsorted_jacob_t, sorted_jacob_t, rtol=1e-3, atol=1e-3)
    self.assertAllClose(unsorted_jacob_n, sorted_jacob_n, rtol=1e-3, atol=1e-3)

  def testBadIndices(self):
    # Note: GPU kernel does not return the out-of-range error needed for this
    # test, so this test is marked as cpu-only.
    with self.test_session(use_gpu=False):
      for bad in [[-1]], [[7]]:
        unsorted = math_ops.unsorted_segment_sum([[17]], bad, num_segments=2)
        with self.assertRaisesOpError(
            r"segment_ids\[0,0\] = %d is out of range \[0, 2\)" % bad[0][0]):
          unsorted.eval()

  def testEmptySecondDimension(self):
    dtypes = [
        np.float32, np.float64, np.int64, np.int32, np.complex64, np.complex128
    ]
    with self.test_session(use_gpu=self.use_gpu):
      for dtype in dtypes:
        for itype in (np.int32, np.int64):
          data = np.zeros((2, 0), dtype=dtype)
          segment_ids = np.array([0, 1], dtype=itype)
          unsorted = math_ops.unsorted_segment_sum(data, segment_ids, 2)
          self.assertAllEqual(unsorted.eval(), np.zeros((2, 0), dtype=dtype))

  def testGradientSegmentMax(self):
    num_cols = 2
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = max(indices_flat) + 3
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (num_cols,)
      with self.test_session():
        tf_x, np_x = self._input(shape, dtype=dtypes_lib.float64)
        s = math_ops.unsorted_segment_max(data=tf_x, segment_ids=indices,
                                    num_segments=num_segments)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s,
            [num_segments, num_cols],
            x_init_value=np_x.astype(np.double), delta=1)
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)

class UnsortedSegmentSumGpuTest(UnsortedSegmentSumTest):
  use_gpu = True


class SparseSegmentReductionHelper(SegmentReductionHelper):

  def _sparse_input(self, input_shape, num_indices, dtype=dtypes_lib.int32):
    a, b = super(SparseSegmentReductionHelper, self)._input(input_shape, dtype)
    indices = np.random.randint(0, input_shape[0], num_indices).astype(np.int32)
    return (constant_op.constant(
        indices, dtype=dtypes_lib.int32), indices, a, b)

  def _sparseSegmentReduce(self, x, indices, segment_indices, op1, op2=None):
    return self._segmentReduce(segment_indices, x[indices], op1, op2)


class SparseSegmentReductionOpTest(SparseSegmentReductionHelper):

  def testValues(self):
    dtypes = [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int64,
        dtypes_lib.int32
    ]

    mean_dtypes = [dtypes_lib.float32, dtypes_lib.float64]

    # Each item is np_op1, np_op2, tf_op
    ops_list = [(np.add, None, math_ops.sparse_segment_sum),
                (self._mean_cum_op, self._mean_reduce_op,
                 math_ops.sparse_segment_mean)]

    n = 400
    shape = [n, 2]
    segment_indices = []
    for i in range(20):
      for _ in range(i + 1):
        segment_indices.append(i)
    num_indices = len(segment_indices)
    for dtype in dtypes:
      with self.test_session(use_gpu=False):
        tf_indices, np_indices, tf_x, np_x = self._sparse_input(
            shape, num_indices, dtype=dtype)
        for np_op1, np_op2, tf_op in ops_list:
          if tf_op == math_ops.sparse_segment_mean and dtype not in mean_dtypes:
            continue
          np_ans = self._sparseSegmentReduce(np_x, np_indices, segment_indices,
                                             np_op1, np_op2)
          s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
          tf_ans = s.eval()
          self._assertAllClose(segment_indices, np_ans, tf_ans)
          # NOTE(mrry): The static shape inference that computes
          # `tf_ans.shape` can only infer that sizes from dimension 1
          # onwards, because the size of dimension 0 is data-dependent
          # and may therefore vary dynamically.
          self.assertAllEqual(np_ans.shape[1:], tf_ans.shape[1:])

  def testValid(self):
    # Baseline for the test*Invalid* methods below.
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        s.eval()

  def testIndicesInvalid1(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, -1, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"indices\[1\] == -1 out of range \[0, 10\)"):
          s.eval()

  def testIndicesInvalid2(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 10]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"indices\[3\] == 10 out of range \[0, 10\)"):
          s.eval()

  def testSegmentsInvalid1(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 2, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids are not increasing by 1"):
          s.eval()

  def testSegmentsInvalid2(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 0, 1]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids are not increasing by 1"):
          s.eval()

  def testSegmentsInvalid3(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 1, 2, 0]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError(
            r"Segment id 1 out of range \[0, 1\), probably because "
            "'segment_ids' input is not sorted"):
          s.eval()

  def testSegmentsInvalid4(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [-1, 0, 1, 1]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids do not start at 0"):
          s.eval()

  def testSegmentsInvalid5(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [1, 2, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids do not start at 0"):
          s.eval()

  def testSegmentsInvalid6(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 0, 0, -1]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          s.eval()

  def testSegmentsInvalid7(self):
    tf_x, _ = self._input([10, 4], dtype=dtypes_lib.float32)
    ops_list = [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]
    segment_indices = [0, 0, 0, -2]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        with self.assertRaisesOpError("segment ids must be >= 0"):
          s.eval()

  def testGradient(self):
    shape = [10, 4]

    segment_indices = [0, 1, 2, 2]
    num_indices = len(segment_indices)
    for tf_op in [math_ops.sparse_segment_sum, math_ops.sparse_segment_mean]:
      with self.test_session():
        tf_indices, _, tf_x, np_x = self._sparse_input(
            shape, num_indices, dtype=dtypes_lib.float64)
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        jacob_t, jacob_n = gradient_checker.compute_gradient(
            tf_x,
            shape,
            s, [3, 4],
            x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)

  def testGradientValid(self):
    # Baseline for the testGradient*Invalid* methods below.
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_mean_grad, math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        s.eval()

  def testGradientIndicesInvalid1(self):
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_mean_grad, math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, 0, 10]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Index 10 out of range \[0, 10\)"):
          s.eval()

  def testGradientIndicesInvalid2(self):
    tf_x, _ = self._input([3, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_mean_grad, math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 2]
    tf_indices = [8, 3, -1, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Index -1 out of range \[0, 10\)"):
          s.eval()

  def testGradientSegmentsInvalid1(self):
    tf_x, _ = self._input(
        [3, 4], dtype=dtypes_lib.float32)  # expecting 3 segments
    ops_list = [
        math_ops.sparse_segment_mean_grad, math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 1, 1]  # 2 segments
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError("Invalid number of segments"):
          s.eval()

  def testGradientSegmentsInvalid2(self):
    tf_x, _ = self._input([1, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_mean_grad, math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, 0]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id 1 out of range \[0, 1\)"):
          s.eval()

  def testGradientSegmentsInvalid3(self):
    tf_x, _ = self._input([2, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_mean_grad, math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [-1, 0, 1, 1]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id -1 out of range \[0, 2\)"):
          s.eval()

  def testGradientSegmentsInvalid4(self):
    tf_x, _ = self._input([0, 4], dtype=dtypes_lib.float32)
    ops_list = [
        math_ops.sparse_segment_mean_grad, math_ops.sparse_segment_sqrt_n_grad
    ]
    segment_indices = [0, 1, 2, -1]
    tf_indices = [8, 3, 0, 9]
    with self.test_session(use_gpu=False):
      for tf_op in ops_list:
        s = tf_op(tf_x, tf_indices, segment_indices, 10)
        with self.assertRaisesOpError(r"Segment id 0 out of range \[0, 0\)"):
          s.eval()


if __name__ == "__main__":
  test.main()
