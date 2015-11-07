"""Functional tests for segment reduction ops."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.python.kernel_tests import gradient_checker


class SegmentReductionHelper(tf.test.TestCase):

  def _input(self, input_shape, dtype=tf.int32):
    num_elem = 1
    for x in input_shape:
      num_elem *= x
    values = range(1, num_elem + 1)
    np_values = np.array(values).reshape(input_shape).astype(
        dtype.as_numpy_dtype)
    return tf.constant(values, shape=input_shape,
                                dtype=dtype), np_values

  def _segmentReduce(self, indices, x, op1, op2=None, num_out_rows=None):
    if not x.size: return np.array([])
    indices = np.asarray(indices)
    if num_out_rows is None:
      num_out_rows = indices[-1] + 1
    output = [None] * num_out_rows
    slice_shape = x.shape[indices.ndim:]
    x_flat = x.reshape((indices.size,) + slice_shape)
    for i, index in enumerate(indices.ravel()):
      if output[index] is not None:
        output[index] = op1(output[index], x_flat[i])
      else:
        output[index] = x_flat[i]
    # zero initialize values that are still uncalcuated.
    output = [o if o is not None else np.zeros(slice_shape) for o in output]
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
    return  x[0] / x[1] if isinstance(x, tuple) else x


class SegmentReductionOpTest(SegmentReductionHelper):

  def testValues(self):
    dtypes = [tf.float32,
              tf.float64,
              tf.int64,
              tf.int32]

    # Each item is np_op1, np_op2, tf_op
    ops_list = [(np.add, None, tf.segment_sum),
                (self._mean_cum_op, self._mean_reduce_op,
                 tf.segment_mean),
                (np.ndarray.__mul__, None, tf.segment_prod),
                (np.minimum, None, tf.segment_min),
                (np.maximum, None, tf.segment_max)]

    n = 10
    shape = [n, 2]
    indices = [int(i / 3) for i in range(n)]
    for dtype in dtypes:
      with self.test_session(use_gpu=False):
        tf_x, np_x = self._input(shape, dtype=dtype)
        for np_op1, np_op2, tf_op in ops_list:
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
    indices = tf.constant([0, 1, 2, 2], shape=[2, 2])
    with self.assertRaises(ValueError):
      tf.segment_sum(data=tf_x, segment_ids=indices)

  def testSegmentIdsSize(self):
    shape = [4, 4]
    with self.test_session():
      tf_x, _ = self._input(shape)
      indices = [0, 1]
      s = tf.segment_sum(data=tf_x, segment_ids=indices)
      with self.assertRaisesOpError("segment_ids should be the same size"):
        s.eval()

  def testGradient(self):
    shape = [4, 4]
    indices = [0, 1, 2, 2]
    for tf_op in [tf.segment_sum,
                  tf.segment_mean,
                  tf.segment_min,
                  tf.segment_max]:
      with self.test_session():
        tf_x, np_x = self._input(shape, dtype=tf.float64)
        s = tf_op(data=tf_x, segment_ids=indices)
        jacob_t, jacob_n = gradient_checker.ComputeGradient(
            tf_x, shape, s, [3, 4], x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)


class UnsortedSegmentSumTest(SegmentReductionHelper):

  def testValues(self):
    dtypes = [tf.float32,
              tf.float64,
              tf.int64,
              tf.int32]
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = 12
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (2,)
      for dtype in dtypes:
        with self.test_session(use_gpu=False):
          tf_x, np_x = self._input(shape, dtype=dtype)
          np_ans = self._segmentReduce(indices,
                                       np_x,
                                       np.add,
                                       op2=None,
                                       num_out_rows=num_segments)
          s = tf.unsorted_segment_sum(data=tf_x,
                                      segment_ids=indices,
                                      num_segments=num_segments)
          tf_ans = s.eval()
        self._assertAllClose(indices, np_ans, tf_ans)
        self.assertShapeEqual(np_ans, s)

  def testGradient(self):
    num_cols = 2
    indices_flat = np.array([0, 4, 0, 8, 3, 8, 4, 7, 7, 3])
    num_segments = max(indices_flat) + 3
    for indices in indices_flat, indices_flat.reshape(5, 2):
      shape = indices.shape + (num_cols,)
      with self.test_session():
        tf_x, np_x = self._input(shape, dtype=tf.float64)
        s = tf.unsorted_segment_sum(data=tf_x,
                                    segment_ids=indices,
                                    num_segments=num_segments)
        jacob_t, jacob_n = gradient_checker.ComputeGradient(
            tf_x,
            shape,
            s,
            [num_segments, num_cols],
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
    with self.test_session():
      tf_x, np_x = self._input(shape, dtype=tf.float64)
      # Results from UnsortedSegmentSum
      unsorted_s = tf.unsorted_segment_sum(data=tf_x,
                                                 segment_ids=indices,
                                                 num_segments=num_segments)
      unsorted_jacob_t, unsorted_jacob_n = gradient_checker.ComputeGradient(
          tf_x, shape, unsorted_s, [num_segments, num_cols],
          x_init_value=np_x.astype(np.double),
          delta=1)
      # Results from SegmentSum
      sorted_s = tf.segment_sum(data=tf_x, segment_ids=indices)
      sorted_jacob_t, sorted_jacob_n = gradient_checker.ComputeGradient(
          tf_x, shape, sorted_s, [num_segments, num_cols],
          x_init_value=np_x.astype(np.double),
          delta=1)
    self.assertAllClose(unsorted_jacob_t, sorted_jacob_t, rtol=1e-3, atol=1e-3)
    self.assertAllClose(unsorted_jacob_n, sorted_jacob_n, rtol=1e-3, atol=1e-3)


class SparseSegmentReductionHelper(SegmentReductionHelper):

  def _sparse_input(self, input_shape, num_indices,
                    dtype=tf.int32):
    a, b = super(SparseSegmentReductionHelper, self)._input(input_shape,
                                                            dtype)
    indices = np.random.randint(0, input_shape[0], num_indices).astype(np.int32)
    return (tf.constant(indices, dtype=tf.int32),
            indices, a, b)

  def _sparseSegmentReduce(self, x, indices, segment_indices, op1, op2=None):
    return self._segmentReduce(segment_indices, x[indices], op1, op2)


class SparseSegmentReductionOpTest(SparseSegmentReductionHelper):

  def testValues(self):
    dtypes = [tf.float32,
              tf.float64,
              tf.int64,
              tf.int32]

    mean_dtypes = [tf.float32,
                   tf.float64]

    # Each item is np_op1, np_op2, tf_op
    ops_list = [(np.add, None, tf.sparse_segment_sum),
                (self._mean_cum_op, self._mean_reduce_op,
                 tf.sparse_segment_mean)]

    n = 400
    shape = [n, 2]
    segment_indices = []
    for i in range(20):
      for _ in range(i + 1):
        segment_indices.append(i)
    num_indices = len(segment_indices)
    for dtype in dtypes:
      with self.test_session(use_gpu=False):
        tf_indices, np_indices, tf_x, np_x = self._sparse_input(shape,
                                                                num_indices,
                                                                dtype=dtype)
        for np_op1, np_op2, tf_op in ops_list:
          if tf_op == tf.sparse_segment_mean and dtype not in mean_dtypes:
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

  def testGradient(self):
    shape = [10, 4]

    segment_indices = [0, 1, 2, 2]
    num_indices = len(segment_indices)
    for tf_op in [tf.sparse_segment_sum,
                  tf.sparse_segment_mean]:
      with self.test_session():
        tf_indices, _, tf_x, np_x = self._sparse_input(
            shape, num_indices, dtype=tf.float64)
        s = tf_op(data=tf_x, indices=tf_indices, segment_ids=segment_indices)
        jacob_t, jacob_n = gradient_checker.ComputeGradient(
            tf_x, shape, s, [3, 4], x_init_value=np_x.astype(np.double),
            delta=1)
      self.assertAllClose(jacob_t, jacob_n, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
  tf.test.main()
