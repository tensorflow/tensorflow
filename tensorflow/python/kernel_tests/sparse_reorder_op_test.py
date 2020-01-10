"""Tests for SparseReorder."""

import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class SparseReorderTest(tf.test.TestCase):

  def _SparseTensorPlaceholder(self):
    return tf.SparseTensor(
        tf.placeholder(tf.int64),
        tf.placeholder(tf.int32),
        tf.placeholder(tf.int64))

  def _SparseTensorValue_5x6(self, permutation):
    ind = np.array([
        [0, 0],
        [1, 0], [1, 3], [1, 4],
        [3, 2], [3, 3]]).astype(np.int64)
    val = np.array([0, 10, 13, 14, 32, 33]).astype(np.int32)

    ind = ind[permutation]
    val = val[permutation]

    shape = np.array([5, 6]).astype(np.int64)
    return tf.SparseTensorValue(ind, val, shape)

  def testAlreadyInOrder(self):
    with self.test_session(use_gpu=False) as sess:
      sp_input = self._SparseTensorPlaceholder()
      input_val = self._SparseTensorValue_5x6(np.arange(6))
      sp_output = tf.sparse_reorder(sp_input)

      output_val = sess.run(sp_output, {sp_input: input_val})
      self.assertAllEqual(output_val.indices, input_val.indices)
      self.assertAllEqual(output_val.values, input_val.values)
      self.assertAllEqual(output_val.shape, input_val.shape)

  def testOutOfOrder(self):
    expected_output_val = self._SparseTensorValue_5x6(np.arange(6))
    with self.test_session(use_gpu=False) as sess:
      for _ in range(5):  # To test various random permutations
        sp_input = self._SparseTensorPlaceholder()
        input_val = self._SparseTensorValue_5x6(np.random.permutation(6))
        sp_output = tf.sparse_reorder(sp_input)

        output_val = sess.run(sp_output, {sp_input: input_val})
        self.assertAllEqual(output_val.indices, expected_output_val.indices)
        self.assertAllEqual(output_val.values, expected_output_val.values)
        self.assertAllEqual(output_val.shape, expected_output_val.shape)


if __name__ == "__main__":
  tf.test.main()
