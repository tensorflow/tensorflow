"""Tests for tensorflow.ops.math_ops.matrix_inverse."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class InverseOpTest(tf.test.TestCase):

  def _verifyInverse(self, x):
    for np_type in [np.float32, np.float64]:
      y = x.astype(np_type)
      with self.test_session():
        # Verify that x^{-1} * x == Identity matrix.
        if x.ndim == 2:
          inv = tf.matrix_inverse(y)
          tf_ans = tf.matmul(inv, y)
          np_ans = np.identity(y.shape[-1])
        else:
          inv = tf.batch_matrix_inverse(y)
          tf_ans = tf.batch_matmul(inv, y)
          tiling = list(y.shape)
          tiling[-2:] = [1, 1]
          np_ans = np.tile(np.identity(y.shape[-1]), tiling)
        out = tf_ans.eval()
      self.assertAllClose(np_ans, out)
      self.assertShapeEqual(y, tf_ans)

  def testBasic(self):
    # 2x2 matrices
    matrix1 = np.array([[1., 2.], [3., 4.]])
    matrix2 = np.array([[1., 3.], [3., 5.]])
    self._verifyInverse(matrix1)
    self._verifyInverse(matrix2)
    # A multidimensional batch of 2x2 matrices
    matrix_batch = np.concatenate([np.expand_dims(matrix1, 0),
                                   np.expand_dims(matrix2, 0)])
    matrix_batch = np.tile(matrix_batch, [2, 3, 1, 1])
    self._verifyInverse(matrix_batch)

  def testNonSquareMatrix(self):
    # When the inverse of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises(ValueError):
      tf.matrix_inverse(np.array([[1., 2., 3.], [3., 4., 5.]]))

  def testWrongDimensions(self):
    # The input to the inverse should be at least a 2-dimensional tensor.
    tensor3 = tf.constant([1., 2.])
    with self.assertRaises(ValueError):
      tf.matrix_inverse(tensor3)

  def testNotInvertible(self):
    # The input should be invertible.
    with self.test_session():
      with self.assertRaisesOpError("Input is not invertible."):
        # All rows of the matrix below add to zero
        tensor3 = tf.constant([[1., 0., -1.], [-1., 1., 0.], [0., -1., 1.]])
        tf.matrix_inverse(tensor3).eval()

    with self.test_session():
      with self.assertRaisesOpError("Input is not invertible."):
        # Determinant of the matrix below is zero
        tensor3 = tf.constant([[1., 1.], [1., 1.]])
        tf.matrix_inverse(tensor3).eval()

    with self.test_session():
      with self.assertRaisesOpError("Input is not invertible."):
        # Determinant of the matrix below is zero
        tensor3 = tf.constant([[np.inf, 1.], [1., 1.]])
        tf.matrix_inverse(tensor3).eval()

  def testEmpty(self):
    self._verifyInverse(np.empty([0, 2, 2]))
    self._verifyInverse(np.empty([2, 0, 0]))


if __name__ == "__main__":
  tf.test.main()
