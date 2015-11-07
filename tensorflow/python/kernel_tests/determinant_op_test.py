"""Tests for tensorflow.ops.tf.MatrixDeterminant."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class DeterminantOpTest(tf.test.TestCase):

  def _compareDeterminant(self, matrix_x):
    with self.test_session():
      if matrix_x.ndim == 2:
        tf_ans = tf.matrix_determinant(matrix_x)
      else:
        tf_ans = tf.batch_matrix_determinant(matrix_x)
      out = tf_ans.eval()
    shape = matrix_x.shape
    if shape[-1] == 0 and shape[-2] == 0:
      np_ans = np.ones(shape[:-2]).astype(matrix_x.dtype)
    else:
      np_ans = np.array(np.linalg.det(matrix_x)).astype(matrix_x.dtype)
    self.assertAllClose(np_ans, out)
    self.assertShapeEqual(np_ans, tf_ans)

  def testBasic(self):
    # 2x2 matrices
    self._compareDeterminant(np.array([[2., 3.], [3., 4.]]).astype(np.float32))
    self._compareDeterminant(np.array([[0., 0.], [0., 0.]]).astype(np.float32))
    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(np.array(
        [[2., 3., 4., 5., 6.], [3., 4., 9., 2., 0.], [2., 5., 8., 3., 8.],
         [1., 6., 7., 4., 7.], [2., 3., 4., 5., 6.]]).astype(np.float32))
    # A multidimensional batch of 2x2 matrices
    self._compareDeterminant(np.random.rand(3, 4, 5, 2, 2).astype(np.float32))

  def testBasicDouble(self):
    # 2x2 matrices
    self._compareDeterminant(np.array([[2., 3.], [3., 4.]]).astype(np.float64))
    self._compareDeterminant(np.array([[0., 0.], [0., 0.]]).astype(np.float64))
    # 5x5 matrices (Eigen forces LU decomposition)
    self._compareDeterminant(np.array(
        [[2., 3., 4., 5., 6.], [3., 4., 9., 2., 0.], [2., 5., 8., 3., 8.],
         [1., 6., 7., 4., 7.], [2., 3., 4., 5., 6.]]).astype(np.float64))
    # A multidimensional batch of 2x2 matrices
    self._compareDeterminant(np.random.rand(3, 4, 5, 2, 2).astype(np.float64))

  def testOverflow(self):
    max_double = np.finfo("d").max
    huge_matrix = np.array([[max_double, 0.0], [0.0, max_double]])
    with self.assertRaisesOpError("not finite"):
      self._compareDeterminant(huge_matrix)

  def testNonSquareMatrix(self):
    # When the determinant of a non-square matrix is attempted we should return
    # an error
    with self.assertRaises(ValueError):
      tf.matrix_determinant(
          np.array([[1., 2., 3.], [3., 5., 4.]]).astype(np.float32))

  def testWrongDimensions(self):
    # The input to the determinant should be a 2-dimensional tensor.
    tensor1 = tf.constant([1., 2.])
    with self.assertRaises(ValueError):
      tf.matrix_determinant(tensor1)

  def testEmpty(self):
    self._compareDeterminant(np.empty([0, 2, 2]))
    self._compareDeterminant(np.empty([2, 0, 0]))


if __name__ == "__main__":
  tf.test.main()
