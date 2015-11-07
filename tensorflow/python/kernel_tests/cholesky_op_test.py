"""Tests for tensorflow.ops.tf.Cholesky."""
import tensorflow.python.platform

import numpy as np
import tensorflow as tf


class CholeskyOpTest(tf.test.TestCase):

  def _verifyCholesky(self, x):
    with self.test_session() as sess:
      # Verify that LL^T == x.
      if x.ndim == 2:
        chol = tf.cholesky(x)
        verification = tf.matmul(chol,
                                 chol,
                                 transpose_a=False,
                                 transpose_b=True)
      else:
        chol = tf.batch_cholesky(x)
        verification = tf.batch_matmul(chol, chol, adj_x=False, adj_y=True)
      chol_np, verification_np = sess.run([chol, verification])
    self.assertAllClose(x, verification_np)
    self.assertShapeEqual(x, chol)
    # Check that the cholesky is lower triangular, and has positive diagonal
    # elements.
    if chol_np.shape[-1] > 0:
      chol_reshaped = np.reshape(chol_np, (-1, chol_np.shape[-2],
                                           chol_np.shape[-1]))
      for chol_matrix in chol_reshaped:
        self.assertAllClose(chol_matrix, np.tril(chol_matrix))
        self.assertTrue((np.diag(chol_matrix) > 0.0).all())

  def testBasic(self):
    self._verifyCholesky(np.array([[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]))

  def testBatch(self):
    simple_array = np.array([[[1., 0.], [0., 5.]]])  # shape (1, 2, 2)
    self._verifyCholesky(simple_array)
    self._verifyCholesky(np.vstack((simple_array, simple_array)))
    odd_sized_array = np.array([[[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]])
    self._verifyCholesky(np.vstack((odd_sized_array, odd_sized_array)))

    # Generate random positive-definite matrices.
    matrices = np.random.rand(10, 5, 5)
    for i in xrange(10):
      matrices[i] = np.dot(matrices[i].T, matrices[i])
    self._verifyCholesky(matrices)

  def testNonSquareMatrix(self):
    with self.assertRaises(ValueError):
      tf.cholesky(np.array([[1., 2., 3.], [3., 4., 5.]]))

  def testWrongDimensions(self):
    tensor3 = tf.constant([1., 2.])
    with self.assertRaises(ValueError):
      tf.cholesky(tensor3)

  def testNotInvertible(self):
    # The input should be invertible.
    with self.test_session():
      with self.assertRaisesOpError("LLT decomposition was not successful. The "
                                    "input might not be valid."):
        # All rows of the matrix below add to zero
        self._verifyCholesky(np.array([[1., -1., 0.], [-1., 1., -1.], [0., -1.,
                                                                       1.]]))

  def testEmpty(self):
    self._verifyCholesky(np.empty([0, 2, 2]))
    self._verifyCholesky(np.empty([2, 0, 0]))


if __name__ == "__main__":
  tf.test.main()
