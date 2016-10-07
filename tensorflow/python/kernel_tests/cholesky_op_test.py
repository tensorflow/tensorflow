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
"""Tests for tensorflow.ops.tf.Cholesky."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


class CholeskyOpTest(tf.test.TestCase):

  def _verifyCholeskyBase(self, sess, x, chol, verification):
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

  def _verifyCholesky(self, x):
    # Verify that LL^T == x.
    with self.test_session() as sess:
      chol = tf.cholesky(x)
      verification = tf.batch_matmul(chol, chol, adj_x=False, adj_y=True)
      self._verifyCholeskyBase(sess, x, chol, verification)

  def testBasic(self):
    self._verifyCholesky(np.array([[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]))

  def testBatch(self):
    simple_array = np.array([[[1., 0.], [0., 5.]]])  # shape (1, 2, 2)
    self._verifyCholesky(simple_array)
    self._verifyCholesky(np.vstack((simple_array, simple_array)))
    odd_sized_array = np.array([[[4., -1., 2.], [-1., 6., 0], [2., 0., 5.]]])
    self._verifyCholesky(np.vstack((odd_sized_array, odd_sized_array)))

    #  Generate random positive-definite matrices.
    matrices = np.random.rand(10, 5, 5)
    for i in xrange(10):
      matrices[i] = np.dot(matrices[i].T, matrices[i])
    self._verifyCholesky(matrices)

  def testNonSquareMatrix(self):
    with self.assertRaises(ValueError):
      tf.cholesky(np.array([[1., 2., 3.], [3., 4., 5.]]))
    with self.assertRaises(ValueError):
      tf.cholesky(
          np.array([[[1., 2., 3.], [3., 4., 5.]], [[1., 2., 3.], [3., 4., 5.]]
                   ]))

  def testWrongDimensions(self):
    tensor3 = tf.constant([1., 2.])
    with self.assertRaises(ValueError):
      tf.cholesky(tensor3)
    with self.assertRaises(ValueError):
      tf.cholesky(tensor3)

  def testNotInvertible(self):
    # The input should be invertible.
    with self.test_session():
      with self.assertRaisesOpError("LLT decomposition was not successful. The"
                                    " input might not be valid."):
        # All rows of the matrix below add to zero
        self._verifyCholesky(
            np.array([[1., -1., 0.], [-1., 1., -1.], [0., -1., 1.]]))

  def testEmpty(self):
    self._verifyCholesky(np.empty([0, 2, 2]))
    self._verifyCholesky(np.empty([2, 0, 0]))


class CholeskyGradTest(tf.test.TestCase):
  _backprop_block_size = 32

  def getShapes(self, shapeList):
    return ((elem, int(np.floor(1.2 * elem))) for elem in shapeList)

  def testSmallMatrices(self):
    np.random.seed(0)
    shapes = self.getShapes([1, 2, 10])
    self.runFiniteDifferences(shapes)

  def testOneBlockMatrices(self):
    np.random.seed(0)
    shapes = self.getShapes([self._backprop_block_size + 1])
    self.runFiniteDifferences(
        shapes, dtypes=(tf.float32, tf.float64), scalarTest=True)

  def testTwoBlockMatrixFloat(self):
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(shapes, dtypes=(tf.float32,), scalarTest=True)

  def testTwoBlockMatrixDouble(self):
    np.random.seed(0)
    shapes = self.getShapes([2 * self._backprop_block_size + 1])
    self.runFiniteDifferences(shapes, dtypes=(tf.float64,), scalarTest=True)

  def runFiniteDifferences(self,
                           shapes,
                           dtypes=(tf.float32, tf.float64),
                           scalarTest=False):
    with self.test_session(use_gpu=False):
      for shape in shapes:
        for batch in False, True:
          for dtype in dtypes:
            if not scalarTest:
              x = tf.constant(np.random.randn(shape[0], shape[1]), dtype)
              tensor = tf.matmul(x, tf.transpose(x)) / shape[0]
            else:
              # This is designed to be a faster test for larger matrices.
              x = tf.constant(np.random.randn(), dtype)
              R = tf.constant(np.random.randn(shape[0], shape[1]), dtype)
              e = tf.mul(R, x)
              tensor = tf.matmul(e, tf.transpose(e)) / shape[0]

            # Inner-most matrices in tensor are positive definite.
            if batch:
              tensor = tf.tile(tf.expand_dims(tensor, 0), [4, 1, 1])
            y = tf.cholesky(tensor)
            if scalarTest:
              y = tf.reduce_mean(y)
            error = tf.test.compute_gradient_error(x, x._shape_as_list(), y,
                                                   y._shape_as_list())
            tf.logging.info("error = %f", error)
            if dtype == tf.float64:
              self.assertLess(error, 1e-5)
            else:
              self.assertLess(error, 3e-3)


if __name__ == "__main__":
  tf.test.main()
