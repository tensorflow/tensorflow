# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

# For private members.
from tensorflow.contrib.distributions.python.ops import operator_pd

distributions = tf.contrib.distributions


class OperatorShape(operator_pd.OperatorPDBase):
  """Operator implements the ABC method ._shape."""

  def __init__(self, shape):
    self._stored_shape = shape

  @property
  def verify_pd(self):
    return True

  def get_shape(self):
    return tf.TensorShape(self._stored_shape)

  def _shape(self):
    return tf.shape(np.random.rand(*self._stored_shape))

  @property
  def name(self):
    return "OperatorShape"

  def dtype(self):
    return tf.int32

  @property
  def inputs(self):
    return []


class OperatorSqrtSolve(OperatorShape):
  """Operator implements .sqrt_solve."""

  def __init__(self, chol_array):
    self._chol = tf.convert_to_tensor(chol_array)
    super(OperatorSqrtSolve, self).__init__(chol_array.shape)

  def _sqrt_solve(self, rhs):
    return tf.matrix_triangular_solve(self._chol, rhs, lower=True)

  def _batch_sqrt_solve(self, rhs):
    return tf.matrix_triangular_solve(self._chol, rhs, lower=True)

  def _inv_quadratic_form_on_vectors(self, x):
    return self._iqfov_via_sqrt_solve(x)


class OperatorSolve(OperatorShape):
  """Operator implements .solve."""

  def __init__(self, chol):
    self._pos_def_matrix = tf.matmul(chol, chol, adjoint_b=True)
    super(OperatorSolve, self).__init__(chol.shape)

  def _solve(self, rhs):
    return tf.matrix_solve(self._pos_def_matrix, rhs)

  def _batch_solve(self, rhs):
    return tf.matrix_solve(self._pos_def_matrix, rhs)

  def _inv_quadratic_form_on_vectors(self, x):
    return self._iqfov_via_solve(x)


class OperatorPDBaseTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _random_cholesky_array(self, shape):
    mat = self._rng.rand(*shape)
    chol = distributions.matrix_diag_transform(mat, transform=tf.nn.softplus)
    # Zero the upper triangle because we're using this as a true Cholesky factor
    # in our tests.
    return tf.matrix_band_part(chol, -1, 0).eval()

  def _numpy_inv_quadratic_form_on_vectors(self, chol, x):
    # Numpy works with batches now (calls them "stacks").
    x_expanded = np.expand_dims(x, -1)
    whitened = np.linalg.solve(chol, x_expanded)
    return (whitened**2).sum(axis=-1).sum(axis=-1)

  def testAllShapesMethodsDefinedByTheOneAbstractpropertyShape(self):

    shape = (1, 2, 3, 3)
    with self.test_session():
      operator = OperatorShape(shape)

      self.assertAllEqual(shape, operator.shape().eval())
      self.assertAllEqual(4, operator.rank().eval())
      self.assertAllEqual((1, 2), operator.batch_shape().eval())
      self.assertAllEqual((1, 2, 3), operator.vector_shape().eval())
      self.assertAllEqual(3, operator.vector_space_dimension().eval())

      self.assertEqual(shape, operator.get_shape())
      self.assertEqual((1, 2), operator.get_batch_shape())
      self.assertEqual((1, 2, 3), operator.get_vector_shape())

  def testIqfovXRankSameAsBroadcastRankUsingSqrtSolve(self):
    with self.test_session():
      for batch_shape in [(), (2,)]:
        for k in [1, 3]:

          x_shape = batch_shape + (k,)
          x = self._rng.randn(*x_shape)

          chol_shape = batch_shape + (k, k)
          chol = self._random_cholesky_array(chol_shape)
          operator = OperatorSqrtSolve(chol)
          qf = operator.inv_quadratic_form_on_vectors(x)

          self.assertEqual(batch_shape, qf.get_shape())

          numpy_qf = self._numpy_inv_quadratic_form_on_vectors(chol, x)
          self.assertAllClose(numpy_qf, qf.eval())

  def testIqfovXRankGreaterThanBroadcastRankUsingSqrtSolve(self):
    with self.test_session():
      for batch_shape in [(), (2,), (2, 3)]:
        for k in [1, 4]:

          x_shape = batch_shape + (k,)
          x = self._rng.randn(*x_shape)

          # chol will not have the leading dimension.
          chol_shape = batch_shape[1:] + (k, k)
          chol = self._random_cholesky_array(chol_shape)
          operator = OperatorSqrtSolve(chol)
          qf = operator.inv_quadratic_form_on_vectors(x)
          numpy_qf = self._numpy_inv_quadratic_form_on_vectors(chol, x)

          self.assertEqual(batch_shape, qf.get_shape())
          self.assertAllClose(numpy_qf, qf.eval())

  def testIqfovXRankTwoGreaterThanBroadcastRankUsingSqrtSolve(self):
    with self.test_session():
      for batch_shape in [(2, 3), (2, 3, 4), (2, 3, 4, 5)]:
        for k in [1, 4]:

          x_shape = batch_shape + (k,)
          x = self._rng.randn(*x_shape)

          # chol will not have the leading two dimensions.
          chol_shape = batch_shape[2:] + (k, k)
          chol = self._random_cholesky_array(chol_shape)
          operator = OperatorSqrtSolve(chol)
          qf = operator.inv_quadratic_form_on_vectors(x)
          numpy_qf = self._numpy_inv_quadratic_form_on_vectors(chol, x)

          self.assertEqual(batch_shape, qf.get_shape())
          self.assertAllClose(numpy_qf, qf.eval())

  def testIqfovXRankSameAsBroadcastRankUsingSolve(self):
    with self.test_session():
      for batch_shape in [(), (2,)]:
        for k in [1, 3]:

          x_shape = batch_shape + (k,)
          x = self._rng.randn(*x_shape)

          chol_shape = batch_shape + (k, k)
          chol = self._random_cholesky_array(chol_shape)
          operator = OperatorSolve(chol)
          qf = operator.inv_quadratic_form_on_vectors(x)

          self.assertEqual(batch_shape, qf.get_shape())

          numpy_qf = self._numpy_inv_quadratic_form_on_vectors(chol, x)
          self.assertAllClose(numpy_qf, qf.eval())

  def testIqfovXRankGreaterThanBroadcastRankUsingSolve(self):
    with self.test_session():
      for batch_shape in [(2,), (2, 3)]:
        for k in [1, 4]:

          x_shape = batch_shape + (k,)
          x = self._rng.randn(*x_shape)

          # chol will not have the leading dimension.
          chol_shape = batch_shape[1:] + (k, k)
          chol = self._random_cholesky_array(chol_shape)
          operator = OperatorSolve(chol)
          qf = operator.inv_quadratic_form_on_vectors(x)
          numpy_qf = self._numpy_inv_quadratic_form_on_vectors(chol, x)

          self.assertEqual(batch_shape, qf.get_shape())
          self.assertAllClose(numpy_qf, qf.eval())

  def testIqfovXRankTwoGreaterThanBroadcastRankUsingSolve(self):
    with self.test_session():
      for batch_shape in [(2, 3), (2, 3, 4), (2, 3, 4, 5)]:
        for k in [1, 4]:

          x_shape = batch_shape + (k,)
          x = self._rng.randn(*x_shape)

          # chol will not have the leading two dimensions.
          chol_shape = batch_shape[2:] + (k, k)
          chol = self._random_cholesky_array(chol_shape)
          operator = OperatorSolve(chol)
          qf = operator.inv_quadratic_form_on_vectors(x)
          numpy_qf = self._numpy_inv_quadratic_form_on_vectors(chol, x)

          self.assertEqual(batch_shape, qf.get_shape())
          self.assertAllClose(numpy_qf, qf.eval())


class FlipMatrixToVectorTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState()

  def testMatrixAndVectorBatchShapesTheSame(self):
    batch_shape = [6, 2, 3]
    for static_batch_shape in [
        tf.TensorShape(batch_shape), tf.TensorShape(None)]:
      with self.test_session():
        mat = self._rng.rand(2, 3, 4, 6)
        vec = operator_pd.flip_matrix_to_vector(
            mat, batch_shape, static_batch_shape)
        vec_v = vec.eval()
        self.assertAllEqual((6, 2, 3, 4), vec_v.shape)
        self.assertAllEqual(mat[1, 2, 3, 4], vec_v[4, 1, 2, 3])

  def testMatrixAndVectorBatchShapesSameRankButPermuted(self):
    batch_shape = [6, 3, 2]
    for static_batch_shape in [
        tf.TensorShape(batch_shape), tf.TensorShape(None)]:
      with self.test_session():
        mat = self._rng.rand(2, 3, 4, 6)
        vec = operator_pd.flip_matrix_to_vector(
            mat, batch_shape, static_batch_shape)
        vec_v = vec.eval()
        self.assertAllEqual((6, 3, 2, 4), vec_v.shape)

  def testVectorBatchShapeLongerThanMatrixBatchShape(self):
    batch_shape = [2, 3, 2, 3]
    for static_batch_shape in [
        tf.TensorShape(batch_shape), tf.TensorShape(None)]:
      with self.test_session():
        mat = self._rng.rand(2, 3, 4, 6)
        vec = operator_pd.flip_matrix_to_vector(
            mat, batch_shape, static_batch_shape)
        vec_v = vec.eval()
        self.assertAllEqual((2, 3, 2, 3, 4), vec_v.shape)

  def testMatrixBatchShapeHasASingletonThatVecBatchShapeDoesnt(self):
    batch_shape = [6, 3]
    for static_batch_shape in [
        tf.TensorShape(batch_shape), tf.TensorShape(None)]:
      with self.test_session():
        mat = self._rng.rand(1, 3, 4, 6)
        vec = operator_pd.flip_matrix_to_vector(
            mat, batch_shape, static_batch_shape)
        vec_v = vec.eval()
        self.assertAllEqual((6, 3, 4), vec_v.shape)
        self.assertAllEqual(mat[0, 2, 3, 4], vec_v[4, 2, 3])


class FlipVectorToMatrixTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState()

  def testWhenXBatchRankIsSameAsBatchRankArg(self):
    batch_shape = [4, 5]
    x = self._rng.rand(4, 5, 6)
    for static_batch_shape in [
        tf.TensorShape(batch_shape), tf.TensorShape(None)]:
      with self.test_session():
        mat = operator_pd.flip_vector_to_matrix(
            x, batch_shape, static_batch_shape)
        mat_v = mat.eval()
        expected_mat_v = x.reshape(x.shape + (1,))
        self.assertAllEqual(expected_mat_v, mat_v)

  def testWhenXHasOneLargerLargerBatchRankThanBatchRankArg(self):
    batch_shape = [4, 5]
    x = self._rng.rand(3, 4, 5, 6)
    for static_batch_shape in [
        tf.TensorShape(batch_shape), tf.TensorShape(None)]:
      with self.test_session():
        mat = operator_pd.flip_vector_to_matrix(
            x, batch_shape, static_batch_shape)
        mat_v = mat.eval()
        self.assertAllEqual((4, 5, 6, 3), mat_v.shape)
        self.assertAllEqual(x[2, 2, 2, 1], mat_v[2, 2, 1, 2])

  def testWhenBatchShapeRequiresReshapeOfVectorBatchShape(self):
    batch_shape = [5, 4]
    x = self._rng.rand(3, 4, 5, 6)  # Note x has (4,5) and batch_shape is (5, 4)
    for static_batch_shape in [
        tf.TensorShape(batch_shape), tf.TensorShape(None)]:
      with self.test_session():
        mat = operator_pd.flip_vector_to_matrix(
            x, batch_shape, static_batch_shape)
        mat_v = mat.eval()
        self.assertAllEqual((5, 4, 6, 3), mat_v.shape)

  def testWhenXHasTwoLargerLargerBatchRankThanBatchRankArg(self):
    batch_shape = [4, 5]
    x = self._rng.rand(2, 3, 4, 5, 6)
    for static_batch_shape in [
        tf.TensorShape(batch_shape), tf.TensorShape(None)]:
      with self.test_session():
        mat = operator_pd.flip_vector_to_matrix(
            x, batch_shape, static_batch_shape)
        mat_v = mat.eval()
        self.assertAllEqual((4, 5, 6, 2*3), mat_v.shape)


class ExtractBatchShapeTest(tf.test.TestCase):

  def setUp(self):
    self._rng = np.random.RandomState()

  def testXHasEmptyBatchShape(self):
    with self.test_session():
      x = self._rng.rand(2, 3)
      num_event_dims = 2
      batch_shape = operator_pd.extract_batch_shape(x, num_event_dims)
      self.assertAllEqual([], batch_shape.eval())

  def testXHasNonEmptyBatchShape(self):
    with self.test_session():
      x = self._rng.rand(2, 3, 4, 5)
      num_event_dims = 2
      batch_shape = operator_pd.extract_batch_shape(x, num_event_dims)
      self.assertAllEqual([2, 3], batch_shape.eval())


if __name__ == "__main__":
  tf.test.main()
