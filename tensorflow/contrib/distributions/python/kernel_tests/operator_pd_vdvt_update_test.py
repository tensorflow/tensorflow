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

from tensorflow.contrib.distributions.python.ops import operator_pd_full
from tensorflow.contrib.distributions.python.ops import operator_pd_vdvt_update
from tensorflow.contrib.distributions.python.ops import operator_test_util

distributions = tf.contrib.distributions


class OperatorPDSqrtVDVTUpdateTest(
    operator_test_util.OperatorPDDerivedClassTest):
  """Most tests done in the base class."""
  _diag_is_none = False

  def setUp(self):
    self._rng = np.random.RandomState(42)

  def _random_pd_matrix(self, shape):
    # With probability 1 this is positive definite.
    sqrt = self._rng.randn(*shape)
    mat = tf.batch_matmul(sqrt, sqrt, adj_y=True)
    return mat.eval()

  def _random_v_and_diag(self, mat_shape, v_matrix_rank):
    # Get the necessary elements to make the sqrt update.
    mat_shape = list(mat_shape)
    batch_shape = mat_shape[:-2]
    diag_shape = mat_shape[:-2] + [v_matrix_rank]
    k = mat_shape[-1]
    assert k == mat_shape[-2], "Must be a square matrix"
    v_shape = batch_shape + [k, v_matrix_rank]
    v = self._rng.randn(*v_shape)  # anything goes with "v"!

    if self._diag_is_none:
      diag = None
    else:
      diag = self._rng.rand(*diag_shape) + 0.1  # Positive diag!
    return v, diag

  def _updated_mat(self, mat, v, diag):
    # Get dense matrix defined by its square root, which is an update of `mat`:
    # A = (mat + v D v^T) (mat + v D v^T)^T
    # D is the diagonal matrix with `diag` on the diagonal.

    # If diag is None, then it defaults to the identity matrix, so DV^T = V^T
    if diag is None:
      diag_vt = tf.matrix_transpose(v)
    else:
      diag_mat = tf.matrix_diag(diag)
      diag_vt = tf.batch_matmul(diag_mat, v, adj_y=True)

    v_diag_vt = tf.batch_matmul(v, diag_vt)
    sqrt = mat + v_diag_vt
    a = tf.batch_matmul(sqrt, sqrt, adj_y=True)
    return a.eval()

  def _build_operator_and_mat(self, batch_shape, k, dtype=np.float64):
    """This method is called by base class, enabling many standard tests."""
    # Create a matrix then explicitly update it with v and diag.
    # Create an OperatorPDSqrtVDVTUpdate from the matrix and v and diag
    # The operator should have the same behavior.
    #
    # The low-rank matrix V will have rank 1/2 of k, unless k is 1, in which
    # case it will be 1 as well.
    if k == 1:
      v_matrix_rank = k
    else:
      v_matrix_rank = k // 2
    mat_shape = list(batch_shape) + [k, k]
    mat = self._random_pd_matrix(mat_shape)
    v, diag = self._random_v_and_diag(mat_shape, v_matrix_rank)

    # Set dtypes
    mat = mat.astype(dtype)
    v = v.astype(dtype)
    if diag is not None:
      diag = diag.astype(dtype)

    # The matrix: (mat + v*diag*v^T) * (mat + v*diag*v^T)^T
    # Our final updated operator should behave like this.
    updated_mat = self._updated_mat(mat, v, diag)

    # Represents the matrix: `mat`, before updating.
    # This is the Operator that we will update.
    o_made_with_mat = operator_pd_full.OperatorPDFull(mat)

    # Represents the matrix: (mat + v*diag*v^T) * (mat + v*diag*v^T)^T,
    # achieved by updating the operator "o_made_with_mat".
    # This is the operator we're testing.
    operator = operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(
        o_made_with_mat, v, diag)

    return operator, updated_mat

  def test_to_dense_placeholder(self):
    # Test simple functionality when the inputs are placeholders.
    mat_shape = [3, 3]
    v_matrix_rank = 2
    with self.test_session():
      # Make an OperatorPDFull with a matrix placeholder.
      mat_ph = tf.placeholder(tf.float64, name="mat_ph")
      mat = self._random_pd_matrix(mat_shape)
      o_made_with_mat = operator_pd_full.OperatorPDFull(mat_ph)

      # Make the placeholders and arrays for the updated operator.
      v_ph = tf.placeholder(tf.float64, name="v_ph")
      v, diag = self._random_v_and_diag(mat_shape, v_matrix_rank)
      if self._diag_is_none:
        diag_ph = None
        feed_dict = {v_ph: v, mat_ph: mat}
      else:
        diag_ph = tf.placeholder(tf.float64, name="diag_ph")
        feed_dict = {v_ph: v, diag_ph: diag, mat_ph: mat}

      # Make the OperatorPDSqrtVDVTUpdate with v and diag placeholders.
      operator = operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(
          o_made_with_mat, v_ph, diag=diag_ph)

      # Should not fail
      operator.to_dense().eval(feed_dict=feed_dict)
      operator.log_det().eval(feed_dict=feed_dict)

  def test_operator_not_subclass_of_operator_pd_raises(self):
    # We enforce that `operator` is an `OperatorPDBase`.
    with self.test_session():
      v, diag = self._random_v_and_diag((3, 3), 2)
      operator_m = "I am not a subclass of OperatorPDBase"

      with self.assertRaisesRegexp(TypeError, "not instance"):
        operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(operator_m, v, diag)

  def test_non_pos_def_diag_raises(self):
    if self._diag_is_none:
      return
    # We enforce that the diag is positive definite.
    with self.test_session():
      matrix_shape = (3, 3)
      v_rank = 2
      v, diag = self._random_v_and_diag(matrix_shape, v_rank)
      mat = self._random_pd_matrix(matrix_shape)
      diag[0] = 0.0

      operator_m = operator_pd_full.OperatorPDFull(mat)
      operator = operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(
          operator_m, v, diag)

      with self.assertRaisesOpError("positive"):
        operator.to_dense().eval()

  def test_non_pos_def_diag_doesnt_raise_if_verify_pd_false(self):
    # We enforce that the diag is positive definite.
    if self._diag_is_none:
      return
    with self.test_session():
      matrix_shape = (3, 3)
      v_rank = 2
      v, diag = self._random_v_and_diag(matrix_shape, v_rank)
      mat = self._random_pd_matrix(matrix_shape)
      diag[0] = 0.0

      operator_m = operator_pd_full.OperatorPDFull(mat)
      operator = operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(
          operator_m, v, diag, verify_pd=False)

      operator.to_dense().eval()  # Should not raise.

  def test_event_shape_mismatch_v_and_diag_raises_static(self):
    v = self._rng.rand(4, 3, 2)
    diag = self._rng.rand(4, 1)  # Should be shape (4, 2,) to match v.
    with self.test_session():

      mat = self._random_pd_matrix((4, 3, 3))  # mat and v match
      operator_m = operator_pd_full.OperatorPDFull(mat)
      with self.assertRaisesRegexp(ValueError, "diag.*v.*last dimension"):
        operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(operator_m, v, diag)

  def test_batch_shape_mismatch_v_and_diag_raises_static(self):
    v = self._rng.rand(4, 3, 2)
    diag = self._rng.rand(5, 1)  # Should be shape (4, 2,) to match v.
    with self.test_session():

      mat = self._random_pd_matrix((4, 3, 3))  # mat and v match
      operator_m = operator_pd_full.OperatorPDFull(mat)
      with self.assertRaisesRegexp(ValueError, "diag.*batch shape"):
        operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(operator_m, v, diag)

  def test_tensor_rank_shape_mismatch_v_and_diag_raises_static(self):
    v = self._rng.rand(1, 2, 2, 2)
    diag = self._rng.rand(5, 1)  # Should have rank 1 less than v.
    with self.test_session():

      mat = self._random_pd_matrix((1, 2, 2, 2))  # mat and v match
      operator_m = operator_pd_full.OperatorPDFull(mat)
      with self.assertRaisesRegexp(ValueError, "diag.*rank"):
        operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(operator_m, v, diag)

  def test_event_shape_mismatch_v_and_diag_raises_dynamic(self):
    with self.test_session():

      v = self._rng.rand(4, 3, 2)
      diag = self._rng.rand(4, 1)  # Should be shape (4, 2,) to match v.
      mat = self._random_pd_matrix((4, 3, 3))  # mat and v match

      v_ph = tf.placeholder(tf.float32, name="v_ph")
      diag_ph = tf.placeholder(tf.float32, name="diag_ph")
      mat_ph = tf.placeholder(tf.float32, name="mat_ph")

      operator_m = operator_pd_full.OperatorPDFull(mat_ph)
      updated = operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(
          operator_m, v_ph, diag_ph)
      with self.assertRaisesOpError("x == y"):
        updated.to_dense().eval(feed_dict={v_ph: v, diag_ph: diag, mat_ph: mat})

  def test_batch_shape_mismatch_v_and_diag_raises_dynamic(self):
    with self.test_session():
      v = self._rng.rand(4, 3, 2)
      diag = self._rng.rand(5, 1)  # Should be shape (4, 2,) to match v.
      mat = self._random_pd_matrix((4, 3, 3))  # mat and v match

      v_ph = tf.placeholder(tf.float32, name="v_ph")
      diag_ph = tf.placeholder(tf.float32, name="diag_ph")
      mat_ph = tf.placeholder(tf.float32, name="mat_ph")

      operator_m = operator_pd_full.OperatorPDFull(mat_ph)
      updated = operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(
          operator_m, v_ph, diag_ph)
      with self.assertRaisesOpError("x == y"):
        updated.to_dense().eval(feed_dict={v_ph: v, diag_ph: diag, mat_ph: mat})

  def test_tensor_rank_shape_mismatch_v_and_diag_raises_dynamic(self):
    with self.test_session():

      v = self._rng.rand(2, 2, 2, 2)
      diag = self._rng.rand(2, 2)  # Should have rank 1 less than v.
      mat = self._random_pd_matrix((2, 2, 2, 2))  # mat and v match

      v_ph = tf.placeholder(tf.float32, name="v_ph")
      diag_ph = tf.placeholder(tf.float32, name="diag_ph")
      mat_ph = tf.placeholder(tf.float32, name="mat_ph")

      operator_m = operator_pd_full.OperatorPDFull(mat_ph)
      updated = operator_pd_vdvt_update.OperatorPDSqrtVDVTUpdate(
          operator_m, v_ph, diag_ph)
      with self.assertRaisesOpError("rank"):
        updated.to_dense().eval(feed_dict={v_ph: v, diag_ph: diag, mat_ph: mat})


class OperatorPDSqrtVDVTUpdateNoneDiagTest(OperatorPDSqrtVDVTUpdateTest):
  _diag_is_none = True


if __name__ == "__main__":
  tf.test.main()
