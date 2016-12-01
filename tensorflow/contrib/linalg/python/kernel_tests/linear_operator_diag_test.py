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

import tensorflow as tf

from tensorflow.contrib.linalg.python.ops import linear_operator_test_util


linalg = tf.contrib.linalg
tf.set_random_seed(23)


class LinearOperatorDiagtest(
    linear_operator_test_util.LinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @property
  def _dtypes_to_test(self):
    return [tf.float32, tf.float64]

  @property
  def _shapes_to_test(self):
    # non-batch operators (n, n) and batch operators.
    return [(0, 0), (1, 1), (1, 3, 3), (3, 2, 2), (2, 1, 3, 3)]

  def _make_rhs(self, operator):
    # This operator is square, so rhs and x will have same shape.
    return self._make_x(operator)

  def _make_x(self, operator):
    # Return the number of systems to solve, R, equal to 1 or 2.
    r = self._get_num_systems(operator)
    # If operator.shape = [B1,...,Bb, N, N] this returns a random matrix of
    # shape [B1,...,Bb, N, R], R = 1 or 2.
    if operator.shape.is_fully_defined():
      batch_shape = operator.batch_shape.as_list()
      n = operator.domain_dimension.value
      rhs_shape = batch_shape + [n, r]
    else:
      batch_shape = operator.batch_shape_dynamic()
      n = operator.domain_dimension_dynamic()
      rhs_shape = tf.concat(0, (batch_shape, [n, r]))
    return tf.random_normal(shape=rhs_shape, dtype=operator.dtype)

  def _get_num_systems(self, operator):
    """Get some number, either 1 or 2, depending on operator."""
    if operator.tensor_rank is None or operator.tensor_rank % 2:
      return 1
    else:
      return 2

  def _operator_and_mat_and_feed_dict(self, shape, dtype, use_placeholder):
    shape = list(shape)
    diag_shape = shape[:-1]

    diag = tf.random_normal(diag_shape, dtype=dtype)
    diag_ph = tf.placeholder(dtype=dtype)

    if use_placeholder:
      # Evaluate the diag here because (i) you cannot feed a tensor, and (ii)
      # diag is random and we want the same value used for both mat and
      # feed_dict.
      diag = diag.eval()
      mat = tf.matrix_diag(diag)
      operator = linalg.LinearOperatorDiag(diag_ph)
      feed_dict = {diag_ph: diag}
    else:
      mat = tf.matrix_diag(diag)
      operator = linalg.LinearOperatorDiag(diag)
      feed_dict = None

    return operator, mat, feed_dict

  def test_assert_positive_definite(self):
    # Singlular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.test_session():
      diag = [1.0, -1.0]
      operator = linalg.LinearOperatorDiag(diag)
      with self.assertRaisesOpError("was not positive definite"):
        operator.assert_positive_definite().run()

  def test_assert_non_singular(self):
    # Singlular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.test_session():
      diag = [1.0, 0.0]
      operator = linalg.LinearOperatorDiag(diag)
      with self.assertRaisesOpError("Singular operator"):
        operator.assert_non_singular().run()

  def test_broadcast_apply_and_solve(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.test_session() as sess:
      x = tf.random_normal(shape=(2, 2, 3, 4))

      # This LinearOperatorDiag will be brodacast to (2, 2, 3, 3) during solve
      # and apply with 'x' as the argument.
      diag = tf.random_uniform(shape=(2, 1, 3))
      operator = linalg.LinearOperatorDiag(diag)
      self.assertAllEqual((2, 1, 3, 3), operator.shape)

      # Create a batch matrix with the broadcast shape of operator.
      diag_broadcast = tf.concat(1, (diag, diag))
      mat = tf.matrix_diag(diag_broadcast)
      self.assertAllEqual((2, 2, 3, 3), mat.get_shape())  # being pedantic.

      operator_apply = operator.apply(x)
      mat_apply = tf.batch_matmul(mat, x)
      self.assertAllEqual(operator_apply.get_shape(), mat_apply.get_shape())
      self.assertAllClose(*sess.run([operator_apply, mat_apply]))

      operator_solve = operator.solve(x)
      mat_solve = tf.matrix_solve(mat, x)
      self.assertAllEqual(operator_solve.get_shape(), mat_solve.get_shape())
      self.assertAllClose(*sess.run([operator_solve, mat_solve]))


if __name__ == "__main__":
  tf.test.main()
