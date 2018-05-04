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

from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib
random_seed.set_random_seed(23)


class LinearOperatorDiagTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def _operator_and_mat_and_feed_dict(self, build_info, dtype, use_placeholder):
    shape = list(build_info.shape)
    diag = linear_operator_test_util.random_sign_uniform(
        shape[:-1], minval=1., maxval=2., dtype=dtype)
    if use_placeholder:
      diag_ph = array_ops.placeholder(dtype=dtype)
      # Evaluate the diag here because (i) you cannot feed a tensor, and (ii)
      # diag is random and we want the same value used for both mat and
      # feed_dict.
      diag = diag.eval()
      operator = linalg.LinearOperatorDiag(diag_ph)
      feed_dict = {diag_ph: diag}
    else:
      operator = linalg.LinearOperatorDiag(diag)
      feed_dict = None

    mat = array_ops.matrix_diag(diag)

    return operator, mat, feed_dict

  def test_assert_positive_definite_raises_for_zero_eigenvalue(self):
    # Matrix with one positive eigenvalue and one zero eigenvalue.
    with self.test_session():
      diag = [1.0, 0.0]
      operator = linalg.LinearOperatorDiag(diag)

      # is_self_adjoint should be auto-set for real diag.
      self.assertTrue(operator.is_self_adjoint)
      with self.assertRaisesOpError("non-positive.*not positive definite"):
        operator.assert_positive_definite().run()

  def test_assert_positive_definite_raises_for_negative_real_eigvalues(self):
    with self.test_session():
      diag_x = [1.0, -2.0]
      diag_y = [0., 0.]  # Imaginary eigenvalues should not matter.
      diag = math_ops.complex(diag_x, diag_y)
      operator = linalg.LinearOperatorDiag(diag)

      # is_self_adjoint should not be auto-set for complex diag.
      self.assertTrue(operator.is_self_adjoint is None)
      with self.assertRaisesOpError("non-positive real.*not positive definite"):
        operator.assert_positive_definite().run()

  def test_assert_positive_definite_does_not_raise_if_pd_and_complex(self):
    with self.test_session():
      x = [1., 2.]
      y = [1., 0.]
      diag = math_ops.complex(x, y)  # Re[diag] > 0.
      # Should not fail
      linalg.LinearOperatorDiag(diag).assert_positive_definite().run()

  def test_assert_non_singular_raises_if_zero_eigenvalue(self):
    # Singlular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.test_session():
      diag = [1.0, 0.0]
      operator = linalg.LinearOperatorDiag(diag, is_self_adjoint=True)
      with self.assertRaisesOpError("Singular operator"):
        operator.assert_non_singular().run()

  def test_assert_non_singular_does_not_raise_for_complex_nonsingular(self):
    with self.test_session():
      x = [1., 0.]
      y = [0., 1.]
      diag = math_ops.complex(x, y)
      # Should not raise.
      linalg.LinearOperatorDiag(diag).assert_non_singular().run()

  def test_assert_self_adjoint_raises_if_diag_has_complex_part(self):
    with self.test_session():
      x = [1., 0.]
      y = [0., 1.]
      diag = math_ops.complex(x, y)
      operator = linalg.LinearOperatorDiag(diag)
      with self.assertRaisesOpError("imaginary.*not self-adjoint"):
        operator.assert_self_adjoint().run()

  def test_assert_self_adjoint_does_not_raise_for_diag_with_zero_imag(self):
    with self.test_session():
      x = [1., 0.]
      y = [0., 0.]
      diag = math_ops.complex(x, y)
      operator = linalg.LinearOperatorDiag(diag)
      # Should not raise
      operator.assert_self_adjoint().run()

  def test_scalar_diag_raises(self):
    with self.assertRaisesRegexp(ValueError, "must have at least 1 dimension"):
      linalg.LinearOperatorDiag(1.)

  def test_broadcast_matmul_and_solve(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.matmul cannot handle.
    # In particular, tf.matmul does not broadcast.
    with self.test_session() as sess:
      x = random_ops.random_normal(shape=(2, 2, 3, 4))

      # This LinearOperatorDiag will be broadcast to (2, 2, 3, 3) during solve
      # and matmul with 'x' as the argument.
      diag = random_ops.random_uniform(shape=(2, 1, 3))
      operator = linalg.LinearOperatorDiag(diag, is_self_adjoint=True)
      self.assertAllEqual((2, 1, 3, 3), operator.shape)

      # Create a batch matrix with the broadcast shape of operator.
      diag_broadcast = array_ops.concat((diag, diag), 1)
      mat = array_ops.matrix_diag(diag_broadcast)
      self.assertAllEqual((2, 2, 3, 3), mat.get_shape())  # being pedantic.

      operator_matmul = operator.matmul(x)
      mat_matmul = math_ops.matmul(mat, x)
      self.assertAllEqual(operator_matmul.get_shape(), mat_matmul.get_shape())
      self.assertAllClose(*sess.run([operator_matmul, mat_matmul]))

      operator_solve = operator.solve(x)
      mat_solve = linalg_ops.matrix_solve(mat, x)
      self.assertAllEqual(operator_solve.get_shape(), mat_solve.get_shape())
      self.assertAllClose(*sess.run([operator_solve, mat_solve]))


if __name__ == "__main__":
  test.main()
