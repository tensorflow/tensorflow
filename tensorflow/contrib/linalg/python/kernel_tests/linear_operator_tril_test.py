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

from tensorflow.contrib import linalg as linalg_lib
from tensorflow.contrib.linalg.python.ops import linear_operator_test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

linalg = linalg_lib
random_seed.set_random_seed(23)


class LinearOperatorTriLTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @property
  def _dtypes_to_test(self):
    # TODO(langmore) Test complex types once supported by
    # matrix_triangular_solve.
    return [dtypes.float32, dtypes.float64]

  def _operator_and_mat_and_feed_dict(self, shape, dtype, use_placeholder):
    # Upper triangle will be nonzero, but ignored.
    # Use a diagonal that ensures this matrix is well conditioned.
    tril = linear_operator_test_util.random_tril_matrix(
        shape, dtype=dtype, force_well_conditioned=True, remove_upper=False)

    if use_placeholder:
      tril_ph = array_ops.placeholder(dtype=dtype)
      # Evaluate the tril here because (i) you cannot feed a tensor, and (ii)
      # tril is random and we want the same value used for both mat and
      # feed_dict.
      tril = tril.eval()
      operator = linalg.LinearOperatorTriL(tril_ph)
      feed_dict = {tril_ph: tril}
    else:
      operator = linalg.LinearOperatorTriL(tril)
      feed_dict = None

    mat = array_ops.matrix_band_part(tril, -1, 0)

    return operator, mat, feed_dict

  def test_assert_non_singular(self):
    # Singlular matrix with one positive eigenvalue and one zero eigenvalue.
    with self.test_session():
      tril = [[1., 0.], [1., 0.]]
      operator = linalg.LinearOperatorTriL(tril)
      with self.assertRaisesOpError("Singular operator"):
        operator.assert_non_singular().run()

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues.
    tril = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorTriL(
        tril,
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_tril_must_have_at_least_two_dims_or_raises(self):
    with self.assertRaisesRegexp(ValueError, "at least 2 dimensions"):
      linalg.LinearOperatorTriL([1.])


if __name__ == "__main__":
  test.main()
