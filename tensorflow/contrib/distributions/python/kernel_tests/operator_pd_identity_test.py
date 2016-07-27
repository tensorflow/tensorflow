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

from tensorflow.contrib.distributions.python.ops import operator_pd_identity
from tensorflow.contrib.distributions.python.ops import operator_test_util

distributions = tf.contrib.distributions


class OperatorPDIdentityTest(operator_test_util.OperatorPDDerivedClassTest):
  """Most tests done in the base class."""

  def _build_operator_and_mat(self, batch_shape, k, dtype=np.float64):
    # Build an identity matrix with right shape and dtype.
    # Build an operator that should act the same way.
    batch_shape = list(batch_shape)
    diag_shape = batch_shape + [k]
    matrix_shape = batch_shape + [k, k]
    diag = tf.ones(diag_shape, dtype=dtype)
    identity_matrix = tf.batch_matrix_diag(diag)
    operator = operator_pd_identity.OperatorPDIdentity(matrix_shape, dtype)
    return operator, identity_matrix.eval()

  def test_bad_dtype_args_raise(self):
    dtype = np.float32
    batch_shape = [2, 3]
    k = 4
    with self.test_session():
      operator, _ = self._build_operator_and_mat(batch_shape, k, dtype=dtype)

      x_good_shape = batch_shape + [k, 5]
      x_good = self._rng.randn(*x_good_shape).astype(dtype)
      x_bad = x_good.astype(np.float64)

      operator.matmul(x_good).eval()  # Should not raise.

      with self.assertRaisesRegexp(TypeError, 'dtype'):
        operator.matmul(x_bad)

      with self.assertRaisesRegexp(TypeError, 'dtype'):
        operator.solve(x_bad)

      with self.assertRaisesRegexp(TypeError, 'dtype'):
        operator.sqrt_solve(x_bad)

  def test_bad_rank_args_raise(self):
    # Prepend a singleton dimension, changing the rank of 'x', but not the size.
    dtype = np.float32
    batch_shape = [2, 3]
    k = 4
    with self.test_session():
      operator, _ = self._build_operator_and_mat(batch_shape, k, dtype=dtype)

      x_good_shape = batch_shape + [k, 5]
      x_good = self._rng.randn(*x_good_shape).astype(dtype)
      x_bad = x_good.reshape(1, 2, 3, 4, 5)

      operator.matmul(x_good).eval()  # Should not raise.

      with self.assertRaisesRegexp(ValueError, 'tensor rank'):
        operator.matmul(x_bad)

      with self.assertRaisesRegexp(ValueError, 'tensor rank'):
        operator.solve(x_bad)

      with self.assertRaisesRegexp(ValueError, 'tensor rank'):
        operator.sqrt_solve(x_bad)

  def test_incompatible_shape_args_raise(self):
    # Test shapes that are the same rank but incompatible for matrix
    # multiplication.
    dtype = np.float32
    batch_shape = [2, 3]
    k = 4
    with self.test_session():
      operator, _ = self._build_operator_and_mat(batch_shape, k, dtype=dtype)

      x_good_shape = batch_shape + [k, 5]
      x_good = self._rng.randn(*x_good_shape).astype(dtype)
      x_bad_shape = batch_shape + [5, k]
      x_bad = x_good.reshape(*x_bad_shape)

      operator.matmul(x_good).eval()  # Should not raise.

      with self.assertRaisesRegexp(ValueError, 'Incompatible'):
        operator.matmul(x_bad)

      with self.assertRaisesRegexp(ValueError, 'Incompatible'):
        operator.solve(x_bad)

      with self.assertRaisesRegexp(ValueError, 'Incompatible'):
        operator.sqrt_solve(x_bad)


if __name__ == '__main__':
  tf.test.main()
