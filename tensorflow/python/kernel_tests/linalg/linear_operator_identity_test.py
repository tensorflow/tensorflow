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

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test


random_seed.set_random_seed(23)
rng = np.random.RandomState(2016)


class LinearOperatorIdentityTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @property
  def _dtypes_to_test(self):
    # TODO(langmore) Test tf.float16 once tf.matrix_solve works in
    # 16bit.
    return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

  def _operator_and_mat_and_feed_dict(self, shape, dtype, use_placeholder):
    shape = list(shape)
    assert shape[-1] == shape[-2]

    batch_shape = shape[:-2]
    num_rows = shape[-1]

    operator = linalg_lib.LinearOperatorIdentity(
        num_rows, batch_shape=batch_shape, dtype=dtype)
    mat = linalg_ops.eye(num_rows, batch_shape=batch_shape, dtype=dtype)

    # Nothing to feed since LinearOperatorIdentity takes no Tensor args.
    if use_placeholder:
      feed_dict = {}
    else:
      feed_dict = None

    return operator, mat, feed_dict

  def test_assert_positive_definite(self):
    with self.test_session():
      operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
      operator.assert_positive_definite().run()  # Should not fail

  def test_assert_non_singular(self):
    with self.test_session():
      operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
      operator.assert_non_singular().run()  # Should not fail

  def test_assert_self_adjoint(self):
    with self.test_session():
      operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
      operator.assert_self_adjoint().run()  # Should not fail

  def test_float16_matmul(self):
    # float16 cannot be tested by base test class because tf.matrix_solve does
    # not work with float16.
    with self.test_session():
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows=2, dtype=dtypes.float16)
      x = rng.randn(2, 3).astype(np.float16)
      y = operator.matmul(x)
      self.assertAllClose(x, y.eval())

  def test_non_scalar_num_rows_raises_static(self):
    with self.assertRaisesRegexp(ValueError, "must be a 0-D Tensor"):
      linalg_lib.LinearOperatorIdentity(num_rows=[2])

  def test_non_integer_num_rows_raises_static(self):
    with self.assertRaisesRegexp(TypeError, "must be integer"):
      linalg_lib.LinearOperatorIdentity(num_rows=2.)

  def test_negative_num_rows_raises_static(self):
    with self.assertRaisesRegexp(ValueError, "must be non-negative"):
      linalg_lib.LinearOperatorIdentity(num_rows=-2)

  def test_non_1d_batch_shape_raises_static(self):
    with self.assertRaisesRegexp(ValueError, "must be a 1-D"):
      linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=2)

  def test_non_integer_batch_shape_raises_static(self):
    with self.assertRaisesRegexp(TypeError, "must be integer"):
      linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=[2.])

  def test_negative_batch_shape_raises_static(self):
    with self.assertRaisesRegexp(ValueError, "must be non-negative"):
      linalg_lib.LinearOperatorIdentity(num_rows=2, batch_shape=[-2])

  def test_non_scalar_num_rows_raises_dynamic(self):
    with self.test_session():
      num_rows = array_ops.placeholder(dtypes.int32)
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows, assert_proper_shapes=True)
      with self.assertRaisesOpError("must be a 0-D Tensor"):
        operator.to_dense().eval(feed_dict={num_rows: [2]})

  def test_negative_num_rows_raises_dynamic(self):
    with self.test_session():
      num_rows = array_ops.placeholder(dtypes.int32)
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows, assert_proper_shapes=True)
      with self.assertRaisesOpError("must be non-negative"):
        operator.to_dense().eval(feed_dict={num_rows: -2})

  def test_non_1d_batch_shape_raises_dynamic(self):
    with self.test_session():
      batch_shape = array_ops.placeholder(dtypes.int32)
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
      with self.assertRaisesOpError("must be a 1-D"):
        operator.to_dense().eval(feed_dict={batch_shape: 2})

  def test_negative_batch_shape_raises_dynamic(self):
    with self.test_session():
      batch_shape = array_ops.placeholder(dtypes.int32)
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows=2, batch_shape=batch_shape, assert_proper_shapes=True)
      with self.assertRaisesOpError("must be non-negative"):
        operator.to_dense().eval(feed_dict={batch_shape: [-2]})

  def test_wrong_matrix_dimensions_raises_static(self):
    operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
    x = rng.randn(3, 3).astype(np.float32)
    with self.assertRaisesRegexp(ValueError, "Dimensions.*not compatible"):
      operator.matmul(x)

  def test_wrong_matrix_dimensions_raises_dynamic(self):
    num_rows = array_ops.placeholder(dtypes.int32)
    x = array_ops.placeholder(dtypes.float32)

    with self.test_session():
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows, assert_proper_shapes=True)
      y = operator.matmul(x)
      with self.assertRaisesOpError("Incompatible.*dimensions"):
        y.eval(feed_dict={num_rows: 2, x: rng.rand(3, 3)})

  def test_default_batch_shape_broadcasts_with_everything_static(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.test_session() as sess:
      x = random_ops.random_normal(shape=(1, 2, 3, 4))
      operator = linalg_lib.LinearOperatorIdentity(num_rows=3, dtype=x.dtype)

      operator_matmul = operator.matmul(x)
      expected = x

      self.assertAllEqual(operator_matmul.get_shape(), expected.get_shape())
      self.assertAllClose(*sess.run([operator_matmul, expected]))

  def test_default_batch_shape_broadcasts_with_everything_dynamic(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.test_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      operator = linalg_lib.LinearOperatorIdentity(num_rows=3, dtype=x.dtype)

      operator_matmul = operator.matmul(x)
      expected = x

      feed_dict = {x: rng.randn(1, 2, 3, 4)}

      self.assertAllClose(
          *sess.run([operator_matmul, expected], feed_dict=feed_dict))

  def test_broadcast_matmul_static_shapes(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.test_session() as sess:
      # Given this x and LinearOperatorIdentity shape of (2, 1, 3, 3), the
      # broadcast shape of operator and 'x' is (2, 2, 3, 4)
      x = random_ops.random_normal(shape=(1, 2, 3, 4))
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows=3, batch_shape=(2, 1), dtype=x.dtype)

      # Batch matrix of zeros with the broadcast shape of x and operator.
      zeros = array_ops.zeros(shape=(2, 2, 3, 4), dtype=x.dtype)

      # Expected result of matmul and solve.
      expected = x + zeros

      operator_matmul = operator.matmul(x)
      self.assertAllEqual(operator_matmul.get_shape(), expected.get_shape())
      self.assertAllClose(*sess.run([operator_matmul, expected]))

  def test_broadcast_matmul_dynamic_shapes(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.test_session() as sess:
      # Given this x and LinearOperatorIdentity shape of (2, 1, 3, 3), the
      # broadcast shape of operator and 'x' is (2, 2, 3, 4)
      x = array_ops.placeholder(dtypes.float32)
      num_rows = array_ops.placeholder(dtypes.int32)
      batch_shape = array_ops.placeholder(dtypes.int32)

      operator = linalg_lib.LinearOperatorIdentity(
          num_rows, batch_shape=batch_shape)
      feed_dict = {x: rng.rand(1, 2, 3, 4), num_rows: 3, batch_shape: (2, 1)}

      # Batch matrix of zeros with the broadcast shape of x and operator.
      zeros = array_ops.zeros(shape=(2, 2, 3, 4), dtype=x.dtype)

      # Expected result of matmul and solve.
      expected = x + zeros

      operator_matmul = operator.matmul(x)
      self.assertAllClose(
          *sess.run([operator_matmul, expected], feed_dict=feed_dict))

  def test_is_x_flags(self):
    # The is_x flags are by default all True.
    operator = linalg_lib.LinearOperatorIdentity(num_rows=2)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint)

    # Any of them False raises because the identity is always self-adjoint etc..
    with self.assertRaisesRegexp(ValueError, "is always non-singular"):
      operator = linalg_lib.LinearOperatorIdentity(
          num_rows=2,
          is_non_singular=None,
      )


class LinearOperatorScaledIdentityTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  @property
  def _dtypes_to_test(self):
    # TODO(langmore) Test tf.float16 once tf.matrix_solve works in
    # 16bit.
    return [dtypes.float32, dtypes.float64, dtypes.complex64, dtypes.complex128]

  def _operator_and_mat_and_feed_dict(self, shape, dtype, use_placeholder):
    shape = list(shape)
    assert shape[-1] == shape[-2]

    batch_shape = shape[:-2]
    num_rows = shape[-1]

    # Uniform values that are at least length 1 from the origin.  Allows the
    # operator to be well conditioned.
    # Shape batch_shape
    multiplier = linear_operator_test_util.random_sign_uniform(
        shape=batch_shape, minval=1., maxval=2., dtype=dtype)

    operator = linalg_lib.LinearOperatorScaledIdentity(num_rows, multiplier)

    # Nothing to feed since LinearOperatorScaledIdentity takes no Tensor args.
    if use_placeholder:
      multiplier_ph = array_ops.placeholder(dtype=dtype)
      multiplier = multiplier.eval()
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows, multiplier_ph)
      feed_dict = {multiplier_ph: multiplier}
    else:
      feed_dict = None

    multiplier_matrix = array_ops.expand_dims(
        array_ops.expand_dims(multiplier, -1), -1)
    mat = multiplier_matrix * linalg_ops.eye(
        num_rows, batch_shape=batch_shape, dtype=dtype)

    return operator, mat, feed_dict

  def test_assert_positive_definite_does_not_raise_when_positive(self):
    with self.test_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=1.)
      operator.assert_positive_definite().run()  # Should not fail

  def test_assert_positive_definite_raises_when_negative(self):
    with self.test_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=-1.)
      with self.assertRaisesOpError("not positive definite"):
        operator.assert_positive_definite().run()

  def test_assert_non_singular_does_not_raise_when_non_singular(self):
    with self.test_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=[1., 2., 3.])
      operator.assert_non_singular().run()  # Should not fail

  def test_assert_non_singular_raises_when_singular(self):
    with self.test_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=[1., 2., 0.])
      with self.assertRaisesOpError("was singular"):
        operator.assert_non_singular().run()

  def test_assert_self_adjoint_does_not_raise_when_self_adjoint(self):
    with self.test_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=[1. + 0J])
      operator.assert_self_adjoint().run()  # Should not fail

  def test_assert_self_adjoint_raises_when_not_self_adjoint(self):
    with self.test_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=[1. + 1J])
      with self.assertRaisesOpError("not self-adjoint"):
        operator.assert_self_adjoint().run()

  def test_float16_matmul(self):
    # float16 cannot be tested by base test class because tf.matrix_solve does
    # not work with float16.
    with self.test_session():
      multiplier = rng.rand(3).astype(np.float16)
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=2, multiplier=multiplier)
      x = rng.randn(2, 3).astype(np.float16)
      y = operator.matmul(x)
      self.assertAllClose(multiplier[..., None, None] * x, y.eval())

  def test_non_scalar_num_rows_raises_static(self):
    # Many "test_...num_rows" tests are performed in LinearOperatorIdentity.
    with self.assertRaisesRegexp(ValueError, "must be a 0-D Tensor"):
      linalg_lib.LinearOperatorScaledIdentity(
          num_rows=[2], multiplier=123.)

  def test_wrong_matrix_dimensions_raises_static(self):
    operator = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=2.2)
    x = rng.randn(3, 3).astype(np.float32)
    with self.assertRaisesRegexp(ValueError, "Dimensions.*not compatible"):
      operator.matmul(x)

  def test_wrong_matrix_dimensions_raises_dynamic(self):
    num_rows = array_ops.placeholder(dtypes.int32)
    x = array_ops.placeholder(dtypes.float32)

    with self.test_session():
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows, multiplier=[1., 2], assert_proper_shapes=True)
      y = operator.matmul(x)
      with self.assertRaisesOpError("Incompatible.*dimensions"):
        y.eval(feed_dict={num_rows: 2, x: rng.rand(3, 3)})

  def test_broadcast_matmul_and_solve(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.test_session() as sess:
      # Given this x and LinearOperatorScaledIdentity shape of (2, 1, 3, 3), the
      # broadcast shape of operator and 'x' is (2, 2, 3, 4)
      x = random_ops.random_normal(shape=(1, 2, 3, 4))

      # operator is 2.2 * identity (with a batch shape).
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=3, multiplier=2.2 * array_ops.ones((2, 1)))

      # Batch matrix of zeros with the broadcast shape of x and operator.
      zeros = array_ops.zeros(shape=(2, 2, 3, 4), dtype=x.dtype)

      # Test matmul
      expected = x * 2.2 + zeros
      operator_matmul = operator.matmul(x)
      self.assertAllEqual(operator_matmul.get_shape(), expected.get_shape())
      self.assertAllClose(*sess.run([operator_matmul, expected]))

      # Test solve
      expected = x / 2.2 + zeros
      operator_solve = operator.solve(x)
      self.assertAllEqual(operator_solve.get_shape(), expected.get_shape())
      self.assertAllClose(*sess.run([operator_solve, expected]))

  def test_broadcast_matmul_and_solve_scalar_scale_multiplier(self):
    # These cannot be done in the automated (base test class) tests since they
    # test shapes that tf.batch_matmul cannot handle.
    # In particular, tf.batch_matmul does not broadcast.
    with self.test_session() as sess:
      # Given this x and LinearOperatorScaledIdentity shape of (3, 3), the
      # broadcast shape of operator and 'x' is (1, 2, 3, 4), which is the same
      # shape as x.
      x = random_ops.random_normal(shape=(1, 2, 3, 4))

      # operator is 2.2 * identity (with a batch shape).
      operator = linalg_lib.LinearOperatorScaledIdentity(
          num_rows=3, multiplier=2.2)

      # Test matmul
      expected = x * 2.2
      operator_matmul = operator.matmul(x)
      self.assertAllEqual(operator_matmul.get_shape(), expected.get_shape())
      self.assertAllClose(*sess.run([operator_matmul, expected]))

      # Test solve
      expected = x / 2.2
      operator_solve = operator.solve(x)
      self.assertAllEqual(operator_solve.get_shape(), expected.get_shape())
      self.assertAllClose(*sess.run([operator_solve, expected]))

  def test_is_x_flags(self):
    operator = linalg_lib.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=1.,
        is_positive_definite=False, is_non_singular=True)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertTrue(operator.is_self_adjoint is None)


if __name__ == "__main__":
  test.main()
