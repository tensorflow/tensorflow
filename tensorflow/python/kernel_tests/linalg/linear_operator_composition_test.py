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
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib
random_seed.set_random_seed(23)
rng = np.random.RandomState(0)


class SquareLinearOperatorCompositionTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def setUp(self):
    # Increase from 1e-6 to 1e-4
    self._atol[dtypes.float32] = 1e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._rtol[dtypes.complex64] = 1e-4

  def _operator_and_mat_and_feed_dict(self, build_info, dtype, use_placeholder):
    sess = ops.get_default_session()
    shape = list(build_info.shape)

    # Either 1 or 2 matrices, depending.
    num_operators = rng.randint(low=1, high=3)
    matrices = [
        linear_operator_test_util.random_positive_definite_matrix(
            shape, dtype, force_well_conditioned=True)
        for _ in range(num_operators)
    ]

    if use_placeholder:
      matrices_ph = [
          array_ops.placeholder(dtype=dtype) for _ in range(num_operators)
      ]
      # Evaluate here because (i) you cannot feed a tensor, and (ii)
      # values are random and we want the same value used for both mat and
      # feed_dict.
      matrices = sess.run(matrices)
      operator = linalg.LinearOperatorComposition(
          [linalg.LinearOperatorFullMatrix(m_ph) for m_ph in matrices_ph],
          is_square=True)
      feed_dict = {m_ph: m for (m_ph, m) in zip(matrices_ph, matrices)}
    else:
      operator = linalg.LinearOperatorComposition(
          [linalg.LinearOperatorFullMatrix(m) for m in matrices])
      feed_dict = None
      # Should be auto-set.
      self.assertTrue(operator.is_square)

    # Convert back to Tensor.  Needed if use_placeholder, since then we have
    # already evaluated each matrix to a numpy array.
    matmul_order_list = list(reversed(matrices))
    mat = ops.convert_to_tensor(matmul_order_list[0])
    for other_mat in matmul_order_list[1:]:
      mat = math_ops.matmul(other_mat, mat)

    return operator, mat, feed_dict

  def test_is_x_flags(self):
    # Matrix with two positive eigenvalues, 1, and 1.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[1., 0.], [1., 1.]]
    operator = linalg.LinearOperatorComposition(
        [linalg.LinearOperatorFullMatrix(matrix)],
        is_positive_definite=True,
        is_non_singular=True,
        is_self_adjoint=False)
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertFalse(operator.is_self_adjoint)

  def test_is_non_singular_auto_set(self):
    # Matrix with two positive eigenvalues, 11 and 8.
    # The matrix values do not effect auto-setting of the flags.
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, is_non_singular=True)

    operator = linalg.LinearOperatorComposition(
        [operator_1, operator_2],
        is_positive_definite=False,  # No reason it HAS to be False...
        is_non_singular=None)
    self.assertFalse(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)

    with self.assertRaisesRegexp(ValueError, "always non-singular"):
      linalg.LinearOperatorComposition(
          [operator_1, operator_2], is_non_singular=False)

  def test_name(self):
    matrix = [[11., 0.], [1., 8.]]
    operator_1 = linalg.LinearOperatorFullMatrix(matrix, name="left")
    operator_2 = linalg.LinearOperatorFullMatrix(matrix, name="right")

    operator = linalg.LinearOperatorComposition([operator_1, operator_2])

    self.assertEqual("left_o_right", operator.name)

  def test_different_dtypes_raises(self):
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3).astype(np.float32))
    ]
    with self.assertRaisesRegexp(TypeError, "same dtype"):
      linalg.LinearOperatorComposition(operators)

  def test_empty_operators_raises(self):
    with self.assertRaisesRegexp(ValueError, "non-empty"):
      linalg.LinearOperatorComposition([])


class NonSquareLinearOperatorCompositionTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def setUp(self):
    # Increase from 1e-6 to 1e-4
    self._atol[dtypes.float32] = 1e-4
    self._atol[dtypes.complex64] = 1e-4
    self._rtol[dtypes.float32] = 1e-4
    self._rtol[dtypes.complex64] = 1e-4

  def _operator_and_mat_and_feed_dict(self, build_info, dtype, use_placeholder):
    sess = ops.get_default_session()
    shape = list(build_info.shape)

    # Test only the case of 2 matrices.
    # The Square test uses either 1 or 2, so we have tested the case of 1 matrix
    # sufficiently.
    num_operators = 2

    # Create 2 matrices/operators, A1, A2, which becomes A = A1 A2.
    # Use inner dimension of 2.
    k = 2
    batch_shape = shape[:-2]
    shape_1 = batch_shape + [shape[-2], k]
    shape_2 = batch_shape + [k, shape[-1]]

    matrices = [
        linear_operator_test_util.random_normal(
            shape_1, dtype=dtype), linear_operator_test_util.random_normal(
                shape_2, dtype=dtype)
    ]

    if use_placeholder:
      matrices_ph = [
          array_ops.placeholder(dtype=dtype) for _ in range(num_operators)
      ]
      # Evaluate here because (i) you cannot feed a tensor, and (ii)
      # values are random and we want the same value used for both mat and
      # feed_dict.
      matrices = sess.run(matrices)
      operator = linalg.LinearOperatorComposition(
          [linalg.LinearOperatorFullMatrix(m_ph) for m_ph in matrices_ph])
      feed_dict = {m_ph: m for (m_ph, m) in zip(matrices_ph, matrices)}
    else:
      operator = linalg.LinearOperatorComposition(
          [linalg.LinearOperatorFullMatrix(m) for m in matrices])
      feed_dict = None

    # Convert back to Tensor.  Needed if use_placeholder, since then we have
    # already evaluated each matrix to a numpy array.
    matmul_order_list = list(reversed(matrices))
    mat = ops.convert_to_tensor(matmul_order_list[0])
    for other_mat in matmul_order_list[1:]:
      mat = math_ops.matmul(other_mat, mat)

    return operator, mat, feed_dict

  def test_static_shapes(self):
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 4)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 4, 5))
    ]
    operator = linalg.LinearOperatorComposition(operators)
    self.assertAllEqual((2, 3, 5), operator.shape)

  def test_shape_tensors_when_statically_available(self):
    operators = [
        linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 4)),
        linalg.LinearOperatorFullMatrix(rng.rand(2, 4, 5))
    ]
    operator = linalg.LinearOperatorComposition(operators)
    with self.test_session():
      self.assertAllEqual((2, 3, 5), operator.shape_tensor().eval())

  def test_shape_tensors_when_only_dynamically_available(self):
    mat_1 = rng.rand(1, 2, 3, 4)
    mat_2 = rng.rand(1, 2, 4, 5)
    mat_ph_1 = array_ops.placeholder(dtypes.float64)
    mat_ph_2 = array_ops.placeholder(dtypes.float64)
    feed_dict = {mat_ph_1: mat_1, mat_ph_2: mat_2}

    operators = [
        linalg.LinearOperatorFullMatrix(mat_ph_1),
        linalg.LinearOperatorFullMatrix(mat_ph_2)
    ]
    operator = linalg.LinearOperatorComposition(operators)
    with self.test_session():
      self.assertAllEqual(
          (1, 2, 3, 5), operator.shape_tensor().eval(feed_dict=feed_dict))


if __name__ == "__main__":
  test.main()
