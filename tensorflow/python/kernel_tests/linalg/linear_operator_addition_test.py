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

from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_addition
from tensorflow.python.platform import test

linalg = linalg_lib
rng = np.random.RandomState(0)

add_operators = linear_operator_addition.add_operators


# pylint: disable=unused-argument
class _BadAdder(linear_operator_addition._Adder):
  """Adder that will fail if used."""

  def can_add(self, op1, op2):
    raise AssertionError("BadAdder.can_add called!")

  def _add(self, op1, op2, operator_name, hints):
    raise AssertionError("This line should not be reached")


# pylint: enable=unused-argument


class LinearOperatorAdditionCorrectnessTest(test.TestCase):
  """Tests correctness of addition with combinations of a few Adders.

  Tests here are done with the _DEFAULT_ADDITION_TIERS, which means
  add_operators should reduce all operators resulting in one single operator.

  This shows that we are able to correctly combine adders using the tiered
  system.  All Adders should be tested separately, and there is no need to test
  every Adder within this class.
  """

  def test_one_operator_is_returned_unchanged(self):
    op_a = linalg.LinearOperatorDiag([1., 1.])
    op_sum = add_operators([op_a])
    self.assertEqual(1, len(op_sum))
    self.assertIs(op_sum[0], op_a)

  def test_at_least_one_operators_required(self):
    with self.assertRaisesRegexp(ValueError, "must contain at least one"):
      add_operators([])

  def test_attempting_to_add_numbers_raises(self):
    with self.assertRaisesRegexp(TypeError, "contain only LinearOperator"):
      add_operators([1, 2])

  def test_two_diag_operators(self):
    op_a = linalg.LinearOperatorDiag(
        [1., 1.], is_positive_definite=True, name="A")
    op_b = linalg.LinearOperatorDiag(
        [2., 2.], is_positive_definite=True, name="B")
    with self.cached_session():
      op_sum = add_operators([op_a, op_b])
      self.assertEqual(1, len(op_sum))
      op = op_sum[0]
      self.assertIsInstance(op, linalg_lib.LinearOperatorDiag)
      self.assertAllClose([[3., 0.], [0., 3.]], op.to_dense().eval())
      # Adding positive definite operators produces positive def.
      self.assertTrue(op.is_positive_definite)
      # Real diagonal ==> self-adjoint.
      self.assertTrue(op.is_self_adjoint)
      # Positive definite ==> non-singular
      self.assertTrue(op.is_non_singular)
      # Enforce particular name for this simple case
      self.assertEqual("Add/B__A/", op.name)

  def test_three_diag_operators(self):
    op1 = linalg.LinearOperatorDiag(
        [1., 1.], is_positive_definite=True, name="op1")
    op2 = linalg.LinearOperatorDiag(
        [2., 2.], is_positive_definite=True, name="op2")
    op3 = linalg.LinearOperatorDiag(
        [3., 3.], is_positive_definite=True, name="op3")
    with self.cached_session():
      op_sum = add_operators([op1, op2, op3])
      self.assertEqual(1, len(op_sum))
      op = op_sum[0]
      self.assertTrue(isinstance(op, linalg_lib.LinearOperatorDiag))
      self.assertAllClose([[6., 0.], [0., 6.]], op.to_dense().eval())
      # Adding positive definite operators produces positive def.
      self.assertTrue(op.is_positive_definite)
      # Real diagonal ==> self-adjoint.
      self.assertTrue(op.is_self_adjoint)
      # Positive definite ==> non-singular
      self.assertTrue(op.is_non_singular)

  def test_diag_tril_diag(self):
    op1 = linalg.LinearOperatorDiag(
        [1., 1.], is_non_singular=True, name="diag_a")
    op2 = linalg.LinearOperatorLowerTriangular(
        [[2., 0.], [0., 2.]],
        is_self_adjoint=True,
        is_non_singular=True,
        name="tril")
    op3 = linalg.LinearOperatorDiag(
        [3., 3.], is_non_singular=True, name="diag_b")
    with self.cached_session():
      op_sum = add_operators([op1, op2, op3])
      self.assertEqual(1, len(op_sum))
      op = op_sum[0]
      self.assertIsInstance(op, linalg_lib.LinearOperatorLowerTriangular)
      self.assertAllClose([[6., 0.], [0., 6.]], op.to_dense().eval())

      # The diag operators will be self-adjoint (because real and diagonal).
      # The TriL operator has the self-adjoint hint set.
      self.assertTrue(op.is_self_adjoint)

      # Even though op1/2/3 are non-singular, this does not imply op is.
      # Since no custom hint was provided, we default to None (unknown).
      self.assertEqual(None, op.is_non_singular)

  def test_matrix_diag_tril_diag_uses_custom_name(self):
    op0 = linalg.LinearOperatorFullMatrix(
        [[-1., -1.], [-1., -1.]], name="matrix")
    op1 = linalg.LinearOperatorDiag([1., 1.], name="diag_a")
    op2 = linalg.LinearOperatorLowerTriangular(
        [[2., 0.], [1.5, 2.]], name="tril")
    op3 = linalg.LinearOperatorDiag([3., 3.], name="diag_b")
    with self.cached_session():
      op_sum = add_operators([op0, op1, op2, op3], operator_name="my_operator")
      self.assertEqual(1, len(op_sum))
      op = op_sum[0]
      self.assertIsInstance(op, linalg_lib.LinearOperatorFullMatrix)
      self.assertAllClose([[5., -1.], [0.5, 5.]], op.to_dense().eval())
      self.assertEqual("my_operator", op.name)

  def test_incompatible_domain_dimensions_raises(self):
    op1 = linalg.LinearOperatorFullMatrix(rng.rand(2, 3))
    op2 = linalg.LinearOperatorDiag(rng.rand(2, 4))
    with self.assertRaisesRegexp(ValueError, "must.*same domain dimension"):
      add_operators([op1, op2])

  def test_incompatible_range_dimensions_raises(self):
    op1 = linalg.LinearOperatorFullMatrix(rng.rand(2, 3))
    op2 = linalg.LinearOperatorDiag(rng.rand(3, 3))
    with self.assertRaisesRegexp(ValueError, "must.*same range dimension"):
      add_operators([op1, op2])

  def test_non_broadcastable_batch_shape_raises(self):
    op1 = linalg.LinearOperatorFullMatrix(rng.rand(2, 3, 3))
    op2 = linalg.LinearOperatorDiag(rng.rand(4, 3, 3))
    with self.assertRaisesRegexp(ValueError, "Incompatible shapes"):
      add_operators([op1, op2])


class LinearOperatorOrderOfAdditionTest(test.TestCase):
  """Test that the order of addition is done as specified by tiers."""

  def test_tier_0_additions_done_in_tier_0(self):
    diag1 = linalg.LinearOperatorDiag([1.])
    diag2 = linalg.LinearOperatorDiag([1.])
    diag3 = linalg.LinearOperatorDiag([1.])
    addition_tiers = [
        [linear_operator_addition._AddAndReturnDiag()],
        [_BadAdder()],
    ]
    # Should not raise since all were added in tier 0, and tier 1 (with the
    # _BadAdder) was never reached.
    op_sum = add_operators([diag1, diag2, diag3], addition_tiers=addition_tiers)
    self.assertEqual(1, len(op_sum))
    self.assertIsInstance(op_sum[0], linalg.LinearOperatorDiag)

  def test_tier_1_additions_done_by_tier_1(self):
    diag1 = linalg.LinearOperatorDiag([1.])
    diag2 = linalg.LinearOperatorDiag([1.])
    tril = linalg.LinearOperatorLowerTriangular([[1.]])
    addition_tiers = [
        [linear_operator_addition._AddAndReturnDiag()],
        [linear_operator_addition._AddAndReturnTriL()],
        [_BadAdder()],
    ]
    # Should not raise since all were added by tier 1, and the
    # _BadAdder) was never reached.
    op_sum = add_operators([diag1, diag2, tril], addition_tiers=addition_tiers)
    self.assertEqual(1, len(op_sum))
    self.assertIsInstance(op_sum[0], linalg.LinearOperatorLowerTriangular)

  def test_tier_1_additions_done_by_tier_1_with_order_flipped(self):
    diag1 = linalg.LinearOperatorDiag([1.])
    diag2 = linalg.LinearOperatorDiag([1.])
    tril = linalg.LinearOperatorLowerTriangular([[1.]])
    addition_tiers = [
        [linear_operator_addition._AddAndReturnTriL()],
        [linear_operator_addition._AddAndReturnDiag()],
        [_BadAdder()],
    ]
    # Tier 0 could convert to TriL, and this converted everything to TriL,
    # including the Diags.
    # Tier 1 was never used.
    # Tier 2 was never used (therefore, _BadAdder didn't raise).
    op_sum = add_operators([diag1, diag2, tril], addition_tiers=addition_tiers)
    self.assertEqual(1, len(op_sum))
    self.assertIsInstance(op_sum[0], linalg.LinearOperatorLowerTriangular)

  def test_cannot_add_everything_so_return_more_than_one_operator(self):
    diag1 = linalg.LinearOperatorDiag([1.])
    diag2 = linalg.LinearOperatorDiag([2.])
    tril5 = linalg.LinearOperatorLowerTriangular([[5.]])
    addition_tiers = [
        [linear_operator_addition._AddAndReturnDiag()],
    ]
    # Tier 0 (the only tier) can only convert to Diag, so it combines the two
    # diags, but the TriL is unchanged.
    # Result should contain two operators, one Diag, one TriL.
    op_sum = add_operators([diag1, diag2, tril5], addition_tiers=addition_tiers)
    self.assertEqual(2, len(op_sum))
    found_diag = False
    found_tril = False
    with self.cached_session():
      for op in op_sum:
        if isinstance(op, linalg.LinearOperatorDiag):
          found_diag = True
          self.assertAllClose([[3.]], op.to_dense().eval())
        if isinstance(op, linalg.LinearOperatorLowerTriangular):
          found_tril = True
          self.assertAllClose([[5.]], op.to_dense().eval())
      self.assertTrue(found_diag and found_tril)

  def test_intermediate_tier_is_not_skipped(self):
    diag1 = linalg.LinearOperatorDiag([1.])
    diag2 = linalg.LinearOperatorDiag([1.])
    tril = linalg.LinearOperatorLowerTriangular([[1.]])
    addition_tiers = [
        [linear_operator_addition._AddAndReturnDiag()],
        [_BadAdder()],
        [linear_operator_addition._AddAndReturnTriL()],
    ]
    # tril cannot be added in tier 0, and the intermediate tier 1 with the
    # BadAdder will catch it and raise.
    with self.assertRaisesRegexp(AssertionError, "BadAdder.can_add called"):
      add_operators([diag1, diag2, tril], addition_tiers=addition_tiers)


class AddAndReturnScaledIdentityTest(test.TestCase):

  def setUp(self):
    self._adder = linear_operator_addition._AddAndReturnScaledIdentity()

  def test_identity_plus_identity(self):
    id1 = linalg.LinearOperatorIdentity(num_rows=2)
    id2 = linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[3])
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(id1, id2))
    operator = self._adder.add(id1, id2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorScaledIdentity)

    with self.cached_session():
      self.assertAllClose(2 *
                          linalg_ops.eye(num_rows=2, batch_shape=[3]).eval(),
                          operator.to_dense().eval())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)

  def test_identity_plus_scaled_identity(self):
    id1 = linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[3])
    id2 = linalg.LinearOperatorScaledIdentity(num_rows=2, multiplier=2.2)
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(id1, id2))
    operator = self._adder.add(id1, id2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorScaledIdentity)

    with self.cached_session():
      self.assertAllClose(3.2 *
                          linalg_ops.eye(num_rows=2, batch_shape=[3]).eval(),
                          operator.to_dense().eval())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)

  def test_scaled_identity_plus_scaled_identity(self):
    id1 = linalg.LinearOperatorScaledIdentity(
        num_rows=2, multiplier=[2.2, 2.2, 2.2])
    id2 = linalg.LinearOperatorScaledIdentity(num_rows=2, multiplier=-1.0)
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(id1, id2))
    operator = self._adder.add(id1, id2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorScaledIdentity)

    with self.cached_session():
      self.assertAllClose(1.2 *
                          linalg_ops.eye(num_rows=2, batch_shape=[3]).eval(),
                          operator.to_dense().eval())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)


class AddAndReturnDiagTest(test.TestCase):

  def setUp(self):
    self._adder = linear_operator_addition._AddAndReturnDiag()

  def test_identity_plus_identity_returns_diag(self):
    id1 = linalg.LinearOperatorIdentity(num_rows=2)
    id2 = linalg.LinearOperatorIdentity(num_rows=2, batch_shape=[3])
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(id1, id2))
    operator = self._adder.add(id1, id2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorDiag)

    with self.cached_session():
      self.assertAllClose(2 *
                          linalg_ops.eye(num_rows=2, batch_shape=[3]).eval(),
                          operator.to_dense().eval())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)

  def test_diag_plus_diag(self):
    diag1 = rng.rand(2, 3, 4)
    diag2 = rng.rand(4)
    op1 = linalg.LinearOperatorDiag(diag1)
    op2 = linalg.LinearOperatorDiag(diag2)
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(op1, op2))
    operator = self._adder.add(op1, op2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorDiag)

    with self.cached_session():
      self.assertAllClose(
          linalg.LinearOperatorDiag(diag1 + diag2).to_dense().eval(),
          operator.to_dense().eval())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)


class AddAndReturnTriLTest(test.TestCase):

  def setUp(self):
    self._adder = linear_operator_addition._AddAndReturnTriL()

  def test_diag_plus_tril(self):
    diag = linalg.LinearOperatorDiag([1., 2.])
    tril = linalg.LinearOperatorLowerTriangular([[10., 0.], [30., 0.]])
    hints = linear_operator_addition._Hints(
        is_positive_definite=True, is_non_singular=True)

    self.assertTrue(self._adder.can_add(diag, diag))
    self.assertTrue(self._adder.can_add(diag, tril))
    operator = self._adder.add(diag, tril, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorLowerTriangular)

    with self.cached_session():
      self.assertAllClose([[11., 0.], [30., 2.]], operator.to_dense().eval())
    self.assertTrue(operator.is_positive_definite)
    self.assertTrue(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)


class AddAndReturnMatrixTest(test.TestCase):

  def setUp(self):
    self._adder = linear_operator_addition._AddAndReturnMatrix()

  def test_diag_plus_diag(self):
    diag1 = linalg.LinearOperatorDiag([1., 2.])
    diag2 = linalg.LinearOperatorDiag([-1., 3.])
    hints = linear_operator_addition._Hints(
        is_positive_definite=False, is_non_singular=False)

    self.assertTrue(self._adder.can_add(diag1, diag2))
    operator = self._adder.add(diag1, diag2, "my_operator", hints)
    self.assertIsInstance(operator, linalg.LinearOperatorFullMatrix)

    with self.cached_session():
      self.assertAllClose([[0., 0.], [0., 5.]], operator.to_dense().eval())
    self.assertFalse(operator.is_positive_definite)
    self.assertFalse(operator.is_non_singular)
    self.assertEqual("my_operator", operator.name)


if __name__ == "__main__":
  test.main()
