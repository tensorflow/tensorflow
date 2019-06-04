# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for registration mechanisms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops.linalg import cholesky_registrations  # pylint: disable=unused-import
from tensorflow.python.ops.linalg import linear_operator
from tensorflow.python.ops.linalg import linear_operator_algebra
from tensorflow.python.ops.linalg import matmul_registrations  # pylint: disable=unused-import
from tensorflow.python.ops.linalg import solve_registrations  # pylint: disable=unused-import
from tensorflow.python.platform import test

# pylint: disable=protected-access
_ADJOINTS = linear_operator_algebra._ADJOINTS
_registered_adjoint = linear_operator_algebra._registered_adjoint
_CHOLESKY_DECOMPS = linear_operator_algebra._CHOLESKY_DECOMPS
_registered_cholesky = linear_operator_algebra._registered_cholesky
_INVERSES = linear_operator_algebra._INVERSES
_registered_inverse = linear_operator_algebra._registered_inverse
_MATMUL = linear_operator_algebra._MATMUL
_registered_matmul = linear_operator_algebra._registered_matmul
_SOLVE = linear_operator_algebra._SOLVE
_registered_solve = linear_operator_algebra._registered_solve
# pylint: enable=protected-access


class AdjointTest(test.TestCase):

  def testRegistration(self):

    class CustomLinOp(linear_operator.LinearOperator):

      def _matmul(self, a):
        pass

      def _shape(self):
        return tensor_shape.TensorShape([1, 1])

      def _shape_tensor(self):
        pass

    # Register Adjoint to a lambda that spits out the name parameter
    @linear_operator_algebra.RegisterAdjoint(CustomLinOp)
    def _adjoint(a):  # pylint: disable=unused-argument,unused-variable
      return "OK"

    self.assertEqual("OK", CustomLinOp(dtype=None).adjoint())

  def testRegistrationFailures(self):

    class CustomLinOp(linear_operator.LinearOperator):
      pass

    with self.assertRaisesRegexp(TypeError, "must be callable"):
      linear_operator_algebra.RegisterAdjoint(CustomLinOp)("blah")

    # First registration is OK
    linear_operator_algebra.RegisterAdjoint(CustomLinOp)(lambda a: None)

    # Second registration fails
    with self.assertRaisesRegexp(ValueError, "has already been registered"):
      linear_operator_algebra.RegisterAdjoint(CustomLinOp)(lambda a: None)

  def testExactAdjointRegistrationsAllMatch(self):
    for (k, v) in _ADJOINTS.items():
      self.assertEqual(v, _registered_adjoint(k[0]))


class CholeskyTest(test.TestCase):

  def testRegistration(self):

    class CustomLinOp(linear_operator.LinearOperator):

      def _matmul(self, a):
        pass

      def _shape(self):
        return tensor_shape.TensorShape([1, 1])

      def _shape_tensor(self):
        pass

    # Register Cholesky to a lambda that spits out the name parameter
    @linear_operator_algebra.RegisterCholesky(CustomLinOp)
    def _cholesky(a):  # pylint: disable=unused-argument,unused-variable
      return "OK"

    with self.assertRaisesRegexp(ValueError, "positive definite"):
      CustomLinOp(dtype=None, is_self_adjoint=True).cholesky()

    with self.assertRaisesRegexp(ValueError, "self adjoint"):
      CustomLinOp(dtype=None, is_positive_definite=True).cholesky()

    custom_linop = CustomLinOp(
        dtype=None, is_self_adjoint=True, is_positive_definite=True)
    self.assertEqual("OK", custom_linop.cholesky())

  def testRegistrationFailures(self):

    class CustomLinOp(linear_operator.LinearOperator):
      pass

    with self.assertRaisesRegexp(TypeError, "must be callable"):
      linear_operator_algebra.RegisterCholesky(CustomLinOp)("blah")

    # First registration is OK
    linear_operator_algebra.RegisterCholesky(CustomLinOp)(lambda a: None)

    # Second registration fails
    with self.assertRaisesRegexp(ValueError, "has already been registered"):
      linear_operator_algebra.RegisterCholesky(CustomLinOp)(lambda a: None)

  def testExactCholeskyRegistrationsAllMatch(self):
    for (k, v) in _CHOLESKY_DECOMPS.items():
      self.assertEqual(v, _registered_cholesky(k[0]))


class MatmulTest(test.TestCase):

  def testRegistration(self):

    class CustomLinOp(linear_operator.LinearOperator):

      def _matmul(self, a):
        pass

      def _shape(self):
        return tensor_shape.TensorShape([1, 1])

      def _shape_tensor(self):
        pass

    # Register Matmul to a lambda that spits out the name parameter
    @linear_operator_algebra.RegisterMatmul(CustomLinOp, CustomLinOp)
    def _matmul(a, b):  # pylint: disable=unused-argument,unused-variable
      return "OK"

    custom_linop = CustomLinOp(
        dtype=None, is_self_adjoint=True, is_positive_definite=True)
    self.assertEqual("OK", custom_linop.matmul(custom_linop))

  def testRegistrationFailures(self):

    class CustomLinOp(linear_operator.LinearOperator):
      pass

    with self.assertRaisesRegexp(TypeError, "must be callable"):
      linear_operator_algebra.RegisterMatmul(CustomLinOp, CustomLinOp)("blah")

    # First registration is OK
    linear_operator_algebra.RegisterMatmul(
        CustomLinOp, CustomLinOp)(lambda a: None)

    # Second registration fails
    with self.assertRaisesRegexp(ValueError, "has already been registered"):
      linear_operator_algebra.RegisterMatmul(
          CustomLinOp, CustomLinOp)(lambda a: None)

  def testExactMatmulRegistrationsAllMatch(self):
    for (k, v) in _MATMUL.items():
      self.assertEqual(v, _registered_matmul(k[0], k[1]))


class SolveTest(test.TestCase):

  def testRegistration(self):

    class CustomLinOp(linear_operator.LinearOperator):

      def _matmul(self, a):
        pass

      def _solve(self, a):
        pass

      def _shape(self):
        return tensor_shape.TensorShape([1, 1])

      def _shape_tensor(self):
        pass

    # Register Solve to a lambda that spits out the name parameter
    @linear_operator_algebra.RegisterSolve(CustomLinOp, CustomLinOp)
    def _solve(a, b):  # pylint: disable=unused-argument,unused-variable
      return "OK"

    custom_linop = CustomLinOp(
        dtype=None, is_self_adjoint=True, is_positive_definite=True)
    self.assertEqual("OK", custom_linop.solve(custom_linop))

  def testRegistrationFailures(self):

    class CustomLinOp(linear_operator.LinearOperator):
      pass

    with self.assertRaisesRegexp(TypeError, "must be callable"):
      linear_operator_algebra.RegisterSolve(CustomLinOp, CustomLinOp)("blah")

    # First registration is OK
    linear_operator_algebra.RegisterSolve(
        CustomLinOp, CustomLinOp)(lambda a: None)

    # Second registration fails
    with self.assertRaisesRegexp(ValueError, "has already been registered"):
      linear_operator_algebra.RegisterSolve(
          CustomLinOp, CustomLinOp)(lambda a: None)

  def testExactSolveRegistrationsAllMatch(self):
    for (k, v) in _SOLVE.items():
      self.assertEqual(v, _registered_solve(k[0], k[1]))


class InverseTest(test.TestCase):

  def testRegistration(self):

    class CustomLinOp(linear_operator.LinearOperator):

      def _matmul(self, a):
        pass

      def _shape(self):
        return tensor_shape.TensorShape([1, 1])

      def _shape_tensor(self):
        pass

    # Register Inverse to a lambda that spits out the name parameter
    @linear_operator_algebra.RegisterInverse(CustomLinOp)
    def _inverse(a):  # pylint: disable=unused-argument,unused-variable
      return "OK"

    with self.assertRaisesRegexp(ValueError, "singular"):
      CustomLinOp(dtype=None, is_non_singular=False).inverse()

    self.assertEqual("OK", CustomLinOp(
        dtype=None, is_non_singular=True).inverse())

  def testRegistrationFailures(self):

    class CustomLinOp(linear_operator.LinearOperator):
      pass

    with self.assertRaisesRegexp(TypeError, "must be callable"):
      linear_operator_algebra.RegisterInverse(CustomLinOp)("blah")

    # First registration is OK
    linear_operator_algebra.RegisterInverse(CustomLinOp)(lambda a: None)

    # Second registration fails
    with self.assertRaisesRegexp(ValueError, "has already been registered"):
      linear_operator_algebra.RegisterInverse(CustomLinOp)(lambda a: None)

  def testExactRegistrationsAllMatch(self):
    for (k, v) in _INVERSES.items():
      self.assertEqual(v, _registered_inverse(k[0]))


if __name__ == "__main__":
  test.main()
