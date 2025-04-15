# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Common utilities for LinearOperator property hints."""


# Note: only use this method in the commuting case.
def combined_commuting_self_adjoint_hint(operator_a, operator_b):
  """Get combined hint for self-adjoint-ness."""

  # The property is preserved under composition when the operators commute.
  if operator_a.is_self_adjoint and operator_b.is_self_adjoint:
    return True

  # The property is not preserved when an operator with the property is composed
  # with an operator without the property.

  # pylint:disable=g-bool-id-comparison
  if ((operator_a.is_self_adjoint is True and
       operator_b.is_self_adjoint is False) or
      (operator_a.is_self_adjoint is False and
       operator_b.is_self_adjoint is True)):
    return False
  # pylint:enable=g-bool-id-comparison

  # The property is not known when operators are not known to have the property
  # or both operators don't have the property (the property for the complement
  # class is not closed under composition).
  return None


def is_square(operator_a, operator_b):
  """Return a hint to whether the composition is square."""
  if operator_a.is_square and operator_b.is_square:
    return True
  if operator_a.is_square is False and operator_b.is_square is False:  # pylint:disable=g-bool-id-comparison
    # Let A have shape [B, M, N], B have shape [B, N, L].
    m = operator_a.range_dimension
    l = operator_b.domain_dimension
    if m is not None and l is not None:
      return m == l

  if (operator_a.is_square != operator_b.is_square) and (
      operator_a.is_square is not None and operator_b.is_square is not None):
    return False

  return None


# Note: Positive definiteness is only guaranteed to be preserved
# when the operators commute and are symmetric. Only use this method in
# commuting cases.
def combined_commuting_positive_definite_hint(operator_a, operator_b):
  """Get combined PD hint for compositions."""
  # pylint:disable=g-bool-id-comparison
  if (operator_a.is_positive_definite is True and
      operator_a.is_self_adjoint is True and
      operator_b.is_positive_definite is True and
      operator_b.is_self_adjoint is True):
    return True
  # pylint:enable=g-bool-id-comparison

  return None


def combined_non_singular_hint(operator_a, operator_b):
  """Get combined hint for when ."""
  # If either operator is not-invertible the composition isn't.

  # pylint:disable=g-bool-id-comparison
  if (operator_a.is_non_singular is False or
      operator_b.is_non_singular is False):
    return False
  # pylint:enable=g-bool-id-comparison

  return operator_a.is_non_singular and operator_b.is_non_singular
