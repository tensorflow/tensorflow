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
"""Utilities for probability distributions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib import linalg
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util
from tensorflow.python.ops.distributions.util import *  # pylint: disable=wildcard-import


# TODO(b/35290280): Add unit-tests.
def make_diag_scale(loc, scale_diag, scale_identity_multiplier,
                    validate_args, assert_positive, name=None):
  """Creates a LinOp from `scale_diag`, `scale_identity_multiplier` kwargs."""
  def _convert_to_tensor(x, name):
    return None if x is None else ops.convert_to_tensor(x, name=name)

  def _maybe_attach_assertion(x):
    if not validate_args:
      return x
    if assert_positive:
      return control_flow_ops.with_dependencies([
          check_ops.assert_positive(
              x, message="diagonal part must be positive"),
      ], x)
    # TODO(b/35157376): Use `assert_none_equal` once it exists.
    return control_flow_ops.with_dependencies([
        check_ops.assert_greater(
            math_ops.abs(x),
            array_ops.zeros([], x.dtype),
            message="diagonal part must be non-zero"),
    ], x)

  with ops.name_scope(name, "make_diag_scale",
                      values=[loc, scale_diag, scale_identity_multiplier]):
    loc = _convert_to_tensor(loc, name="loc")
    scale_diag = _convert_to_tensor(scale_diag, name="scale_diag")
    scale_identity_multiplier = _convert_to_tensor(
        scale_identity_multiplier,
        name="scale_identity_multiplier")

    if scale_diag is not None:
      if scale_identity_multiplier is not None:
        scale_diag += scale_identity_multiplier[..., array_ops.newaxis]
      return linalg.LinearOperatorDiag(
          diag=_maybe_attach_assertion(scale_diag),
          is_non_singular=True,
          is_self_adjoint=True,
          is_positive_definite=assert_positive)

    # TODO(b/35290280): Consider inferring shape from scale_perturb_factor.
    if loc is None:
      raise ValueError(
          "Cannot infer `event_shape` unless `loc` is specified.")

    num_rows = util.dimension_size(loc, -1)

    if scale_identity_multiplier is None:
      return linalg.LinearOperatorIdentity(
          num_rows=num_rows,
          dtype=loc.dtype.base_dtype,
          is_self_adjoint=True,
          is_positive_definite=True,
          assert_proper_shapes=validate_args)

    return linalg.LinearOperatorScaledIdentity(
        num_rows=num_rows,
        multiplier=_maybe_attach_assertion(scale_identity_multiplier),
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=assert_positive,
        assert_proper_shapes=validate_args)
