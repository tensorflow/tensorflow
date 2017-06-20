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
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
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
    return control_flow_ops.with_dependencies([
        check_ops.assert_none_equal(
            x,
            array_ops.zeros([], x.dtype),
            message="diagonal part must be non-zero")], x)

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


def shapes_from_loc_and_scale(loc, scale, name="shapes_from_loc_and_scale"):
  """Infer distribution batch and event shapes from a location and scale.

  Location and scale family distributions determine their batch/event shape by
  broadcasting the `loc` and `scale` args.  This helper does that broadcast,
  statically if possible.

  Batch shape broadcasts as per the normal rules.
  We allow the `loc` event shape to broadcast up to that of `scale`.  We do not
  allow `scale`'s event shape to change.  Therefore, the last dimension of `loc`
  must either be size `1`, or the same as `scale.range_dimension`.

  See `MultivariateNormalLinearOperator` for a usage example.

  Args:
    loc:  `N-D` `Tensor` with `N >= 1` (already converted to tensor) or `None`.
      If `None`, both batch and event shape are determined by `scale`.
    scale:  A `LinearOperator` instance.
    name:  A string name to prepend to created ops.

  Returns:
    batch_shape:  `TensorShape` (if broadcast is done statically), or `Tensor`.
    event_shape:  `TensorShape` (if broadcast is done statically), or `Tensor`.

  Raises:
    ValueError:  If the last dimension of `loc` is determined statically to be
      different than the range of `scale`.
  """
  with ops.name_scope(name, values=[loc] + scale.graph_parents):
    # Get event shape.
    event_size = scale.range_dimension_tensor()
    event_size_const = tensor_util.constant_value(event_size)
    if event_size_const is not None:
      event_shape = event_size_const.reshape([1])
    else:
      event_shape = event_size[array_ops.newaxis]

    # Static check that event shapes match.
    if loc is not None:
      loc_event_size = loc.get_shape()[-1].value
      if loc_event_size is not None and event_size_const is not None:
        if loc_event_size != 1 and loc_event_size != event_size_const:
          raise ValueError(
              "Event size of 'scale' (%d) could not be broadcast up to that of "
              "'loc' (%d)." % (loc_event_size, event_size_const))

    # Get batch shape.
    batch_shape = scale.batch_shape_tensor()
    if loc is None:
      batch_shape_const = tensor_util.constant_value(batch_shape)
      batch_shape = (
          batch_shape_const if batch_shape_const is not None else batch_shape)
    else:
      loc_batch_shape = loc.get_shape().with_rank_at_least(1)[:-1]
      if (loc.get_shape().ndims is None or
          not loc_batch_shape.is_fully_defined()):
        loc_batch_shape = array_ops.shape(loc)[:-1]
      else:
        loc_batch_shape = ops.convert_to_tensor(loc_batch_shape,
                                                name="loc_batch_shape")
      batch_shape = prefer_static_broadcast_shape(batch_shape, loc_batch_shape)

  return batch_shape, event_shape


def prefer_static_broadcast_shape(
    shape1, shape2, name="prefer_static_broadcast_shape"):
  """Convenience function which statically broadcasts shape when possible.

  Args:
    shape1:  `1-D` integer `Tensor`.  Already converted to tensor!
    shape2:  `1-D` integer `Tensor`.  Already converted to tensor!
    name:  A string name to prepend to created ops.

  Returns:
    The broadcast shape, either as `TensorShape` (if broadcast can be done
      statically), or as a `Tensor`.
  """
  with ops.name_scope(name, values=[shape1, shape2]):
    if (tensor_util.constant_value(shape1) is not None and
        tensor_util.constant_value(shape2) is not None):
      return array_ops.broadcast_static_shape(
          tensor_shape.TensorShape(tensor_util.constant_value(shape1)),
          tensor_shape.TensorShape(tensor_util.constant_value(shape2)))
    return array_ops.broadcast_dynamic_shape(shape1, shape2)


def is_diagonal_scale(scale):
  """Returns `True` if `scale` is a `LinearOperator` that is known to be diag.

  Args:
    scale:  `LinearOperator` instance.

  Returns:
    Python `bool`.

  Raises:
    TypeError:  If `scale` is not a `LinearOperator`.
  """
  if not isinstance(scale, linalg.LinearOperator):
    raise TypeError("Expected argument 'scale' to be instance of LinearOperator"
                    ". Found: %s" % scale)
  return (isinstance(scale, linalg.LinearOperatorIdentity) or
          isinstance(scale, linalg.LinearOperatorScaledIdentity) or
          isinstance(scale, linalg.LinearOperatorDiag))
