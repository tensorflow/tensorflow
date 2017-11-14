# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Reshape bijectors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector as bijector_lib


__all__ = [
    "Reshape",
]


class Reshape(bijector_lib.Bijector):
  """Reshapes the `event_shape` of a `Tensor`.

  The semantics generally follow that of `tf.reshape()`, with
  a few differences:
   * The user must provide both the input and output shape, so that
     the transformation can be inverted.
   * The `Reshape` bijector automatically broadcasts over the leftmost
     dimensions of its input (`sample_shape` and `batch_shape`); only
     the rightmost `event_ndims_in` dimensions are reshaped. The
     number of dimensions to reshape is inferred from the provided
     `event_shape_in` (`event_ndims_in = len(event_shape_in)`).
   * The `Reshape` bijector does not currently support
     partially-specified shapes, i.e., those with a dimension
     implicitly specified by `-1`.

  Example usage:
  ```python

  bs = tf.contrib.distributions.bijectors

  reverse = bs.Reshape(event_shape_out=[1,2],
                       event_shape_in=[2,])

  reverse.forward([1., 2.])    # shape [2,]
  # ==> [[1., 2.]]             # shape [1,2]

  reverse.forward([[1., 2.], [3., 4.]])  # shape [2, 2]
  # ==> [[[1., 2.]], [[3., 4.]]]         # shape [2, 1, 2]

  reverse.inverse([[1., 2.]])  # shape [1,2]
  # ==> [1., 2.]               # shape [2,]

  reverse.forward_log_det_jacobian(any_value)
  # ==> 0.

  reverse.inverse_log_det_jacobian(any_value)
  # ==> 0.
  ```

  """

  def __init__(self, event_shape_out, event_shape_in,
               validate_args=False, name=None):
    """Creates a `Reshape` bijector.

    Args:
      event_shape_out: An `int`-like vector-shaped `Tensor`
        representing the fully specified (no -1's) event shape of the
        transformed output.
      event_shape_in: An `int`-like vector-shaped `Tensor`
        representing the fully specified (no -1's) event shape of the
        input.
      validate_args: Python `bool` indicating whether arguments should
        be checked for correctness.
      name: Python `str`, name given to ops managed by this object.

    Raises:
      TypeError: if either `event_shape_in` or `event_shape_out` has
       non-vector shape (`rank > 1`), or non-integer `dtype`.
      ValueError: if either `event_shape_in` or `event_shape_out`
       contains non-positive entries, or if their sizes do not match
       (`prod(event_shape_in)` != `prod(event_shape_out)`), or if
       their dimensionality(s) cannot be statically inferred.
    """
    with ops.name_scope(name, "reshape",
                        values=[event_shape_out, event_shape_in]):

      event_shape_out = ops.convert_to_tensor(event_shape_out,
                                              name="event_shape_out",
                                              preferred_dtype=dtypes.int32)
      event_shape_in = ops.convert_to_tensor(event_shape_in,
                                             name="event_shape_in",
                                             preferred_dtype=dtypes.int32)

      # check that input shapes are positive integers
      assertions = []
      assertions += self._maybe_check_valid_shape(
          event_shape_out, "event_shape_out",
          validate_args=validate_args)
      assertions += self._maybe_check_valid_shape(
          event_shape_in, "event_shape_in", validate_args=validate_args)

      # check that prod(event_shape_in) = prod(event_shape_out)
      assertions += self._maybe_check_matching_sizes(
          event_shape_in, event_shape_out, validate_args=validate_args)

      self._assertions = assertions
      self._event_shape_in = event_shape_in
      self._event_shape_out = event_shape_out
      self._event_shape_in_static = tensor_util.constant_value_as_shape(
          event_shape_in)
      self._event_shape_out_static = tensor_util.constant_value_as_shape(
          event_shape_out)

      super(Reshape, self).__init__(is_constant_jacobian=True,
                                    validate_args=validate_args,
                                    name=name or "reshape")

  def _maybe_check_valid_shape(self, shape_tensor, label,
                               validate_args=False):
    """Check that a shape Tensor is int-type and positive."""

    assertions = []

    if not shape_tensor.dtype.is_integer:
      raise TypeError("{} dtype ({}) should be `int`-like.".format(
          label, shape_tensor.dtype.name))

    shape_rank = tensor_util.constant_value(array_ops.rank(shape_tensor))
    if shape_rank is not None and shape_rank > 1:
      raise ValueError("{} rank should be <= 1.".format(label))

    s = tensor_util.constant_value(shape_tensor)
    if s is not None:
      if (s <= 0).any():
        raise ValueError("{} entries must be positive, but found {}".format(
            label, s))
    elif validate_args:
      assertions.append(check_ops.assert_positive(
          shape_tensor, message="{} entries must be positive".format(label)))

    return assertions

  def _maybe_check_matching_sizes(self, event_shape_in, event_shape_out,
                                  validate_args=False):
    """Check that prod(event_shape_in)==prod(event_shape_out)."""

    def _get_size_from_shape(shape):
      """Computes size from a shape `Tensor`, statically if possible."""
      s = tensor_util.constant_value(shape)
      if s is not None:
        return [np.int32(np.prod(s))]*2
      return None, math_ops.reduce_prod(shape, name="size")

    # Ensure `event_shape_in` is compatible with `event_shape_out`.
    event_size_in_, event_size_in = _get_size_from_shape(  # pylint: disable=unbalanced-tuple-unpacking
        event_shape_in)
    event_size_out_, event_size_out = _get_size_from_shape(  # pylint: disable=unbalanced-tuple-unpacking
        event_shape_out)

    assertions = []
    if event_size_in_ is not None and event_size_out_ is not None:
      if event_size_in_ != event_size_out_:
        raise ValueError(
            "Input `event_size` ({}) does not match output `event_size` ({}).".
            format(event_size_in, event_size_out_))
    elif validate_args:
      assertions.append(check_ops.assert_equal(
          event_size_in, event_size_out,
          message="Input/output `event_size`s do not match."))

    return assertions

  def _reshape_helper(self, x, event_shape_in, event_shape_out):
    """Reshape only the event_shape of an input `Tensor`."""

    def _get_rank_from_shape(shape):
      """Computes rank from a shape `Tensor`, statically if possible."""
      # Uses fact that rank is "shape of shape".
      ndims = shape.shape.with_rank_at_least(1)[0].value
      if ndims is not None:
        return ndims, ndims
      return None, array_ops.shape(shape)[0]

    event_ndims_in_, event_ndims_in = _get_rank_from_shape(event_shape_in)

    assertions = []
    # Ensure x.event_shape is compatible with event_shape_in.
    if x.shape.ndims is not None:
      x_ndims_, x_ndims = [x.shape.ndims]*2
    else:
      x_ndims_, x_ndims = None, array_ops.rank(x)

    if (event_ndims_in_ is not None
        and x_ndims_ is not None
        and x.shape.with_rank_at_least(event_ndims_in_)[
            x_ndims_-event_ndims_in_:].is_fully_defined()):
      x_event_shape_, x_event_shape = [  # pylint: disable=unbalanced-tuple-unpacking
          np.int32(x.shape[x_ndims_-event_ndims_in_:])]*2
    else:
      x_event_shape_, x_event_shape = (
          None, array_ops.shape(x)[x_ndims-event_ndims_in:])

    event_shape_in_ = tensor_util.constant_value(event_shape_in)

    if x_event_shape_ is not None and event_shape_in_ is not None:
      if not np.equal(x_event_shape_, event_shape_in_).all():
        raise ValueError(
            "Input `event_shape` ({}) does not match `event_shape_in` ({}).".
            format(x_event_shape_, event_shape_in_))
    elif self.validate_args:
      assertions.append(check_ops.assert_equal(
          x_event_shape, event_shape_in,
          message="Input `event_shape` does not match `event_shape_in`."))

    if assertions:
      x = control_flow_ops.with_dependencies(assertions, x)

    # get the parts of shape(x) that will not change
    sample_and_batch_shape = array_ops.shape(x)

    ndims = (x.shape.ndims if x.shape.ndims is not None
             else array_ops.rank(x))
    sample_and_batch_shape = sample_and_batch_shape[
        :(ndims - math_ops.abs(event_ndims_in))]

    new_shape = array_ops.concat(
        [sample_and_batch_shape, event_shape_out], axis=0)

    return array_ops.reshape(x, new_shape)

  def _forward(self, x):
    with ops.control_dependencies(self._assertions):
      return self._reshape_helper(x,
                                  self._event_shape_in,
                                  self._event_shape_out)

  def _inverse(self, y):
    with ops.control_dependencies(self._assertions):
      return self._reshape_helper(y,
                                  self._event_shape_out,
                                  self._event_shape_in)

  def _inverse_log_det_jacobian(self, y):
    with ops.control_dependencies(self._assertions):
      return constant_op.constant(0., dtype=y.dtype)

  def _forward_log_det_jacobian(self, x):
    with ops.control_dependencies(self._assertions):
      return constant_op.constant(0., dtype=x.dtype)

  def _forward_event_shape(self, input_shape):
    self._event_shape_in_static.assert_is_compatible_with(input_shape)
    return self._event_shape_out_static

  def _inverse_event_shape(self, output_shape):
    self._event_shape_out_static.assert_is_compatible_with(output_shape)
    return self._event_shape_in_static

  def _forward_event_shape_tensor(self, input_shape):
    input_assertions = self._maybe_check_valid_shape(
        input_shape, "input event shape", validate_args=self.validate_args)
    input_assertions += self._maybe_check_matching_sizes(
        input_shape, self._event_shape_out,
        validate_args=self.validate_args)

    return control_flow_ops.with_dependencies(
        input_assertions + self._assertions, self._event_shape_out)

  def _inverse_event_shape_tensor(self, output_shape):

    output_assertions = self._maybe_check_valid_shape(
        output_shape, "output event shape", validate_args=self.validate_args)
    output_assertions += self._maybe_check_matching_sizes(
        output_shape, self._event_shape_in, validate_args=self.validate_args)

    return control_flow_ops.with_dependencies(
        output_assertions + self._assertions, self._event_shape_in)
