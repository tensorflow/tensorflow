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
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.util import deprecation


__all__ = [
    "Reshape",
]


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def _static_ndims_from_shape(shape):
  return tensor_shape.dimension_value(shape.shape.with_rank_at_least(1)[0])


@deprecation.deprecated(
    "2018-10-01",
    "The TensorFlow Distributions library has moved to "
    "TensorFlow Probability "
    "(https://github.com/tensorflow/probability). You "
    "should update all references to use `tfp.distributions` "
    "instead of `tf.contrib.distributions`.",
    warn_once=True)
def _ndims_from_shape(shape):
  return array_ops.shape(shape)[0]


class Reshape(bijector.Bijector):
  """Reshapes the `event_shape` of a `Tensor`.

  The semantics generally follow that of `tf.reshape()`, with
  a few differences:

  * The user must provide both the input and output shape, so that
    the transformation can be inverted. If an input shape is not
    specified, the default assumes a vector-shaped input, i.e.,
    event_shape_in = (-1,).
  * The `Reshape` bijector automatically broadcasts over the leftmost
    dimensions of its input (`sample_shape` and `batch_shape`); only
    the rightmost `event_ndims_in` dimensions are reshaped. The
    number of dimensions to reshape is inferred from the provided
    `event_shape_in` (`event_ndims_in = len(event_shape_in)`).

  Example usage:
  ```python

  import tensorflow_probability as tfp
  tfb = tfp.bijectors

  r = tfb.Reshape(event_shape_out=[1, -1])

  r.forward([3., 4.])    # shape [2]
  # ==> [[3., 4.]]       # shape [1, 2]

  r.forward([[1., 2.], [3., 4.]])  # shape [2, 2]
  # ==> [[[1., 2.]],
  #      [[3., 4.]]]   # shape [2, 1, 2]

  r.inverse([[3., 4.]])  # shape [1,2]
  # ==> [3., 4.]         # shape [2]

  r.forward_log_det_jacobian(any_value)
  # ==> 0.

  r.inverse_log_det_jacobian(any_value)
  # ==> 0.
  ```

  """

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self, event_shape_out, event_shape_in=(-1,),
               validate_args=False, name=None):
    """Creates a `Reshape` bijector.

    Args:
      event_shape_out: An `int`-like vector-shaped `Tensor`
        representing the event shape of the transformed output.
      event_shape_in: An optional `int`-like vector-shape `Tensor`
        representing the event shape of the input. This is required in
        order to define inverse operations; the default of (-1,)
        assumes a vector-shaped input.
      validate_args: Python `bool` indicating whether arguments should
        be checked for correctness.
      name: Python `str`, name given to ops managed by this object.

    Raises:
      TypeError: if either `event_shape_in` or `event_shape_out` has
        non-integer `dtype`.
      ValueError: if either of `event_shape_in` or `event_shape_out`
       has non-vector shape (`rank > 1`), or if their sizes do not
       match.
    """
    with ops.name_scope(name, "reshape",
                        values=[event_shape_out, event_shape_in]):

      event_shape_out = ops.convert_to_tensor(event_shape_out,
                                              name="event_shape_out",
                                              preferred_dtype=dtypes.int32)
      event_shape_in = ops.convert_to_tensor(event_shape_in,
                                             name="event_shape_in",
                                             preferred_dtype=dtypes.int32)

      assertions = []
      assertions.extend(self._maybe_check_valid_shape(
          event_shape_out, validate_args))
      assertions.extend(self._maybe_check_valid_shape(
          event_shape_in, validate_args))

      self._assertions = assertions
      self._event_shape_in = event_shape_in
      self._event_shape_out = event_shape_out

      super(Reshape, self).__init__(
          forward_min_event_ndims=0,
          is_constant_jacobian=True,
          validate_args=validate_args,
          name=name or "reshape")

  def _maybe_check_valid_shape(self, shape, validate_args):
    """Check that a shape Tensor is int-type and otherwise sane."""
    if not shape.dtype.is_integer:
      raise TypeError("{} dtype ({}) should be `int`-like.".format(
          shape, shape.dtype.name))

    assertions = []

    ndims = array_ops.rank(shape)
    ndims_ = tensor_util.constant_value(ndims)
    if ndims_ is not None and ndims_ > 1:
      raise ValueError("`{}` rank ({}) should be <= 1.".format(
          shape, ndims_))
    elif validate_args:
      assertions.append(check_ops.assert_less_equal(
          ndims, 1, message="`{}` rank should be <= 1.".format(shape)))

    shape_ = tensor_util.constant_value_as_shape(shape)
    if shape_.is_fully_defined():
      es = np.int32(shape_.as_list())
      if sum(es == -1) > 1:
        raise ValueError(
            "`{}` must have at most one `-1` (given {})"
            .format(shape, es))
      if np.any(es < -1):
        raise ValueError(
            "`{}` elements must be either positive integers or `-1`"
            "(given {})."
            .format(shape, es))
    elif validate_args:
      assertions.extend([
          check_ops.assert_less_equal(
              math_ops.reduce_sum(
                  math_ops.cast(math_ops.equal(shape, -1), dtypes.int32)),
              1,
              message="`{}` elements must have at most one `-1`."
              .format(shape)),
          check_ops.assert_greater_equal(
              shape, -1,
              message="`{}` elements must be either positive integers or `-1`."
              .format(shape)),
      ])
    return assertions

  def _reshape_helper(self, x, event_shape_in, event_shape_out):
    """Reshape only the event_shape of an input `Tensor`."""

    event_ndims_in_ = _static_ndims_from_shape(event_shape_in)
    event_ndims_in = _ndims_from_shape(event_shape_in)
    x_ndims_, x_ndims = x.shape.ndims, array_ops.rank(x)

    assertions = []

    # Ensure x.event_shape is compatible with event_shape_in.
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
      # Compare the shape dimensions that are fully specified in the
      # input (i.e., for which event_shape_in is not -1). If x_event_shape
      # matches along all of these dimensions, it is compatible with
      # the desired input shape and any further mismatches (i.e.,
      # imcompatibility with the desired *output* shape) will be
      # caught inside of array_ops.reshape() below.
      x_event_shape_specified_ = x_event_shape_[event_shape_in_ >= 0]
      event_shape_in_specified_ = event_shape_in_[event_shape_in_ >= 0]
      if not np.equal(x_event_shape_specified_,
                      event_shape_in_specified_).all():
        raise ValueError(
            "Input `event_shape` does not match `event_shape_in` ({} vs {}).".
            format(x_event_shape_, event_shape_in_))
    elif self.validate_args:
      # Similarly to the static case, we compare the shape dimensions
      # that are fully specified in the input. We extract these
      # dimensions using boolean_mask(), which requires that the mask
      # have known ndims. We can assume that shape Tensors always have
      # ndims==1 (this assumption is verified inside of
      # _maybe_check_valid_shape), so the reshape operation is just a
      # no-op that formally encodes this fact to make boolean_mask()
      # happy.
      event_shape_mask = array_ops.reshape(event_shape_in >= 0, [-1])
      x_event_shape_specified = array_ops.boolean_mask(x_event_shape,
                                                       event_shape_mask)
      event_shape_in_specified = array_ops.boolean_mask(event_shape_in,
                                                        event_shape_mask)
      assertions.append(check_ops.assert_equal(
          x_event_shape_specified, event_shape_in_specified,
          message="Input `event_shape` does not match `event_shape_in`."))

    if assertions:
      x = control_flow_ops.with_dependencies(assertions, x)

    # get the parts of shape(x) that will not change
    sample_and_batch_shape = array_ops.shape(x)

    ndims = (x.shape.ndims if x.shape.ndims is not None
             else array_ops.rank(x))
    sample_and_batch_shape = sample_and_batch_shape[
        :(ndims - math_ops.abs(event_ndims_in))]

    if (event_ndims_in_ is not None
        and x_ndims_ is not None
        and event_ndims_in_ == x_ndims_):
      # Hack to allow forward/inverse_event_shape to do shape
      # inference by calling this helper method with a dummy Tensor of
      # shape event_shape_in. In this special case,
      # sample_and_batch_shape will be empty so we can preserve static
      # shape information by avoiding the concat operation below
      # (which would be a no-op).
      new_shape = event_shape_out
    else:
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
    # NOTE: this method and the other *_event_shape* methods
    # compute shape by explicit transformation of a dummy
    # variable. This approach is not generally recommended because it
    # bloats the graph and could in general trigger side effects.
    #
    # In this particular case of the Reshape bijector, the
    # forward and inverse transforms have no side effects, and we
    # believe the reduction in code complexity from delegating the
    # heavy lifting to tf.reshape() is worth the added graph ops.
    # However, you should think hard before implementing this approach
    # in other Bijectors; it is strongly preferred to compute
    # shapes explicitly whenever it's feasible to do so.
    with ops.control_dependencies(self._assertions):
      dummy = array_ops.zeros(dtype=dtypes.float32, shape=input_shape)
      dummy_reshaped = self.forward(dummy)
      return dummy_reshaped.shape

  def _inverse_event_shape(self, output_shape):
    with ops.control_dependencies(self._assertions):
      dummy = array_ops.zeros(dtype=dtypes.float32, shape=output_shape)
      dummy_reshaped = self.inverse(dummy)
      return dummy_reshaped.shape

  def _forward_event_shape_tensor(self, input_shape):
    with ops.control_dependencies(self._assertions):
      dummy = array_ops.zeros(dtype=dtypes.float32, shape=input_shape)
      dummy_reshaped = self.forward(dummy)
      return array_ops.shape(dummy_reshaped)

  def _inverse_event_shape_tensor(self, output_shape):
    with ops.control_dependencies(self._assertions):
      dummy = array_ops.zeros(dtype=dtypes.float32, shape=output_shape)
      dummy_reshaped = self.inverse(dummy)
      return array_ops.shape(dummy_reshaped)
