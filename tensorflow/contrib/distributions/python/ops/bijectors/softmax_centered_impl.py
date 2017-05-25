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
"""SoftmaxCentered bijector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import bijector


__all__ = [
    "SoftmaxCentered",
]


class SoftmaxCentered(bijector.Bijector):
  """Bijector which computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`.

  To implement [softmax](https://en.wikipedia.org/wiki/Softmax_function) as a
  bijection, the forward transformation appends a value to the input and the
  inverse removes this coordinate. The appended coordinate represents a pivot,
  e.g., `softmax(x) = exp(x-c) / sum(exp(x-c))` where `c` is the implicit last
  coordinate.

  Because we append a coordinate, this bijector only supports `event_ndim in [0,
  1]`, i.e., scalars and vectors.

  Example Use:

  ```python
  bijector.SoftmaxCentered(event_ndims=1).forward(tf.log([2, 3, 4]))
  # Result: [0.2, 0.3, 0.4, 0.1]
  # Extra result: 0.1

  bijector.SoftmaxCentered(event_ndims=1).inverse([0.2, 0.3, 0.4, 0.1])
  # Result: tf.log([2, 3, 4])
  # Extra coordinate removed.
  ```

  At first blush it may seem like the [Invariance of domain](
  https://en.wikipedia.org/wiki/Invariance_of_domain) theorem implies this
  implementation is not a bijection. However, the appended dimension
  makes the (forward) image non-open and the theorem does not directly apply.
  """

  def __init__(self,
               event_ndims=0,
               validate_args=False,
               name="softmax_centered"):
    self._graph_parents = []
    self._name = name
    with self._name_scope("init", values=[event_ndims]):
      event_ndims = ops.convert_to_tensor(event_ndims, name="event_ndims")
      event_ndims = tensor_util.constant_value(event_ndims)
      if event_ndims is None or event_ndims not in [0, 1]:
        raise ValueError("`event_ndims` must be a TF constant which is 0 or 1")
    self._static_event_ndims = event_ndims
    super(SoftmaxCentered, self).__init__(
        event_ndims=event_ndims,
        validate_args=validate_args,
        name=name)

  def _forward_event_shape(self, input_shape):
    if input_shape.ndims is None:
      return input_shape
    if input_shape.ndims != self._static_event_ndims:
      raise ValueError("input_shape.dims = %d != %d" %
                       (input_shape.ndims, self._static_event_ndims))
    if input_shape.ndims == 0:
      return tensor_shape.TensorShape([2])
    if input_shape.ndims == 1:
      return tensor_shape.TensorShape(input_shape[0] + 1)
    # Unreachable code:
    raise ValueError("event_ndims = %d must be 0 or 1" % input_shape.ndims)

  def _forward_event_shape_tensor(self, input_shape):
    ndims = array_ops.shape(input_shape)
    if self.validate_args:
      # It is not possible for a negative shape so we need only check <= 1.
      is_zero_or_one = check_ops.assert_equal(
          ndims, 0 if self._static_event_ndims == 0 else 1,
          message="event_ndims must be 0 or 1")
      ndims = control_flow_ops.with_dependencies([is_zero_or_one], ndims)
    if self._static_event_ndims == 0:
      return ops.convert_to_tensor(
          [2], dtype=dtypes.int32, name="output_shape")
    return input_shape + 1

  def _inverse_event_shape(self, output_shape):
    if output_shape.ndims is None:
      return output_shape
    if output_shape.ndims != 1:
      raise ValueError("output_shape.ndims = %d != 1" % output_shape.ndims)
    if self._static_event_ndims == 0:
      return tensor_shape.TensorShape([])
    return tensor_shape.TensorShape(output_shape[0] - 1)

  def _inverse_event_shape_tensor(self, output_shape):
    ndims = array_ops.shape(output_shape)[0]
    if self.validate_args:
      # It is not possible for a negative shape so we need only check <= 1.
      is_one = check_ops.assert_equal(
          ndims, 1, message="event_ndims must be 1")
      ndims = control_flow_ops.with_dependencies([is_one], ndims)
    if self._static_event_ndims == 0:
      return ops.convert_to_tensor([], dtype=dtypes.int32, name="output_shape")
    return array_ops.expand_dims(output_shape[0] - 1, dim=0)

  def _forward(self, x):
    # Pad the last dim with a zeros vector. We need this because it lets us
    # infer the scale in the inverse function.
    y = array_ops.expand_dims(x, dim=-1) if self._static_event_ndims == 0 else x
    ndims = (y.get_shape().ndims if y.get_shape().ndims is not None
             else array_ops.rank(y))
    y = array_ops.pad(y,
                      paddings=array_ops.concat(
                          (array_ops.zeros(
                              (ndims - 1, 2), dtype=dtypes.int32), [[0, 1]]),
                          0))

    # Set shape hints.
    if x.get_shape().ndims is not None:
      shape = x.get_shape().as_list()
      if self._static_event_ndims == 0:
        shape += [2]
      elif shape[-1] is not None:
        shape[-1] += 1
      shape = tensor_shape.TensorShape(shape)
      y.get_shape().assert_is_compatible_with(shape)
      y.set_shape(shape)

    # Since we only support event_ndims in [0, 1] and we do padding, we always
    # reduce over the last dimension, i.e., dim=-1 (which is the default).
    return nn_ops.softmax(y)

  def _inverse(self, y):
    # To derive the inverse mapping note that:
    #   y[i] = exp(x[i]) / normalization
    # and
    #   y[end] = 1 / normalization.
    # Thus:
    # x[i] = log(exp(x[i])) - log(y[end]) - log(normalization)
    #      = log(exp(x[i])/normalization) - log(y[end])
    #      = log(y[i]) - log(y[end])
    shape = (np.asarray(y.get_shape().as_list(), dtype=np.int32)
             if y.get_shape().is_fully_defined()
             else array_ops.shape(y, name="shape"))
    ndims = y.get_shape().ndims or math_ops.rank(y, name="ndims")

    # Do this first to make sure CSE catches that it'll happen again in
    # _inverse_log_det_jacobian.
    x = math_ops.log(y)

    # We now extract the last coordinate of the rightmost dimension.
    # Our trick is to slice from [0,0,...,shape[-1]-1] to shape[:-1]+[1].
    begin = array_ops.one_hot(indices=ndims-1,
                              depth=ndims,
                              on_value=shape[-1]-np.array(1, dtype=shape.dtype),
                              dtype=shape.dtype)
    size = array_ops.concat([shape[:-1], np.asarray([1], dtype=shape.dtype)], 0)
    log_normalization = -array_ops.strided_slice(x, begin, begin + size)

    # Here we slice out all but the last coordinate; see above for idea.
    begin = array_ops.zeros_like(shape)
    size = array_ops.concat([shape[:-1], [shape[-1] - 1]], 0)
    x = array_ops.strided_slice(x, begin, begin + size)

    x += log_normalization

    if self._static_event_ndims == 0:
      x = array_ops.squeeze(x, squeeze_dims=[ndims-1])

    # Set shape hints.
    if y.get_shape().ndims is not None:
      shape = y.get_shape().as_list()
      if self._static_event_ndims == 0:
        shape = shape[:-1]
      elif shape[-1] is not None:
        shape[-1] -= 1
      shape = tensor_shape.TensorShape(shape)
      x.get_shape().assert_is_compatible_with(shape)
      x.set_shape(shape)

    return x

  def _inverse_log_det_jacobian(self, y):
    # WLOG, consider the vector case:
    #   x = log(y[:-1]) - log(y[-1])
    # where,
    #   y[-1] = 1 - sum(y[:-1]).
    # We have:
    #   det{ dX/dY } = det{ diag(1 ./ y[:-1]) + 1 / y[-1] }
    #                = det{ inv{ diag(y[:-1]) - y[:-1]' y[:-1] } }   (1)
    #                = 1 / det{ diag(y[:-1]) - y[:-1]' y[:-1] }
    #                = 1 / { (1 + y[:-1]' inv(diag(y[:-1])) y[:-1]) *
    #                        det(diag(y[:-1])) }                     (2)
    #                = 1 / { y[-1] prod(y[:-1]) }
    #                = 1 / prod(y)
    # (1) - https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    #       or by noting that det{ dX/dY } = 1 / det{ dY/dX } from Bijector
    #       docstring "Tip".
    # (2) - https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    return -math_ops.reduce_sum(math_ops.log(y), axis=-1)

  def _forward_log_det_jacobian(self, x):
    if self._static_event_ndims == 0:
      return x - 2. * nn_ops.softplus(x)
    else:
      # This code is similar to nn_ops.log_softmax but different because we have
      # an implicit zero column to handle. I.e., instead of:
      #   reduce_sum(logits - reduce_sum(exp(logits), dim))
      # we must do:
      #   log_normalization = 1 + reduce_sum(exp(logits))
      #   -log_normalization + reduce_sum(logits - log_normalization)
      log_normalization = nn_ops.softplus(
          math_ops.reduce_logsumexp(x, axis=-1, keep_dims=True))
      fldj = (-log_normalization +
              math_ops.reduce_sum(x - log_normalization,
                                  axis=-1,
                                  keep_dims=True))
      return array_ops.squeeze(fldj, squeeze_dims=-1)
