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

from tensorflow.contrib.distributions.python.ops import distribution_util
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops.distributions import bijector
from tensorflow.python.util import deprecation


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

  Example Use:

  ```python
  bijector.SoftmaxCentered().forward(tf.log([2, 3, 4]))
  # Result: [0.2, 0.3, 0.4, 0.1]
  # Extra result: 0.1

  bijector.SoftmaxCentered().inverse([0.2, 0.3, 0.4, 0.1])
  # Result: tf.log([2, 3, 4])
  # Extra coordinate removed.
  ```

  At first blush it may seem like the [Invariance of domain](
  https://en.wikipedia.org/wiki/Invariance_of_domain) theorem implies this
  implementation is not a bijection. However, the appended dimension
  makes the (forward) image non-open and the theorem does not directly apply.
  """

  @deprecation.deprecated(
      "2018-10-01",
      "The TensorFlow Distributions library has moved to "
      "TensorFlow Probability "
      "(https://github.com/tensorflow/probability). You "
      "should update all references to use `tfp.distributions` "
      "instead of `tf.contrib.distributions`.",
      warn_once=True)
  def __init__(self,
               validate_args=False,
               name="softmax_centered"):
    self._graph_parents = []
    self._name = name
    super(SoftmaxCentered, self).__init__(
        forward_min_event_ndims=1,
        validate_args=validate_args,
        name=name)

  def _forward_event_shape(self, input_shape):
    if input_shape.ndims is None or input_shape[-1] is None:
      return input_shape
    return tensor_shape.TensorShape([input_shape[-1] + 1])

  def _forward_event_shape_tensor(self, input_shape):
    return (input_shape[-1] + 1)[..., array_ops.newaxis]

  def _inverse_event_shape(self, output_shape):
    if output_shape.ndims is None or output_shape[-1] is None:
      return output_shape
    if output_shape[-1] <= 1:
      raise ValueError("output_shape[-1] = %d <= 1" % output_shape[-1])
    return tensor_shape.TensorShape([output_shape[-1] - 1])

  def _inverse_event_shape_tensor(self, output_shape):
    if self.validate_args:
      # It is not possible for a negative shape so we need only check <= 1.
      is_greater_one = check_ops.assert_greater(
          output_shape[-1], 1, message="Need last dimension greater than 1.")
      output_shape = control_flow_ops.with_dependencies(
          [is_greater_one], output_shape)
    return (output_shape[-1] - 1)[..., array_ops.newaxis]

  def _forward(self, x):
    # Pad the last dim with a zeros vector. We need this because it lets us
    # infer the scale in the inverse function.
    y = distribution_util.pad(x, axis=-1, back=True)

    # Set shape hints.
    if x.shape.ndims is not None:
      shape = x.shape[:-1].concatenate(x.shape[-1] + 1)
      y.shape.assert_is_compatible_with(shape)
      y.set_shape(shape)

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

    # Do this first to make sure CSE catches that it'll happen again in
    # _inverse_log_det_jacobian.
    x = math_ops.log(y)

    log_normalization = (-x[..., -1])[..., array_ops.newaxis]
    x = x[..., :-1] + log_normalization

    # Set shape hints.
    if y.shape.ndims is not None:
      shape = y.shape[:-1].concatenate(y.shape[-1] - 1)
      x.shape.assert_is_compatible_with(shape)
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
    # This code is similar to nn_ops.log_softmax but different because we have
    # an implicit zero column to handle. I.e., instead of:
    #   reduce_sum(logits - reduce_sum(exp(logits), dim))
    # we must do:
    #   log_normalization = 1 + reduce_sum(exp(logits))
    #   -log_normalization + reduce_sum(logits - log_normalization)
    log_normalization = nn_ops.softplus(
        math_ops.reduce_logsumexp(x, axis=-1, keep_dims=True))
    return array_ops.squeeze(
        (-log_normalization + math_ops.reduce_sum(
            x - log_normalization, axis=-1, keepdims=True)), axis=-1)
