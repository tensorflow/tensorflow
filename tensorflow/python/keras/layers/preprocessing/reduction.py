# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras categorical preprocessing layers."""
# pylint: disable=g-classes-have-attributes

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging


def get_reduce_op(reduction_str):
  """Translate a reduction string name to a reduction op."""
  if reduction_str == "max":
    return math_ops.reduce_max
  elif reduction_str == "mean":
    return math_ops.reduce_mean
  elif reduction_str == "min":
    return math_ops.reduce_min
  elif reduction_str == "prod":
    return math_ops.reduce_prod
  elif reduction_str == "sum":
    return math_ops.reduce_sum
  else:
    raise ValueError("Reduction %s is not supported for unweighted inputs." %
                     reduction_str)


class Reduction(Layer):
  """Performs an optionally-weighted reduction.

  This layer performs a reduction across one axis of its input data. This
  data may optionally be weighted by passing in an identical float tensor.

  Arguments:
    reduction: The type of reduction to perform. Can be one of the following:
      "max", "mean", "min", "prod", or "sum". This layer uses the Tensorflow
      reduce op which corresponds to that reduction (so, for "mean", we use
      "reduce_mean").
    axis: The axis to reduce along. Defaults to '-2', which is usually the axis
      that contains embeddings (but is not within the embedding itself).

  Input shape:
    A tensor of 2 or more dimensions of any numeric dtype.

  Output:
    A tensor of 1 less dimension than the input tensor, of the same dtype.

  Call arguments:
    inputs: The data to reduce.
    weights: An optional tensor or constant of the same shape as inputs that
      will weight the input data before it is reduced.
  """
  # TODO(momernick): Add example here.

  def __init__(self, reduction, axis=-2, **kwargs):
    self.reduction = reduction
    self.axis = axis
    # We temporarily turn off autocasting, as it does not apply to named call
    # kwargs.
    super(Reduction, self).__init__(**kwargs)
    self._supports_ragged_inputs = True

  def call(self, inputs, weights=None):
    # If we are not weighting the inputs we can immediately reduce the data
    # and return it.
    if weights is None:
      return get_reduce_op(self.reduction)(inputs, axis=self.axis)

    # TODO(momernick): Add checks for this and a decent error message if the
    # weight shape isn't compatible.
    if weights.shape.rank + 1 == inputs.shape.rank:
      weights = array_ops.expand_dims(weights, -1)

    weighted_inputs = math_ops.multiply(inputs, weights)

    # Weighted sum and prod can be expressed as reductions over the weighted
    # values, as can min and max.
    if self.reduction in ("sum", "prod", "min", "max"):
      return get_reduce_op(self.reduction)(weighted_inputs, axis=self.axis)

    # Weighted mean is a bit more complicated: we have to do a sum of the
    # weighted values and divide by the sum of the weights.
    if self.reduction == "mean":
      input_sum = math_ops.reduce_sum(weighted_inputs, axis=self.axis)
      weight_sum = math_ops.reduce_sum(weights, axis=self.axis)
      return math_ops.divide(input_sum, weight_sum)

    # sqrtn is also more complicated: it's like mean but with a normalized
    # divisor.
    if self.reduction == "sqrtn":
      logging.warning("Reduction `sqrtn` is deprecated and will be removed "
                      "2021-01-01. Please use the `sum` reduction and divide "
                      "the output by the normalized weights instead.")
      input_sum = math_ops.reduce_sum(weighted_inputs, axis=self.axis)
      squared_weights = math_ops.pow(weights, 2)
      squared_weights_sum = math_ops.reduce_sum(squared_weights, axis=self.axis)
      sqrt_weights_sum = math_ops.sqrt(squared_weights_sum)
      return math_ops.divide(input_sum, sqrt_weights_sum)

    raise ValueError("%s is not a supported weighted reduction." %
                     self.reduction)
