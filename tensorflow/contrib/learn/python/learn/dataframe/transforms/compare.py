# Copyright 2016 Google Inc. All Rights Reserved.
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

"""Transforms for comparing pairs of `Series`."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import series
from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

# Each entry is a mapping from registered_name to operation. Each operation is
# wrapped in a transform and then registered as a member function
# `Series`.registered_name().
COMPARISON_TRANSFORMS = [("__eq__", math_ops.equal),
                         ("__gt__", math_ops.greater),
                         ("__ge__", math_ops.greater_equal),
                         ("__lt__", math_ops.less),
                         ("__le__", math_ops.less_equal)]

SERIES_DOC_FORMAT_STRING = (
    "A `Transform` that uses `{0}` to compare two Series. "
    "Documentation for `{0}`: \n\n {1}"
)

SCALAR_DOC_FORMAT_STRING = (
    "A `Transform` that uses `{0}` to compare a Series and a scalar. "
    "Documentation for `{0}`: \n\n {1}"
)


class SeriesComparisonTransform(transform.Transform):
  """Parent class for `Transform`s that compare `Series` elementwise."""

  @property
  def input_valency(self):
    return 2

  @property
  def _output_names(self):
    return "output",

  def _apply_transform(self, input_tensors):
    # TODO(jamieas): consider supporting sparse comparisons.
    if isinstance(input_tensors[0], ops.SparseTensor) or isinstance(
        input_tensors[1], ops.SparseTensor):
      raise TypeError("{} does not support SparseTensors".format(type(
          self).__name__))

    # pylint: disable=not-callable
    return self.return_type(self._compare(input_tensors[0], input_tensors[1]))


class ScalarComparisonTransform(transform.Transform):
  """Parent class for `Transform`s that compare `Series` to a scalar."""

  def __init__(self, threshold):
    if isinstance(threshold, series.Series):
      raise ValueError(
          "{} is used to compare Series with scalars. It was called with "
          "another Series.".format(
              type(self).__name__))
    super(ScalarComparisonTransform, self).__init__()
    self._threshold = threshold

  @transform.parameter
  def threshold(self):
    return self._threshold

  @property
  def input_valency(self):
    return 1

  @property
  def _output_names(self):
    return "output",

  def _apply_transform(self, input_tensors):
    input_tensor = input_tensors[0]
    if isinstance(input_tensor, ops.SparseTensor):
      result = ops.SparseTensor(input_tensor.indices,
                                self._compare(input_tensor.values),
                                input_tensor.shape)
    else:
      result = self._compare(input_tensor)

    # pylint: disable=not-callable
    return self.return_type(result)


# pylint: disable=unused-argument
def register_comparison_ops(method_name, operation):
  """Registers `Series` member functions for comparisons.

  Args:
    method_name: the name of the method that will be created in `Series`.
    operation: TensorFlow operation used for comparison.
  """

  # Define series-series comparison `Transform`.
  @property
  def series_name(self):
    return operation.__name__

  series_doc = SERIES_DOC_FORMAT_STRING.format(operation.__name__,
                                               operation.__doc__)
  def series_compare(self, x, y):
    return operation(x, y)

  series_transform_cls = type("scalar_{}".format(operation.__name__),
                              (SeriesComparisonTransform,),
                              {"name": series_name,
                               "__doc__": series_doc,
                               "_compare": series_compare})

  # Define series-scalar comparison `Transform`.
  @property
  def scalar_name(self):
    return "scalar_{}".format(operation.__name__)

  scalar_doc = SCALAR_DOC_FORMAT_STRING.format(operation.__name__,
                                               operation.__doc__)

  def scalar_compare(self, x):
    return operation(x, self.threshold)

  scalar_transform_cls = type("scalar_{}".format(operation.__name__),
                              (ScalarComparisonTransform,),
                              {"name": scalar_name,
                               "__doc__": scalar_doc,
                               "_compare": scalar_compare})

  # Define function that delegates to the two `Transforms`.
  def _comparison_fn(self, other, *args, **kwargs):
    # pylint: disable=not-callable,abstract-class-instantiated
    if isinstance(other, series.Series):
      return series_transform_cls(*args, **kwargs)([self, other])[0]
    return scalar_transform_cls(other, *args, **kwargs)([self])[0]

  # Register new member function of `Series`.
  setattr(series.Series, method_name, _comparison_fn)
