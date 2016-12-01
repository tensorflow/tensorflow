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
"""Transforms that wrap binary TensorFlow operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import series
from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import math_ops

# Each entry is a mapping from registered_name to operation. Each operation is
# wrapped in a transform and then registered as a member function
# `Series`.registered_name().
BINARY_TRANSFORMS = [("__eq__", math_ops.equal),
                     ("__gt__", math_ops.greater),
                     ("__ge__", math_ops.greater_equal),
                     ("__lt__", math_ops.less),
                     ("__le__", math_ops.less_equal),
                     ("__mul__", math_ops.mul),
                     ("__div__", math_ops.div),
                     ("__truediv__", math_ops.truediv),
                     ("__floordiv__", math_ops.floordiv),
                     ("__mod__", math_ops.mod),
                     ("pow", math_ops.pow)]

_DOC_FORMAT_STRING = ("A `Transform` that wraps `{0}`. "
                      "Documentation for `{0}`: \n\n {1}")


class SeriesBinaryTransform(transform.TensorFlowTransform):
  """Parent class for `Transform`s that operate on two `Series`."""

  @property
  def input_valency(self):
    return 2

  @property
  def _output_names(self):
    return "output",

  def _apply_transform(self, input_tensors, **kwargs):
    # TODO(jamieas): consider supporting sparse inputs.
    if isinstance(input_tensors[0], sparse_tensor.SparseTensor) or isinstance(
        input_tensors[1], sparse_tensor.SparseTensor):
      raise TypeError("{} does not support SparseTensors".format(
          type(self).__name__))

    # pylint: disable=not-callable
    return self.return_type(self._apply_op(input_tensors[0], input_tensors[1]))


class ScalarBinaryTransform(transform.TensorFlowTransform):
  """Parent class for `Transform`s that combine `Series` to a scalar."""

  def __init__(self, scalar):
    if isinstance(scalar, series.Series):
      raise ValueError("{} takes a Series and a scalar. "
                       "It was called with another Series.".format(
                           type(self).__name__))
    super(ScalarBinaryTransform, self).__init__()
    self._scalar = scalar

  @transform.parameter
  def scalar(self):
    return self._scalar

  @property
  def input_valency(self):
    return 1

  @property
  def _output_names(self):
    return "output",

  def _apply_transform(self, input_tensors, **kwargs):
    input_tensor = input_tensors[0]
    if isinstance(input_tensor, sparse_tensor.SparseTensor):
      result = sparse_tensor.SparseTensor(input_tensor.indices,
                                          self._apply_op(input_tensor.values),
                                          input_tensor.shape)
    else:
      result = self._apply_op(input_tensor)

    # pylint: disable=not-callable
    return self.return_type(result)


# pylint: disable=unused-argument
def register_binary_op(method_name, operation):
  """Registers `Series` member functions for binary operations.

  Args:
    method_name: the name of the method that will be created in `Series`.
    operation: underlying TensorFlow operation.
  """

  # Define series-series `Transform`.
  @property
  def series_name(self):
    return operation.__name__

  series_doc = _DOC_FORMAT_STRING.format(operation.__name__, operation.__doc__)

  def series_apply_op(self, x, y):
    return operation(x, y)

  series_transform_cls = type("scalar_{}".format(operation.__name__),
                              (SeriesBinaryTransform,),
                              {"name": series_name,
                               "__doc__": series_doc,
                               "_apply_op": series_apply_op})

  # Define series-scalar `Transform`.
  @property
  def scalar_name(self):
    return "scalar_{}".format(operation.__name__)

  scalar_doc = _DOC_FORMAT_STRING.format(operation.__name__, operation.__doc__)

  def scalar_apply_op(self, x):
    return operation(x, self.scalar)

  scalar_transform_cls = type("scalar_{}".format(operation.__name__),
                              (ScalarBinaryTransform,),
                              {"name": scalar_name,
                               "__doc__": scalar_doc,
                               "_apply_op": scalar_apply_op})

  # Define function that delegates to the two `Transforms`.
  def _fn(self, other, *args, **kwargs):
    # pylint: disable=not-callable,abstract-class-instantiated
    if isinstance(other, series.Series):
      return series_transform_cls(*args, **kwargs)([self, other])[0]
    return scalar_transform_cls(other, *args, **kwargs)([self])[0]

  # Register new member function of `Series`.
  setattr(series.Series, method_name, _fn)
