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
"""Helpers constructing Datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape


def optional_param_to_tensor(argument_name,
                             argument_value,
                             argument_default=0,
                             argument_dtype=dtypes.int64):
  if argument_value is not None:
    return ops.convert_to_tensor(
        argument_value, dtype=argument_dtype, name=argument_name)
  else:
    return constant_op.constant(
        argument_default, dtype=argument_dtype, name=argument_name)


def partial_shape_to_tensor(shape_like):
  """Returns a @{tf.Tensor} that represents the given shape.

  Args:
    shape_like: A value that can be converted to a @{tf.TensorShape} or a
      @{tf.Tensor}.

  Returns:
    A 1-D `tf.Tensor` of `tf.int64` elements representing the given shape, where
    `-1` is substituted for any unknown dimensions.
  """
  try:
    # First attempt to convert the input to a shape, and return the
    # "canonical" tensor representation, which uses `-1` in place of
    # `None`.
    shape_like = tensor_shape.as_shape(shape_like)
    return ops.convert_to_tensor(
        [dim if dim is not None else -1 for dim in shape_like.as_list()],
        dtype=dtypes.int64)
  except (TypeError, ValueError):
    # The argument was not trivially convertible to a
    # `tf.TensorShape`, so fall back on the conversion to tensor
    # machinery.
    ret = ops.convert_to_tensor(shape_like, preferred_dtype=dtypes.int64)
    if ret.shape.dims is not None and len(ret.shape.dims) != 1:
      raise ValueError("The given shape %s must be a 1-D tensor of tf.int64 "
                       "values, but the shape was %s."
                       % (shape_like, ret.shape))
    if ret.dtype != dtypes.int64:
      raise TypeError("The given shape %s must be a 1-D tensor of tf.int64 "
                      "values, but the element type was %s."
                      % (shape_like, ret.dtype.name))

    return ret
