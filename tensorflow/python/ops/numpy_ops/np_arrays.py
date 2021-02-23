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
"""ndarray class."""

# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_dtypes


def convert_to_tensor(value, dtype=None, dtype_hint=None):
  """Wrapper over `tf.convert_to_tensor`.

  Args:
    value: value to convert
    dtype: (optional) the type we would like it to be converted to.
    dtype_hint: (optional) soft preference for the type we would like it to be
      converted to. `tf.convert_to_tensor` will attempt to convert value to this
      type first, but will not fail if conversion is not possible falling back
      to inferring the type instead.

  Returns:
    Value converted to tf.Tensor.
  """
  # A safer version of `tf.convert_to_tensor` to work around b/149876037.
  # TODO(wangpeng): Remove this function once the bug is fixed.
  if (dtype is None and isinstance(value, six.integer_types) and
      value >= 2**63):
    dtype = dtypes.uint64
  elif dtype is None and dtype_hint is None and isinstance(value, float):
    dtype = np_dtypes.default_float_type()
  return ops.convert_to_tensor(value, dtype=dtype, dtype_hint=dtype_hint)


ndarray = ops.Tensor
