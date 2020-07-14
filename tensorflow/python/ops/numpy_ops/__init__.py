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
"""tf.experimental.numpy: Numpy API on top of TensorFlow.

This module provides a subset of numpy APIs, built on top of TensorFlow
operations. Please see documentation here:
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/ops/numpy_ops.
"""
# TODO(wangpeng): Append `np_export`ed symbols to the comments above.

# pylint: disable=g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.array_ops import newaxis
from tensorflow.python.ops.numpy_ops import np_random as random
from tensorflow.python.ops.numpy_ops import np_utils
# pylint: disable=wildcard-import
from tensorflow.python.ops.numpy_ops.np_array_ops import *
from tensorflow.python.ops.numpy_ops.np_arrays import ndarray
from tensorflow.python.ops.numpy_ops.np_dtypes import *
from tensorflow.python.ops.numpy_ops.np_math_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.ops.numpy_ops.np_utils import finfo
from tensorflow.python.ops.numpy_ops.np_utils import promote_types
from tensorflow.python.ops.numpy_ops.np_utils import result_type


# pylint: disable=redefined-builtin,undefined-variable
@np_utils.np_doc("max")
def max(a, axis=None, keepdims=None):
  return amax(a, axis=axis, keepdims=keepdims)


@np_utils.np_doc("min")
def min(a, axis=None, keepdims=None):
  return amin(a, axis=axis, keepdims=keepdims)


@np_utils.np_doc("round")
def round(a, decimals=0):
  return around(a, decimals=decimals)
# pylint: enable=redefined-builtin,undefined-variable
