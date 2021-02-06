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
"""Config functions for TF NumPy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.ops.numpy_ops import np_dtypes
from tensorflow.python.ops.numpy_ops import np_math_ops


def enable_numpy_behavior(prefer_float32=False):
  """Enable NumPy behavior on Tensors.

  Includes addition of methods, type promotion on operator overloads and
  support for NumPy-style slicing.

  Args:
    prefer_float32: Whether to allow type inference to use float32, or use
    float64 similar to NumPy.
  """
  ops.enable_numpy_style_type_promotion()
  ops.enable_numpy_style_slicing()
  np_math_ops.enable_numpy_methods_on_tensor()
  np_dtypes.set_prefer_float32(prefer_float32)
