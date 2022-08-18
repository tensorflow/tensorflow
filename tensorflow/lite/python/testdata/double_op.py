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
"""Double op is a user's defined op for testing purpose."""

from tensorflow.lite.python.testdata import double_op_wrapper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

_double_op = load_library.load_op_library(
    resource_loader.get_path_to_datafile('_double_op.so'))


def double(input_tensor):
  """Double op applies element-wise double to input data."""
  if (input_tensor.dtype != dtypes.int32 and
      input_tensor.dtype != dtypes.float32):
    raise ValueError('Double op only accept int32 or float32 values.')
  return double_op_wrapper.double(input_tensor)
