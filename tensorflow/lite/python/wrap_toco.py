# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Wraps toco interface with python lazy loader."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python import pywrap_tensorflow

# TODO(b/137402359): Remove lazy loading wrapper


def wrapped_toco_convert(model_flags_str, toco_flags_str, input_data_str,
                         debug_info_str, enable_mlir_converter):
  """Wraps TocoConvert with lazy loader."""
  return pywrap_tensorflow.TocoConvert(
      model_flags_str,
      toco_flags_str,
      input_data_str,
      False,  # extended_return
      debug_info_str,
      enable_mlir_converter)


def wrapped_get_potentially_supported_ops():
  """Wraps TocoGetPotentiallySupportedOps with lazy loader."""
  return pywrap_tensorflow.TocoGetPotentiallySupportedOps()
