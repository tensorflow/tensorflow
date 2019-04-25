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

from tensorflow.python.util.lazy_loader import LazyLoader


# TODO(b/131123224): Lazy load since some of the performance benchmark skylark
# rules and monolithic build break dependencies.
_toco_python = LazyLoader(
    "tensorflow_wrap_toco", globals(),
    "tensorflow.lite.toco.python."
    "tensorflow_wrap_toco")
del LazyLoader


def wrapped_toco_convert(model_flags_str, toco_flags_str, input_data_str):
  """Wraps TocoConvert with lazy loader."""
  return _toco_python.TocoConvert(model_flags_str, toco_flags_str,
                                  input_data_str)


def wrapped_get_potentially_supported_ops():
  """Wraps TocoGetPotentiallySupportedOps with lazy loader."""
  return _toco_python.TocoGetPotentiallySupportedOps()
