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
# We need to import pywrap_tensorflow prior to the toco wrapper.
# pylint: disable=invalid-import-order,g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import _pywrap_toco_api


def wrapped_toco_convert(
    model_flags_str,
    toco_flags_str,
    input_data_str,
):
  """Wraps TocoConvert with lazy loader."""
  return _pywrap_toco_api.TocoConvert(
      model_flags_str,
      toco_flags_str,
      input_data_str,
      False,  # extended_return
  )
