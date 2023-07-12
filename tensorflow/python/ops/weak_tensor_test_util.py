# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Utils for WeakTensor related tests."""

from tensorflow.python.framework import ops


class DtypeConversionTestEnv:
  """Test environment for different dtype conversion semantics."""

  def __init__(self, promo_mode):
    self._old_promo_mode = ops.promo_mode_enum_to_string(
        ops.get_dtype_conversion_mode()
    )
    self._new_promo_mode = promo_mode

  def __enter__(self):
    ops.set_dtype_conversion_mode(self._new_promo_mode)
    return self

  def __exit__(self, exc_type, exc_value, exc_traceback):
    ops.set_dtype_conversion_mode(self._old_promo_mode)
