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
from typing import Any

from tensorflow.compiler.mlir.quantization.tensorflow.python import py_function_lib
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as rd

# LINT.IfChange(static_range_ptq)
def static_range_ptq(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_config_serialized: bytes,
    *,
    signature_keys: list[str],
    signature_def_map_serialized: dict[str, bytes],
    py_function_library: py_function_lib.PyFunctionLibrary,
) -> Any: ...  # Status

# LINT.ThenChange()

# LINT.IfChange(weight_only_ptq)
def weight_only_ptq(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_config_serialized: bytes,
    *,
    signature_keys: list[str],
    signature_def_map_serialized: dict[str, bytes],
    py_function_library: py_function_lib.PyFunctionLibrary,
) -> Any: ...  # Status

# LINT.ThenChange()

# LINT.IfChange(populate_default_configs)
def populate_default_configs(
    user_provided_quantization_config_serialized: bytes,
) -> bytes: ...  # QuantizationConfig

# LINT.ThenChange()

# LINT.IfChange(expand_preset_configs)
def expand_preset_configs(
    quantization_config_serialized: bytes,
) -> bytes: ...  # QuantizationConfig

# LINT.ThenChange()
