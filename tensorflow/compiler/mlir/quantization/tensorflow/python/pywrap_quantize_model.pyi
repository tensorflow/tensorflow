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

from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import py_function_lib
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as rd

# LINT.IfChange(quantize_qat_model)
def quantize_qat_model(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_options_serialized: bytes,
    *,
    signature_keys: list[str],
    signature_def_map_serialized: dict[str, bytes],
    function_aliases: dict[str, str],
    py_function_library: py_function_lib.PyFunctionLibrary,
) -> Any: ...  # Status

# LINT.ThenChange()

# LINT.IfChange(quantize_ptq_dynamic_range)
def quantize_ptq_dynamic_range(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_options_serialized: bytes,
    *,
    signature_keys: list[str],
    signature_def_map_serialized: dict[str, bytes],
    function_aliases: dict[str, str],
    py_function_library: py_function_lib.PyFunctionLibrary,
) -> Any: ...  # Status

# LINT.ThenChange()

# LINT.IfChange(quantize_weight_only)
def quantize_weight_only(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_options_serialized: bytes,
    *,
    signature_def_map_serialized: dict[str, bytes],
    function_aliases: dict[str, str],
    py_function_library: py_function_lib.PyFunctionLibrary,
) -> Any: ...  # Status

# LINT.ThenChange()

# LINT.IfChange(quantize_ptq_static_range)
def quantize_ptq_static_range(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_options_serialized: bytes,
    *,
    signature_keys: list[str],
    signature_def_map_serialized: dict[str, bytes],
    function_aliases: dict[str, str],
    py_function_library: py_function_lib.PyFunctionLibrary,
    representative_dataset: rd.RepresentativeDatasetOrMapping,
) -> Any: ...  # Status

# LINT.ThenChange()
