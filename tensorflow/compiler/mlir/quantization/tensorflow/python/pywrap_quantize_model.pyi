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
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import py_function_lib

def clear_calibrator() -> None: ...
def clear_data_from_calibrator(id: bytes) -> None: ...
def get_statistics_from_calibrator(
    id: bytes,
) -> calibration_statistics_pb2.CalibrationStatistics: ...
def quantize_qat_model(
    saved_model_path: str,
    signature_keys: list[str],
    tags: set[str],
    quantization_options_serialized: bytes,
    function_aliases: dict[str, str],
) -> bytes: ...
def quantize_ptq_dynamic_range(
    saved_model_path: str,
    signature_keys: list[str],
    tags: set[str],
    quantization_options_serialized: bytes,
    function_aliases: dict[str, str],
) -> bytes: ...
def quantize_weight_only(
    saved_model_path: str,
    quantization_options_serialized: bytes,
    function_aliases: dict[str, str],
) -> bytes: ...
def quantize_ptq_model_pre_calibration(
    saved_model_path: str,
    signature_keys: list[str],
    tags: set[str],
    quantization_options_serialized: bytes,
    function_aliases: dict[str, str],
    py_function_library: py_function_lib.PyFunctionLibrary,
) -> bytes: ...
def quantize_ptq_model_post_calibration(
    saved_model_path: str,
    signature_keys: list[str],
    tags: set[str],
    quantization_options_serialized: bytes,
    function_aliases: dict[str, str],
) -> bytes: ...
