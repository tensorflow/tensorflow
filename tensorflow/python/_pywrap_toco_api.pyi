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

def ExperimentalMlirQuantizeModel(input_contents_txt_raw: object, disable_per_channel: bool = ..., fully_quantize: bool = ..., inference_type: int = ..., input_data_type: int = ..., output_data_type: int = ..., enable_numeric_verify: bool = ..., enable_whole_model_verify: bool = ..., op_blocklist: object = ..., node_blocklist: object = ..., enable_variable_quantization: bool = ..., disable_per_channel_for_dense_layers: bool = ...) -> object: ...
def ExperimentalMlirSparsifyModel(input_contents_txt_raw: object) -> object: ...
def FlatBufferToMlir(arg0: str, arg1: bool) -> str: ...
def RegisterCustomOpdefs(custom_opdefs_txt_raw: object) -> object: ...
def RetrieveCollectedErrors() -> list: ...
def TocoConvert(model_flags_proto_txt_raw: object, toco_flags_proto_txt_raw: object, input_contents_txt_raw: object, extended_return: bool = ..., debug_info_txt_raw: object = ..., enable_mlir_converter: bool = ..., quantization_py_function_library = ...) -> object: ...
