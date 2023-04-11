# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from typing import Union
from . import XlaComputation

def xla_computation_to_mlir_module(computation: XlaComputation) -> str: ...
def mlir_module_to_xla_computation(
    mlir_module: str, use_tuple_args: bool = ...,
    return_tuple: bool = ...) -> XlaComputation: ...
def mhlo_to_stablehlo(mlir_module: Union[bytes, str]) -> str: ...
def stablehlo_to_mhlo(mlir_module: Union[bytes, str]) -> str: ...
def serialize_portable_artifact(mlir_module: str, target:str) -> bytes: ...
def deserialize_portable_artifact(mlir_module: bytes) -> str: ...
