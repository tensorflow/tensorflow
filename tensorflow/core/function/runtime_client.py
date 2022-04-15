# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Low level TF runtime client."""

# TF oddity: this import loads TF-specific dynamic libraries.
from tensorflow.python import pywrap_tensorflow  # pylint:disable=g-bad-import-order,unused-import

from tensorflow.core.framework import function_pb2
from tensorflow.core.function import runtime_client_pybind

GlobalEagerContext = runtime_client_pybind.GlobalEagerContext
GlobalPythonEagerContext = runtime_client_pybind.GlobalPythonEagerContext


# TODO(mdan): Map without adapters once pybind11_protobuf available
class Runtime(runtime_client_pybind.Runtime):

  def GetFunctionProto(self, name: str) -> function_pb2.FunctionDef:
    return function_pb2.FunctionDef.FromString(
        self.GetFunctionProtoString(name))

  def CreateFunction(self, function_def: function_pb2.FunctionDef):
    self.CreateFunctionFromString(function_def.SerializeToString())
