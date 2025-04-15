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

from typing import overload

def ExperimentalConvertSavedModelToMlir(arg0: str, arg1: str, arg2: bool) -> str: ...
def ExperimentalConvertSavedModelV1ToMlir(arg0: str, arg1: str, arg2: str, arg3: bool, arg4: bool, arg5: bool, arg6: bool) -> str: ...
def ExperimentalConvertSavedModelV1ToMlirLite(arg0: str, arg1: str, arg2: str, arg3: bool, arg4: bool) -> str: ...
def ExperimentalRunPassPipeline(arg0: str, arg1: str, arg2: bool) -> str: ...
def ExperimentalWriteBytecode(arg0: str, arg1: str) -> None: ...
def ImportFunction(arg0: object, arg1: str, arg2: str, arg3: bool) -> str: ...
@overload
def ImportGraphDef(arg0: str, arg1: str, arg2: bool) -> str: ...
@overload
def ImportGraphDef(arg0: str, arg1: str, arg2: bool, arg3: str, arg4: str, arg5: str, arg6: str) -> str: ...
