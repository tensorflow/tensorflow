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

def monitor(arg0: str, arg1: int, arg2: int, arg3: bool) -> str: ...
def trace(arg0: str, arg1: str, arg2: str, arg3: bool, arg4: int, arg5: int, arg6: dict) -> None: ...
def xspace_to_tools_data(arg0: list, arg1: str, arg2: dict = ...) -> tuple: ...
def xspace_to_tools_data_from_byte_string(arg0: list, arg1: list, arg2: str, arg3: dict) -> tuple: ...
