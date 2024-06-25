# Copyright 2024 The OpenXLA Authors.
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

from typing import Any, Optional, Callable, Sequence, Union

from xla.python import xla_extension

class Program:  ...

class CompileOptions:  ...

def make_hlo_program(mlir_module: Union[str, bytes]) -> Program: ...

def make_plugin_program(data: Union[str, bytes]) -> Program: ...

def make_xla_compile_options(
    compile_options: xla_extension.CompileOptions,
    host_callbacks: Sequence[Any]
) -> CompileOptions: ...

def make_plugin_compile_options() -> CompileOptions: ...
