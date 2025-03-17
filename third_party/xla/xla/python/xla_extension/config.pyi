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

from typing import Any, Generic, TypeVar

from xla.python import xla_extension

unset: object

_T = TypeVar('_T')

class Config(Generic[_T]):
  def __init__(self, value: _T, include_in_jit_key: bool = False): ...

  @property
  def value(self) -> _T: ...

  def get_local(self) -> Any: ...
  def get_global(self) -> _T: ...
  def set_local(self, value: Any) -> None: ...
  def swap_local(self, value: Any) -> Any: ...
  def set_global(self, value: _T) -> None: ...
