# Copyright 2021 The OpenXLA Authors.
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

from types import TracebackType
from typing import Any, Optional, Type, Union, List

_Status = Any

class ProfilerServer: ...
def start_server(port: int) -> ProfilerServer: ...

def register_plugin_profiler(c_api: Any) -> None: ...

def get_profiled_instructions_proto(tensorboard_dir: str) -> bytes: ...
def get_fdo_profile(
    xspace: bytes, as_textproto: bool = ...
) -> Union[bytes, str]: ...

class ProfilerSession:
  def __init__(self, options: Optional[ProfileOptions] = ...) -> None: ...
  def stop(self) -> bytes: ...
  def export(self, xspace: bytes, tensorboard_dir: str) -> _Status:...

class ProfileOptions:
  include_dataset_ops: bool
  host_tracer_level: int
  python_tracer_level: int
  enable_hlo_proto: bool
  start_timestamp_ns: int
  duration_ms: int
  repository_path: str

def aggregate_profiled_instructions(profiles: List[bytes], percentile: int) -> str: ...

class TraceMe:
  def __init__(self, name: str, **kwargs: Any) -> None: ...
  def __enter__(self) -> TraceMe: ...
  def __exit__(
      self,
      exc_type: Optional[Type[BaseException]],
      exc_value: Optional[BaseException],
      exc_tb: Optional[TracebackType]) -> Optional[bool]:...
  def set_metadata(self, **kwargs): ...
  @staticmethod
  def is_enabled() -> bool: ...
