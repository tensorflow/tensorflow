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

from types import TracebackType
from typing import Any, Optional, Type

_Status = Any

class ProfilerServer: ...
def start_server(port: int) -> ProfilerServer: ...

class ProfilerSession:
  def __init__(self, options: Optional[ProfileOptions] = ...) -> None: ...
  def stop_and_export(self, tensorboard_dir: str) -> _Status: ...

class ProfileOptions:
  include_dataset_ops: bool
  host_tracer_level: int
  python_tracer_level: int
  enable_hlo_proto: bool
  start_timestamp_ns: int
  duration_ms: int
  repository_path: str

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
