# Copyright 2025 The OpenXLA Authors.
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
from __future__ import annotations

from types import TracebackType
from typing import Any, Iterator, List, Mapping, Optional, Sequence, Tuple, Type, Union


_Status = Any

class ProfilerServer: ...
def start_server(port: int) -> ProfilerServer: ...

def register_plugin_profiler(c_api: Any) -> None: ...

def get_profiled_instructions_proto(tensorboard_dir: str) -> bytes: ...
def get_instructins_profile(tensorboard_dir: str) -> List[Tuple[str, float]]: ...
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

class ProfileData:
  """Program execution data captured by jax.profiler functions."""

  def __init__(self, serialized_xspace: bytes):
    ...

  @classmethod
  def from_file(cls, path: str) -> 'ProfileData':
    """Creates a ProfileData from a serialized XSpace proto file."""
    ...

  @classmethod
  def from_serialized_xspace(cls, serialized_xspace: bytes) -> 'ProfileData':
    """Creates a ProfileData from a serialized XSpace proto."""
    ...

  @classmethod
  def from_raw_cpp_ptr(cls, raw_proto_ptr: object) -> 'ProfileData':
    """Creates a ProfileData from a raw C++ pointer enclosed in a capsule to a XSpace proto."""
    ...

  @classmethod
  def from_text_proto(text_proto: str) -> ProfileData:
    """Creates a ProfileData from a text proto."""
    ...

  @classmethod
  def text_proto_to_serialized_xspace(text_proto: str) -> bytes:
    """Converts a text proto to a serialized XSpace."""
    ...

  @property
  def planes(self) -> Iterator['ProfilePlane']:
    ...

  def find_plane_with_name(self, name: str) -> Optional['ProfilePlane']:
    """Finds the plane with the given name."""
    ...

class ProfilePlane:
  """Wraps XPlane protobuf and provides accessors to its contents."""

  @property
  def name(self) -> str:
    """Name of the plane."""
    ...

  @property
  def lines(self) -> Iterator['ProfileLine']:
    """Lines in the plane."""
    ...

  @property
  def stats(self) -> Iterator[tuple[str, Any] | tuple[None, None]]:
    """Stats in the plane.

    Returns
      An iterator of (name, value) tuples, note that for metadata ids that
      are not found, the returned tuple will be (None, None). The caller should
      check the tuple value before using it.
    """
    ...

class ProfileLine:
  """Wraps XLine protobuf and provides accessors to its contents."""

  @property
  def name(self) -> str:
    """Name of the line."""
    ...

  @property
  def events(self) -> Iterator['ProfileEvent']:
    """Events in the line."""
    ...

class ProfileEvent:
  """Wraps XEvent protobuf and provides accessors to its contents."""

  @property
  def start_ns(self) -> float:
    """Start time of the event in nanoseconds."""
    ...

  @property
  def duration_ns(self) -> float:
    """Duration of the event in nanoseconds."""
    ...

  @property
  def end_ns(self) -> float:
    """End time of the event in nanoseconds."""
    ...

  @property
  def name(self) -> str:
    """Name of the event."""
    ...

  @property
  def stats(self) -> Iterator[tuple[str, Any] | tuple[None, None]]:
    """Stats of the event.

    Returns
      An iterator of (name, value) tuples, note that for metadata ids that
      are not found, the returned tuple will be (None, None). The caller should
      check the tuple value before using it.
    """
    ...
