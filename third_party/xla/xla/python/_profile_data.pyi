# Copyright 2026 The OpenXLA Authors
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

"""Utilities for visiting program execution data."""

from collections.abc import Iterator

from typing_extensions import CapsuleType

class ProfileEvent:
  """Wraps XEvent protobuf and provides accessors to its contents."""

  @property
  def start_ns(self) -> float:
    """Start time of the event in nanoseconds."""

  @property
  def duration_ns(self) -> float:
    """Duration of the event in nanoseconds."""

  @property
  def end_ns(self) -> float:
    """End time of the event in nanoseconds."""

  @property
  def name(self) -> str:
    """Name of the event."""

  @property
  def stats(self) -> Iterator[tuple]:
    """Stats of the event.

    Returns
      An iterator of (name, value) tuples, note that for metadata ids that
      are not found, the returned tuple will be (None, None). The caller should
      check the tuple value before using it.
    """

class ProfileLine:
  """Wraps XLine protobuf and provides accessors to its contents."""

  @property
  def name(self) -> str:
    """Name of the line."""

  @property
  def events(self) -> Iterator[ProfileEvent]:
    """Events in the line."""

class ProfilePlane:
  """Wraps XPlane protobuf and provides accessors to its contents."""

  @property
  def name(self) -> str:
    """Name of the plane."""

  @property
  def lines(self) -> Iterator[ProfileLine]:
    """Lines in the plane."""

  @property
  def stats(self) -> Iterator[tuple]:
    """Stats in the plane.

    Returns
      An iterator of (name, value) tuples, note that for metadata ids that
      are not found, the returned tuple will be (None, None). The caller should
      check the tuple value before using it.
    """

class ProfileData:
  """Program execution data captured by jax.profiler functions."""

  def __init__(self, arg: bytes, /) -> None: ...
  @staticmethod
  def from_raw_cpp_ptr(
      capsule: CapsuleType,
  ) -> ProfileData:
    """Creates a ProfileData from a raw C++ pointer enclosed in a capsule to a XSpace proto."""

  @staticmethod
  def from_file(proto_file_path: str) -> ProfileData:
    """Creates a ProfileData from a serialized XSpace proto file."""

  @staticmethod
  def from_serialized_xspace(serialized_xspace: bytes) -> ProfileData:
    """Creates a ProfileData from a serialized XSpace proto."""

  @staticmethod
  def from_text_proto(arg: str, /) -> ProfileData:
    """Creates a ProfileData from a text proto."""

  @staticmethod
  def text_proto_to_serialized_xspace(arg: str, /) -> bytes:
    """Converts a text proto to a serialized XSpace."""

  def find_plane_with_name(self, name: str) -> ProfilePlane:
    """Finds the plane with the given name."""

  @property
  def planes(self) -> Iterator[ProfilePlane]: ...
