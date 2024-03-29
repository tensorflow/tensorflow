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
"""Utilities for visiting XPlanes."""

from typing import Any, Iterator, Mapping, Optional, Sequence, Tuple

from tsl.profiler.protobuf import xplane_pb2  # pylint: disable=g-direct-tensorflow-import


def find_plane_with_name(space: xplane_pb2.XSpace,
                         name: str) -> Optional['XPlaneVisitor']:
  """Returns the XPlane with the given name."""
  for plane in space.planes:
    if plane.name == name:
      return XPlaneVisitor(plane)


def _visit_stats(
    stats: Sequence[xplane_pb2.XStat],
    stat_metadata: Mapping[int, xplane_pb2.XStatMetadata]
) -> Iterator[Tuple[str, Any]]:
  """Generates the names and values of the given stats."""
  for stat in stats:
    if stat.metadata_id not in stat_metadata:
      continue
    name: str = stat_metadata[stat.metadata_id].name
    value_field: str = stat.WhichOneof('value')
    value: Any = getattr(stat, value_field)
    if value_field == 'ref_value':
      value = stat_metadata[value].name
    yield name, value


class XSpaceVisitor:
  """Wraps XSpace protobuf and provides accessors to its contents."""

  def __init__(self, space: xplane_pb2.XSpace):
    """Initializes the space visitor with the space to visit."""
    self._space = space

  @property
  def planes(self) -> Iterator['XPlaneVisitor']:
    """Planes in the space."""
    for plane in self._space.planes:
      yield XPlaneVisitor(plane)


class XPlaneVisitor:
  """Wraps XPlane protobuf and provides accessors to its contents."""

  def __init__(self, plane: xplane_pb2.XPlane):
    """Initializes the plane visitor with the plane to visit."""
    self._plane = plane

  @property
  def name(self) -> str:
    """Name of the plane."""
    return self._plane.name

  @property
  def lines(self) -> Iterator['XLineVisitor']:
    """Lines in the plane."""
    for line in self._plane.lines:
      yield XLineVisitor(line, self._plane)

  @property
  def stats(self) -> Iterator[Tuple[str, Any]]:
    """Stats in the plane."""
    return _visit_stats(self._plane.stats, self._plane.stat_metadata)


class XLineVisitor:
  """Wraps XLine protobuf and provides accessors to its contents."""

  def __init__(self, line: xplane_pb2.XLine, plane: xplane_pb2.XPlane):
    """Initializes the line visitor with a line and its plane."""
    self._line = line
    self._plane = plane

  @property
  def name(self) -> str:
    """Name of the line."""
    return self._line.name

  @property
  def events(self) -> Iterator['XEventVisitor']:
    """Events in the line."""
    for event in self._line.events:
      yield XEventVisitor(event, self._line, self._plane)


class XEventVisitor:
  """Wraps XEvent protobuf and provides accessors to its contents."""

  def __init__(self, event: xplane_pb2.XEvent, line: xplane_pb2.XLine,
               plane: xplane_pb2.XPlane):
    """Initializes the event visitor with an event, its line and its plane."""
    self._event = event
    self._line = line
    self._plane = plane

  @property
  def start_ns(self) -> float:
    """Start time of the event in nanoseconds."""
    return self._line.timestamp_ns + self._event.offset_ps / 1e3

  @property
  def duration_ns(self) -> float:
    """Duration of the event in nanoseconds."""
    return self._event.duration_ps / 1e3

  @property
  def end_ns(self) -> float:
    """End time of the event in nanoseconds."""
    return self.start_ns + self.duration_ns

  @property
  def name(self) -> str:
    """Name of the event."""
    return self._plane.event_metadata[self._event.metadata_id].name

  @property
  def stats(self) -> Iterator[Tuple[str, Any]]:
    """Stats of the event."""
    return _visit_stats(self._event.stats, self._plane.stat_metadata)
