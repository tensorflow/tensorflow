/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/profiler/utils/xplane_utils.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/time_utils.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

// Returns the index of the first element in array for which pred is true.
// Returns -1 if no such element is found.
template <typename T, typename Pred>
int FindIf(const protobuf::RepeatedPtrField<T>& array, Pred&& pred) {
  for (int i = 0; i < array.size(); ++i) {
    if (pred(&array.Get(i))) return i;
  }
  return -1;
}

// Removes the given element from array.
template <typename T>
void Remove(protobuf::RepeatedPtrField<T>* array, const T* elem) {
  int i = FindIf(*array, [elem](const T* e) { return elem == e; });
  if (i == -1) return;
  for (; i < array->size() - 1; ++i) {
    array->SwapElements(i + 1, i);
  }
  array->RemoveLast();
}

template <typename T, typename Pred>
void RemoveIf(protobuf::RepeatedPtrField<T>* array, Pred&& pred) {
  int i = FindIf(*array, pred);
  if (i == -1) return;
  for (int j = i + 1; j < array->size(); ++j) {
    if (!pred(&array->Get(j))) array->SwapElements(j, i++);
  }
  array->DeleteSubrange(i, array->size() - i);
}

}  // namespace

const XPlane* FindPlaneWithName(const XSpace& space, absl::string_view name) {
  int i = FindIf(space.planes(),
                 [name](const XPlane* plane) { return plane->name() == name; });
  return (i != -1) ? &space.planes(i) : nullptr;
}

XPlane* FindMutablePlaneWithName(XSpace* space, absl::string_view name) {
  int i = FindIf(space->planes(),
                 [name](const XPlane* plane) { return plane->name() == name; });
  return (i != -1) ? space->mutable_planes(i) : nullptr;
}

XPlane* FindOrAddMutablePlaneWithName(XSpace* space, absl::string_view name) {
  XPlane* plane = FindMutablePlaneWithName(space, name);
  if (plane == nullptr) {
    plane = space->add_planes();
    plane->set_name(name.data(), name.size());
  }
  return plane;
}

std::vector<const XPlane*> FindPlanesWithPrefix(const XSpace& space,
                                                absl::string_view prefix) {
  std::vector<const XPlane*> result;
  for (const XPlane& plane : space.planes()) {
    if (absl::StartsWith(plane.name(), prefix)) result.push_back(&plane);
  }
  return result;
}

std::vector<XPlane*> FindMutablePlanesWithPrefix(XSpace* space,
                                                 absl::string_view prefix) {
  std::vector<XPlane*> result;
  for (XPlane& plane : *space->mutable_planes()) {
    if (absl::StartsWith(plane.name(), prefix)) result.push_back(&plane);
  }
  return result;
}

const XLine* FindLineWithId(const XPlane& plane, int64 id) {
  int i = FindIf(plane.lines(),
                 [id](const XLine* line) { return line->id() == id; });
  return (i != -1) ? &plane.lines(i) : nullptr;
}

XStat* FindOrAddMutableStat(const XStatMetadata& stat_metadata, XEvent* event) {
  for (auto& stat : *event->mutable_stats()) {
    if (stat.metadata_id() == stat_metadata.id()) {
      return &stat;
    }
  }
  XStat* stat = event->add_stats();
  stat->set_metadata_id(stat_metadata.id());
  return stat;
}

void RemovePlane(XSpace* space, const XPlane* plane) {
  DCHECK(plane != nullptr);
  Remove(space->mutable_planes(), plane);
}

void RemoveLine(XPlane* plane, const XLine* line) {
  DCHECK(line != nullptr);
  Remove(plane->mutable_lines(), line);
}

void RemoveEvents(XLine* line,
                  const absl::flat_hash_set<const XEvent*>& events) {
  RemoveIf(line->mutable_events(),
           [&](const XEvent* event) { return events.contains(event); });
}

void RemoveEmptyPlanes(XSpace* space) {
  RemoveIf(space->mutable_planes(),
           [&](const XPlane* plane) { return plane->lines().empty(); });
}

void RemoveEmptyLines(XPlane* plane) {
  RemoveIf(plane->mutable_lines(),
           [&](const XLine* line) { return line->events().empty(); });
}

bool XEventsComparator::operator()(const XEvent* a, const XEvent* b) const {
  return XEventTimespan(*a) < XEventTimespan(*b);
}

void SortXPlane(XPlane* plane) {
  for (XLine& line : *plane->mutable_lines()) {
    auto& events = *line.mutable_events();
    std::sort(events.pointer_begin(), events.pointer_end(),
              XEventsComparator());
  }
}

void SortXSpace(XSpace* space) {
  for (XPlane& plane : *space->mutable_planes()) SortXPlane(&plane);
}

// Normalize the line's timestamp in this XPlane.
// NOTE: This can be called multiple times on the same plane. Only the first
// call will do the normalization, subsequent calls will do nothing.
// The assumption is that both line's timestamp_ns and start_time_ns are
// nano-seconds from epoch time, the different of these values is much
// smaller than these value.
void NormalizeTimestamps(XPlane* plane, uint64 start_time_ns) {
  for (XLine& line : *plane->mutable_lines()) {
    if (line.timestamp_ns() >= static_cast<int64>(start_time_ns)) {
      line.set_timestamp_ns(line.timestamp_ns() - start_time_ns);
    }
  }
}

void NormalizeTimestamps(XSpace* space, uint64 start_time_ns) {
  for (XPlane& plane : *space->mutable_planes()) {
    NormalizeTimestamps(&plane, start_time_ns);
  }
}

void MergePlanes(const XPlane& src_plane, XPlane* dst_plane) {
  RemoveEmptyLines(dst_plane);
  XPlaneVisitor src(&src_plane);
  XPlaneBuilder dst(dst_plane);
  src.ForEachStat([&](const tensorflow::profiler::XStatVisitor& stat) {
    XStatMetadata* stat_metadata = dst.GetOrCreateStatMetadata(stat.Name());
    // Use SetOrAddStat to avoid duplicating stats in dst_plane.
    dst.SetOrAddStat(*stat_metadata, stat.RawStat(), src_plane);
  });
  src.ForEachLine([&](const tensorflow::profiler::XLineVisitor& line) {
    XLineBuilder dst_line = dst.GetOrCreateLine(line.Id());
    int64 time_offset_ps = 0LL;
    if (dst_line.NumEvents() == 0) {
      // Since we RemoveEmptyLines above, this could only mean that current
      // line only exist in src plane.
      dst_line.SetTimestampNs(line.TimestampNs());
      dst_line.SetName(line.Name());
      dst_line.SetDisplayNameIfEmpty(line.DisplayName());
    } else {
      if (line.TimestampNs() <= dst_line.TimestampNs()) {
        dst_line.SetTimestampNsAndAdjustEventOffsets(line.TimestampNs());
      } else {
        time_offset_ps =
            NanosToPicos(line.TimestampNs() - dst_line.TimestampNs());
      }
      dst_line.SetNameIfEmpty(line.Name());
      // Don't override dst_line's display name because if both lines have name,
      // but no display name, line's name will became display name of dst_line.
    }

    line.ForEachEvent([&](const tensorflow::profiler::XEventVisitor& event) {
      const XEventMetadata* src_event_metadata = event.metadata();
      XEventMetadata* dst_event_metadata =
          dst.GetOrCreateEventMetadata(event.Name());
      if (dst_event_metadata->display_name().empty() &&
          !src_event_metadata->display_name().empty()) {
        dst_event_metadata->set_display_name(
            src_event_metadata->display_name());
      }
      if (dst_event_metadata->metadata().empty() &&
          !src_event_metadata->metadata().empty()) {
        dst_event_metadata->set_metadata(src_event_metadata->metadata());
      }
      XEventBuilder dst_event = dst_line.AddEvent(*dst_event_metadata);
      dst_event.SetOffsetPs(event.OffsetPs() + time_offset_ps);
      dst_event.SetDurationPs(event.DurationPs());
      if (event.NumOccurrences()) {
        dst_event.SetNumOccurrences(event.NumOccurrences());
      }
      event.ForEachStat([&](const tensorflow::profiler::XStatVisitor& stat) {
        // Here we can call AddStat instead of SetOrAddStat because dst_event
        // was just added.
        dst_event.AddStat(*dst.GetOrCreateStatMetadata(stat.Name()),
                          stat.RawStat(), src_plane);
      });
    });
  });
}

uint64 GetStartTimestampNs(const XPlane& plane) {
  int64 plane_timestamp = 0;
  for (const auto& line : plane.lines()) {
    plane_timestamp = std::min<int64>(plane_timestamp, line.timestamp_ns());
  }
  return plane_timestamp;
}

bool IsEmpty(const XSpace& space) {
  for (const auto& plane : space.planes()) {
    for (const auto& line : plane.lines()) {
      if (!line.events().empty()) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace profiler
}  // namespace tensorflow
