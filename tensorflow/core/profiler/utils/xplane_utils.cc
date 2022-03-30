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
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/context_types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

// Returns the index of the first element in array for which pred is true.
// Returns -1 if no such element is found.
template <typename T, typename Pred>
int Find(const protobuf::RepeatedPtrField<T>& array, const Pred& pred) {
  for (int i = 0; i < array.size(); ++i) {
    if (pred(&array.Get(i))) return i;
  }
  return -1;
}

// Returns the indices of all elements in array for which pred is true.
template <typename T, typename Pred>
std::vector<int> FindAll(const protobuf::RepeatedPtrField<T>& array,
                         const Pred& pred) {
  std::vector<int> indices;
  for (int i = 0; i < array.size(); ++i) {
    if (pred(&array.Get(i))) indices.push_back(i);
  }
  return indices;
}

template <typename T>
void RemoveAt(protobuf::RepeatedPtrField<T>* array,
              const std::vector<int>& indices) {
  if (indices.empty()) return;
  if (array->size() == indices.size()) {
    // Assumes that 'indices' consists of [0 ... N-1].
    array->Clear();
    return;
  }
  auto remove_iter = indices.begin();
  int i = *(remove_iter++);
  for (int j = i + 1; j < array->size(); ++j) {
    if (remove_iter != indices.end() && *remove_iter == j) {
      ++remove_iter;
    } else {
      array->SwapElements(j, i++);
    }
  }
  array->DeleteSubrange(i, array->size() - i);
}

// Removes the given element from array.
template <typename T>
void Remove(protobuf::RepeatedPtrField<T>* array, const T* elem) {
  int i = Find(*array, [elem](const T* e) { return elem == e; });
  RemoveAt(array, {i});
}

template <typename T, typename Pred>
void RemoveIf(protobuf::RepeatedPtrField<T>* array, Pred&& pred) {
  std::vector<int> indices = FindAll(*array, pred);
  RemoveAt(array, indices);
}

}  // namespace

const XPlane* FindPlaneWithName(const XSpace& space, absl::string_view name) {
  int i = Find(space.planes(),
               [name](const XPlane* plane) { return plane->name() == name; });
  return (i != -1) ? &space.planes(i) : nullptr;
}

std::vector<const XPlane*> FindPlanesWithNames(
    const XSpace& space, const std::vector<absl::string_view>& names) {
  absl::flat_hash_set<absl::string_view> names_set(names.begin(), names.end());
  std::vector<int> indices =
      FindAll(space.planes(), [&names_set](const XPlane* plane) {
        return names_set.contains(plane->name());
      });
  std::vector<const XPlane*> planes;
  planes.reserve(indices.size());
  for (int i : indices) {
    planes.push_back(&space.planes(i));
  }
  return planes;
}

XPlane* FindMutablePlaneWithName(XSpace* space, absl::string_view name) {
  int i = Find(space->planes(),
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

const XLine* FindLineWithId(const XPlane& plane, int64_t id) {
  int i =
      Find(plane.lines(), [id](const XLine* line) { return line->id() == id; });
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

void RemovePlanes(XSpace* space, const std::vector<const XPlane*>& planes) {
  absl::flat_hash_set<const XPlane*> planes_set(planes.begin(), planes.end());
  RemoveIf(space->mutable_planes(), [&planes_set](const XPlane* plane) {
    return planes_set.contains(plane);
  });
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
    if (line.timestamp_ns() >= static_cast<int64_t>(start_time_ns)) {
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
  src.ForEachLine([&](const XLineVisitor& line) {
    XLineBuilder dst_line = dst.GetOrCreateLine(line.Id());
    int64_t time_offset_ps = 0LL;
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
            NanoToPico(line.TimestampNs() - dst_line.TimestampNs());
      }
      dst_line.SetNameIfEmpty(line.Name());
      // Don't override dst_line's display name because if both lines have name,
      // but no display name, line's name will became display name of dst_line.
    }

    line.ForEachEvent([&](const XEventVisitor& event) {
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
      event.ForEachStat([&](const XStatVisitor& stat) {
        // Here we can call AddStat instead of SetOrAddStat because dst_event
        // was just added.
        dst_event.AddStat(*dst.GetOrCreateStatMetadata(stat.Name()),
                          stat.RawStat(), src_plane);
      });
    });
  });
}

void MergePlanes(const std::vector<const XPlane*>& src_planes,
                 XPlane* dst_plane) {
  for (const XPlane* src_plane : src_planes) {
    MergePlanes(*src_plane, dst_plane);
  }
}

uint64 GetStartTimestampNs(const XPlane& plane) {
  int64_t plane_timestamp = 0;
  for (const auto& line : plane.lines()) {
    plane_timestamp = std::min<int64_t>(plane_timestamp, line.timestamp_ns());
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

void AddFlowsToXplane(int32_t host_id, bool is_host_plane, bool connect_traceme,
                      XPlane* xplane) {
  if (!xplane) return;
  XPlaneBuilder plane(xplane);
  XStatMetadata* correlation_id_stats_metadata =
      plane.GetStatMetadata(GetStatTypeStr(StatType::kCorrelationId));
  XStatMetadata* producer_type_stats_metadata =
      plane.GetStatMetadata(GetStatTypeStr(StatType::kProducerType));
  XStatMetadata* consumer_type_stats_metadata =
      plane.GetStatMetadata(GetStatTypeStr(StatType::kConsumerType));
  XStatMetadata* producer_id_stats_metadata =
      plane.GetStatMetadata(GetStatTypeStr(StatType::kProducerId));
  XStatMetadata* consumer_id_stats_metadata =
      plane.GetStatMetadata(GetStatTypeStr(StatType::kConsumerId));
  XStatMetadata* flow_stats_metadata =
      plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kFlow));
  XFlow::FlowDirection direction = is_host_plane
                                       ? XFlow::FlowDirection::kFlowOut
                                       : XFlow::FlowDirection::kFlowIn;

  plane.ForEachLine([&](XLineBuilder line) {
    line.ForEachEvent([&](XEventBuilder event) {
      absl::optional<uint64_t> correlation_id;
      absl::optional<uint64_t> producer_type;
      absl::optional<uint64_t> consumer_type;
      absl::optional<uint64_t> producer_id;
      absl::optional<uint64_t> consumer_id;
      event.ForEachStat([&](XStat* stat) {
        if (correlation_id_stats_metadata &&
            stat->metadata_id() == correlation_id_stats_metadata->id()) {
          correlation_id = stat->uint64_value();
        } else if (connect_traceme) {
          if (producer_type_stats_metadata &&
              stat->metadata_id() == producer_type_stats_metadata->id()) {
            producer_type = XStatsBuilder<XPlane>::IntOrUintValue(*stat);
          } else if (consumer_type_stats_metadata &&
                     stat->metadata_id() ==
                         consumer_type_stats_metadata->id()) {
            consumer_type = XStatsBuilder<XPlane>::IntOrUintValue(*stat);
          } else if (producer_id_stats_metadata &&
                     stat->metadata_id() == producer_id_stats_metadata->id()) {
            producer_id = XStatsBuilder<XPlane>::IntOrUintValue(*stat);
          } else if (consumer_id_stats_metadata &&
                     stat->metadata_id() == consumer_id_stats_metadata->id()) {
            consumer_id = XStatsBuilder<XPlane>::IntOrUintValue(*stat);
          }
        }
      });
      if (correlation_id) {
        XFlow flow(XFlow::GetFlowId(host_id, *correlation_id), direction,
                   ContextType::kGpuLaunch);
        event.AddStatValue(*flow_stats_metadata, flow.ToStatValue());
      }
      if (connect_traceme) {
        if (producer_type && producer_id) {
          auto context_type = GetSafeContextType(*producer_type);
          XFlow flow(XFlow::GetFlowId(host_id, *producer_id, context_type),
                     XFlow::FlowDirection::kFlowOut, context_type);
          event.AddStatValue(*flow_stats_metadata, flow.ToStatValue());
        }
        if (consumer_type && consumer_id) {
          auto context_type = GetSafeContextType(*consumer_type);
          XFlow flow(XFlow::GetFlowId(host_id, *consumer_id, context_type),
                     XFlow::FlowDirection::kFlowIn, context_type);
          event.AddStatValue(*flow_stats_metadata, flow.ToStatValue());
        }
      }
    });
  });
}

}  // namespace profiler
}  // namespace tensorflow
