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
#include "xla/tsl/profiler/utils/xplane_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/math_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xla/tsl/util/stats_calculator.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tsl {
namespace profiler {
namespace {

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

// Returns the index of the first element in array for which pred is true.
// Returns -1 if no such element is found.
template <typename T, typename Pred>
int Find(const protobuf::RepeatedPtrField<T>& array, const Pred& pred) {
  std::vector<int> indices = FindAll(array, pred);
  if (indices.size() > 1) {
    LOG(WARNING) << "Found multiple " << T().GetTypeName()
                 << " when only one was expected.";
  }
  return indices.empty() ? -1 : indices.front();
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

// Copy XEventMetadata from source to destination. Also copies the associated
// XStats.
void CopyEventMetadata(const XEventMetadata& src_event_metadata,
                       const XPlaneVisitor& src_plane,
                       XEventMetadata& dst_event_metadata,
                       XPlaneBuilder& dst_plane) {
  if (dst_event_metadata.display_name().empty() &&
      !src_event_metadata.display_name().empty()) {
    dst_event_metadata.set_display_name(src_event_metadata.display_name());
  }
  if (dst_event_metadata.name().empty() && !src_event_metadata.name().empty()) {
    dst_event_metadata.set_name(src_event_metadata.name());
  }
  if (dst_event_metadata.metadata().empty() &&
      !src_event_metadata.metadata().empty()) {
    dst_event_metadata.set_metadata(src_event_metadata.metadata());
  }

  if (dst_event_metadata.stats().empty() &&
      !src_event_metadata.stats().empty()) {
    XEventMetadataVisitor src_event_metadata_visitor(&src_plane,
                                                     &src_event_metadata);
    src_event_metadata_visitor.ForEachStat([&](const XStatVisitor& src_stat) {
      XStatMetadata& metadata =
          *dst_plane.GetOrCreateStatMetadata(src_stat.Name());
      XStat& dst_stat = *dst_event_metadata.add_stats();
      dst_stat = src_stat.RawStat();
      if (src_stat.ValueCase() == XStat::kRefValue) {
        XStatMetadata& value_metadata =
            *dst_plane.GetOrCreateStatMetadata(src_stat.StrOrRefValue());
        dst_stat.set_ref_value(value_metadata.id());
      }
      dst_stat.set_metadata_id(metadata.id());
    });
  }
  DCHECK_EQ(src_event_metadata.stats_size(), dst_event_metadata.stats_size());
}

// Copies src_event from source line to the destination line in the destination
// plane as is, along with event metadata and stats.
void CopyEvent(const XEventVisitor& src_event, const XPlaneVisitor& src,
               const XPlane& src_plane, int64_t time_offset_ps,
               XPlaneBuilder& dst_plane, XLineBuilder& dst_line) {
  XEventMetadata* dst_event_metadata =
      dst_plane.GetOrCreateEventMetadata(src_event.Name());
  CopyEventMetadata(*src_event.metadata(), src, *dst_event_metadata, dst_plane);
  XEventBuilder dst_event = dst_line.AddEvent(*dst_event_metadata);
  if (src_event.IsAggregatedEvent()) {
    dst_event.SetNumOccurrences(src_event.NumOccurrences());
  } else {
    dst_event.SetOffsetPs(src_event.OffsetPs() + time_offset_ps);
  }
  dst_event.SetDurationPs(src_event.DurationPs());

  src_event.ForEachStat([&](const XStatVisitor& stat) {
    // Here we can call AddStat instead of SetOrAddStat because dst_event
    // was just added.
    dst_event.AddStat(*dst_plane.GetOrCreateStatMetadata(stat.Name()),
                      stat.RawStat(), src_plane);
  });
}

bool IsOpLineName(absl::string_view line_name) {
  return line_name == kXlaOpLineName || line_name == kTensorFlowOpLineName;
}

Timespan GetEventTimespan(const XEventVisitor& event) {
  const std::optional<XStatVisitor> device_offset_ps =
      event.GetStat(StatType::kDeviceOffsetPs);
  const std::optional<XStatVisitor> device_duration_ps =
      event.GetStat(StatType::kDeviceDurationPs);
  if (device_offset_ps.has_value() && device_duration_ps.has_value()) {
    return Timespan(device_offset_ps->IntOrUintValue(),
                    device_duration_ps->IntOrUintValue());
  }

  return event.GetTimespan();
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
  return FindPlanes(space, [&](const XPlane& plane) {
    return absl::StartsWith(plane.name(), prefix);
  });
}

std::vector<XPlane*> FindMutablePlanesWithPrefix(XSpace* space,
                                                 absl::string_view prefix) {
  return FindMutablePlanes(space, [&](XPlane& plane) {
    return absl::StartsWith(plane.name(), prefix);
  });
}

const XLine* FindLineWithId(const XPlane& plane, int64_t id) {
  int i =
      Find(plane.lines(), [id](const XLine* line) { return line->id() == id; });
  return (i != -1) ? &plane.lines(i) : nullptr;
}
std::vector<const XLine*> FindLinesWithId(const XPlane& plane, int64_t id) {
  std::vector<int> indices = FindAll(
      plane.lines(), [id](const XLine* line) { return line->id() == id; });
  std::vector<const XLine*> lines;
  lines.reserve(indices.size());
  for (int index : indices) {
    lines.push_back(&plane.lines(index));
  }
  return lines;
}

const XLine* FindLineWithName(const XPlane& plane, absl::string_view name) {
  int i = Find(plane.lines(),
               [name](const XLine* line) { return line->name() == name; });
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
  src.ForEachStat([&](const XStatVisitor& stat) {
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
      CopyEvent(event, src, src_plane, time_offset_ps, dst, dst_line);
    });
  });
}

void MergePlanes(const std::vector<const XPlane*>& src_planes,
                 XPlane* dst_plane) {
  for (const XPlane* src_plane : src_planes) {
    MergePlanes(*src_plane, dst_plane);
  }
}

int64_t GetStartTimestampNs(const XPlane& plane) {
  if (plane.lines().empty()) return 0LL;
  int64_t plane_timestamp = std::numeric_limits<int64_t>::max();
  for (const auto& line : plane.lines()) {
    plane_timestamp = std::min(plane_timestamp, line.timestamp_ns());
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

bool IsXSpaceGrouped(const XSpace& space) {
  for (const auto& plane : space.planes()) {
    // If any plane has been grouped, consider space as grouped.
    // CreateTfXPlaneVisitor is necessary because we need check "group_id" stat
    // by its type StatType::kGroupId.
    XPlaneVisitor xplane = tsl::profiler::CreateTfXPlaneVisitor(&plane);
    const XStatMetadata* group_id_stat =
        xplane.GetStatMetadataByType(StatType::kGroupId);
    if (group_id_stat) return true;
  }
  return false;
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
      std::optional<uint64_t> correlation_id;
      std::optional<uint64_t> producer_type;
      std::optional<uint64_t> consumer_type;
      std::optional<uint64_t> producer_id;
      std::optional<uint64_t> consumer_id;
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

uint64_t GetDevicePlaneFingerprint(const XPlane& plane) {
  const XLine* xla_module_line = FindLineWithName(plane, kXlaModuleLineName);
  if (!xla_module_line) return 0ULL;

  XPlaneVisitor xplane(&plane);
  XLineVisitor xline(&xplane, xla_module_line);
  std::set<uint64_t> ordered_module_fps;
  xline.ForEachEvent([&](const XEventVisitor& xevent) {
    ordered_module_fps.insert(Fingerprint64(xevent.Name()));
  });
  if (ordered_module_fps.empty()) return 0ULL;
  uint64_t output = 0ULL;
  for (const auto& fp : ordered_module_fps) {
    output = FingerprintCat64(output, fp);
  }
  return output;
}

std::optional<XEventVisitor> XEventContextTracker::GetContainingEvent(
    const Timespan& event) {
  if (!line_) return std::nullopt;
  if (current_index_ != -1) {
    XEventVisitor current_event(plane_, line_, &line_->events(current_index_));
    if (current_event.GetTimespan().Includes(event)) {
      return current_event;
    }
  }
  for (int i = current_index_ + 1; i < line_->events_size(); ++i) {
    XEventVisitor current_event(plane_, line_, &line_->events(i));
    if (current_event.TimestampPs() > event.end_ps()) break;
    if (current_event.EndTimestampPs() < event.begin_ps()) continue;
    current_index_ = i;
    if (current_event.GetTimespan().Includes(event)) {
      return current_event;
    }
    break;  // overlapping
  }
  return std::nullopt;
}

std::optional<XEventVisitor> XEventContextTracker::GetOverlappingEvent(
    const Timespan& event) {
  if (!line_) return std::nullopt;
  if (current_index_ != -1) {
    XEventVisitor current_event(plane_, line_, &line_->events(current_index_));
    if (current_event.GetTimespan().Overlaps(event)) {
      return current_event;
    }
  }
  for (int i = current_index_ + 1; i < line_->events_size(); ++i) {
    XEventVisitor current_event(plane_, line_, &line_->events(i));
    if (current_event.TimestampPs() > event.end_ps()) break;
    if (current_event.EndTimestampPs() < event.begin_ps()) continue;
    current_index_ = i;
    if (current_event.GetTimespan().Overlaps(event)) {
      return current_event;
    }
    break;  // overlapping
  }
  return std::nullopt;
}

void AggregateXPlane(const XPlane& full_trace, XPlane& aggregated_trace) {
  struct EventStat {
    tsl::Stat<int64_t> stat;
    int64_t children_duration;
  };
  using StatByEvent = absl::flat_hash_map<int64_t /*event_id*/, EventStat>;
  using StatByGroup = absl::flat_hash_map<int64_t /*group_id*/, StatByEvent>;

  absl::flat_hash_map<int64_t /*line_id*/, StatByGroup> stats;

  const XPlaneVisitor& plane = CreateTfXPlaneVisitor(&full_trace);
  XPlaneBuilder aggregated_plane(&aggregated_trace);
  aggregated_plane.SetName(plane.Name());

  uint64_t first_op_start_ps = kint64max;
  uint64_t last_op_end_ps = 0;

  plane.ForEachLine([&](const XLineVisitor& line) {
    if (line.Name() == kStepLineName ||
        line.Name() == kSparseCoreStepLineName) {
      XLineBuilder aggregated_line =
          aggregated_plane.GetOrCreateLine(line.Id());
      aggregated_line.SetName(kStepLineName);
      line.ForEachEvent([&](const XEventVisitor& event) {
        CopyEvent(event, plane, full_trace, 0LL, aggregated_plane,
                  aggregated_line);
      });
    }
    if (!IsOpLineName(line.Name())) return;
    XLineBuilder aggregated_line = aggregated_plane.GetOrCreateLine(line.Id());
    aggregated_line.SetName(line.Name());
    std::vector<XEventVisitor> event_stack;
    line.ForEachEvent([&](XEventVisitor event) {
      Timespan timespan = GetEventTimespan(event);
      first_op_start_ps = first_op_start_ps <= event.TimestampPs()
                              ? first_op_start_ps
                              : timespan.begin_ps();
      last_op_end_ps = last_op_end_ps >= event.EndTimestampPs()
                           ? last_op_end_ps
                           : timespan.end_ps();
      const auto& group_stat = event.GetStat(StatType::kGroupId);
      int64_t group_id =
          group_stat.has_value() ? group_stat->IntOrUintValue() : kint64max;

      StatByEvent& line_stats = stats[line.Id()][group_id];
      line_stats[event.Id()].stat.UpdateStat(timespan.duration_ps());
      DCHECK(event_stack.empty() || !(event < event_stack.back()));
      while (!event_stack.empty() &&
             !GetEventTimespan(event_stack.back()).Includes(timespan)) {
        event_stack.pop_back();
      }
      if (!event_stack.empty()) {
        line_stats[event_stack.back().Id()].children_duration +=
            timespan.duration_ps();
      }
      event_stack.push_back(std::move(event));
    });
  });

  uint64_t total_time_ps =
      (last_op_end_ps && last_op_end_ps > first_op_start_ps)
          ? last_op_end_ps - first_op_start_ps
          : 0;

  aggregated_plane.AddStatValue(
      *aggregated_plane.GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kTotalProfileDurationPs)),
      total_time_ps);

  XStatMetadata* kMinDurationPs = aggregated_plane.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kMinDurationPs));
  XStatMetadata* kSelfDurationPs = aggregated_plane.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kSelfDurationPs));
  XStatMetadata* kGroupId = aggregated_plane.GetOrCreateStatMetadata(
      GetStatTypeStr(StatType::kGroupId));

  for (const auto& [line_id, stats_by_group] : stats) {
    XLineBuilder aggregated_line = aggregated_plane.GetOrCreateLine(line_id);
    for (const auto& [group_id, stat_by_event] : stats_by_group) {
      for (const auto& [event_id, event_stat] : stat_by_event) {
        const auto& src_event_metadata = *plane.GetEventMetadata(event_id);
        XEventMetadata& event_metadata =
            *aggregated_plane.GetOrCreateEventMetadata(
                src_event_metadata.name());
        CopyEventMetadata(src_event_metadata, plane, event_metadata,
                          aggregated_plane);
        XEventBuilder aggregated_event =
            aggregated_line.AddEvent(event_metadata);
        aggregated_event.SetNumOccurrences(event_stat.stat.count());
        aggregated_event.SetDurationPs(event_stat.stat.sum());
        if (group_id != kint64max) {
          aggregated_event.AddStatValue(*kGroupId, group_id);
        }
        if (event_stat.stat.count() > 1) {
          aggregated_event.AddStatValue(*kMinDurationPs, event_stat.stat.min());
        }
        if (event_stat.children_duration != 0) {
          aggregated_event.AddStatValue(
              *kSelfDurationPs,
              event_stat.stat.sum() - event_stat.children_duration);
        }
      }
    }
  }
}

bool IsCustomPlane(const XPlane& plane) {
  // NOTE: remove me after all legacy traces are gone (i.e. 2022/08/04).
  constexpr absl::string_view kLegacyCustomPlanePrefix = "/custom:";
  return absl::StartsWith(plane.name(), kCustomPlanePrefix) ||
         absl::StartsWith(plane.name(), kLegacyCustomPlanePrefix);
}

bool IsHostPlane(const XPlane& plane) {
  return plane.name() == kHostThreadsPlaneName ||
         plane.name() == kHostCpusPlaneName ||
         plane.name() == kTFStreamzPlaneName ||
         plane.name() == kMetadataPlaneName ||
         plane.name() == kSyscallsPlaneName ||
         plane.name() == kPythonTracerPlaneName ||
         plane.name() == kCuptiDriverApiPlaneName ||
         plane.name() == kScopeRangeIdTreePlaneName;
}

bool IsDevicePlane(const XPlane& plane) {
  // Device and host planes should be mutually exclusive.
  if (IsHostPlane(plane)) return false;
  return absl::StartsWith(plane.name(), "/device") ||
         absl::StartsWith(plane.name(), kTpuNonCorePlaneNamePrefix) ||
         IsCustomPlane(plane);
}

}  // namespace profiler
}  // namespace tsl
