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
#include "tensorflow/core/platform/env_time.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

// Creates a Timespan from an XEvent.
// WARNING: This should only be used when comparing events from the same XLine.
Timespan XEventTimespan(const XEvent& event) {
  return Timespan(event.offset_ps(), event.duration_ps());
}

// Creates a Timespan from a non-empty XLine.
Timespan XLineTimespan(const XLine& line) {
  uint64 begin_ps = kuint64max, end_ps = 0;
  for (const XEvent& event : line.events()) {
    // Don't use XEventTimespan. We need the absolute event start time as lines
    // might have different timestamps.
    Timespan span(line.timestamp_ns() * 1000 + event.offset_ps(),
                  event.duration_ps());
    begin_ps = std::min(span.begin_ps(), begin_ps);
    end_ps = std::max(span.end_ps(), end_ps);
  }
  return Timespan::FromEndPoints(begin_ps, end_ps);
}

// Functor that compares XEvents of the same XLine for sorting by timespan.
struct XEventsComparator {
  bool operator()(const XEvent* a, const XEvent* b) const {
    return XEventTimespan(*a) < XEventTimespan(*b);
  }
};

// Functor that compares XLines of the same XPlane for sorting by timespan.
class XLinesComparator {
 public:
  bool operator()(const XLine* a, const XLine* b) const {
    return CachedXLineTimespan(a) < CachedXLineTimespan(b);
  }

 private:
  Timespan CachedXLineTimespan(const XLine* line) const {
    DCHECK_GT(line->events_size(), 0);
    Timespan& line_timespan = line_timespan_[line];
    if (line_timespan.Instant()) {
      line_timespan = XLineTimespan(*line);
    }
    return line_timespan;
  }

  mutable absl::flat_hash_map<const XLine*, Timespan> line_timespan_;
};

}  // namespace

const XPlane* FindPlaneWithName(const XSpace& space, absl::string_view name) {
  for (const XPlane& plane : space.planes()) {
    if (plane.name() == name) return &plane;
  }
  return nullptr;
}

std::vector<const XPlane*> FindPlanesWithPrefix(const XSpace& space,
                                                absl::string_view prefix) {
  std::vector<const XPlane*> result;
  for (const XPlane& plane : space.planes()) {
    if (absl::StartsWith(plane.name(), prefix)) result.push_back(&plane);
  }
  return result;
}

XPlane* GetOrCreatePlane(XSpace* space, absl::string_view name) {
  for (XPlane& plane : *space->mutable_planes()) {
    if (plane.name() == name) return &plane;
  }
  XPlane* plane = space->add_planes();
  plane->set_name(std::string(name));
  return plane;
}

bool IsNested(const XEvent& event, const XEvent& parent) {
  return XEventTimespan(parent).Includes(XEventTimespan(event));
}

void AddOrUpdateIntStat(int64 metadata_id, int64 value, XEvent* event) {
  for (auto& stat : *event->mutable_stats()) {
    if (stat.metadata_id() == metadata_id) {
      stat.set_int64_value(value);
      return;
    }
  }
  XStat* stat = event->add_stats();
  stat->set_metadata_id(metadata_id);
  stat->set_int64_value(value);
}

void AddOrUpdateStrStat(int64 metadata_id, absl::string_view value,
                        XEvent* event) {
  for (auto& stat : *event->mutable_stats()) {
    if (stat.metadata_id() == metadata_id) {
      stat.set_str_value(std::string(value));
      return;
    }
  }
  XStat* stat = event->add_stats();
  stat->set_metadata_id(metadata_id);
  stat->set_str_value(std::string(value));
}

XEventBuilder CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    absl::string_view event_name, int64 offset_ps, int64 duration_ps,
    const absl::flat_hash_map<StatType, int64 /*stat_value*/>& stats) {
  auto event_builder = line_builder->AddEvent(
      *plane_builder->GetOrCreateEventMetadata(event_name));
  event_builder.SetOffsetPs(offset_ps);
  event_builder.SetDurationPs(duration_ps);
  for (const auto& stat_type_and_value : stats) {
    event_builder.AddStatValue(*plane_builder->GetOrCreateStatMetadata(
                                   GetStatTypeStr(stat_type_and_value.first)),
                               stat_type_and_value.second);
  }
  return event_builder;
}

XEventBuilder CreateXEvent(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    HostEventType event_type, int64 offset_ps, int64 duration_ps,
    const absl::flat_hash_map<StatType, int64 /*stat_value*/>& stats) {
  return CreateXEvent(plane_builder, line_builder,
                      GetHostEventTypeStr(event_type), offset_ps, duration_ps,
                      stats);
}

XEventBuilder CreateXEventWithStringViewMetadataValue(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    absl::string_view event_name, int64 offset_ps, int64 duration_ps,
    const absl::flat_hash_map<StatType, absl::string_view /*stat_value*/>&
        stats) {
  auto event_builder = line_builder->AddEvent(
      *plane_builder->GetOrCreateEventMetadata(event_name));
  event_builder.SetOffsetPs(offset_ps);
  event_builder.SetDurationPs(duration_ps);
  for (const auto& stat_type_and_value : stats) {
    event_builder.AddStatValue(*plane_builder->GetOrCreateStatMetadata(
                                   GetStatTypeStr(stat_type_and_value.first)),
                               stat_type_and_value.second);
  }
  return event_builder;
}

XEventBuilder CreateXEventWithIntAndStringViewMetadataValue(
    XPlaneBuilder* plane_builder, XLineBuilder* line_builder,
    absl::string_view event_name, int64 offset_ps, int64 duration_ps,
    const absl::flat_hash_map<StatType, int64 /*stat_value*/>& int_stats,
    const absl::flat_hash_map<StatType, absl::string_view /*stat_value*/>&
        str_stats) {
  auto event_builder = line_builder->AddEvent(
      *plane_builder->GetOrCreateEventMetadata(event_name));
  event_builder.SetOffsetPs(offset_ps);
  event_builder.SetDurationPs(duration_ps);
  for (const auto& stat_type_and_value : int_stats) {
    event_builder.AddStatValue(*plane_builder->GetOrCreateStatMetadata(
                                   GetStatTypeStr(stat_type_and_value.first)),
                               stat_type_and_value.second);
  }
  for (const auto& stat_type_and_value : str_stats) {
    event_builder.AddStatValue(*plane_builder->GetOrCreateStatMetadata(
                                   GetStatTypeStr(stat_type_and_value.first)),
                               stat_type_and_value.second);
  }
  return event_builder;
}

void RemovePlaneWithName(XSpace* space, absl::string_view name) {
  auto* planes = space->mutable_planes();
  planes->erase(
      std::remove_if(planes->begin(), planes->end(),
                     [&](const XPlane& plane) { return plane.name() == name; }),
      planes->end());
}

void RemoveEmptyPlanes(XSpace* space) {
  auto* planes = space->mutable_planes();
  planes->erase(std::remove_if(planes->begin(), planes->end(),
                               [&](const XPlane& plane) {
                                 return plane.lines_size() == 0;
                               }),
                planes->end());
}

void RemoveEmptyLines(XPlane* plane) {
  auto* lines = plane->mutable_lines();
  lines->erase(std::remove_if(
                   lines->begin(), lines->end(),
                   [&](const XLine& line) { return line.events_size() == 0; }),
               lines->end());
}

XPlane* FindMutablePlaneWithName(XSpace* space, absl::string_view name) {
  for (XPlane& plane : *space->mutable_planes()) {
    if (plane.name() == name) return &plane;
  }
  return nullptr;
}

XPlane* FindOrAddMutablePlaneWithName(XSpace* space, absl::string_view name) {
  XPlane* plane = FindMutablePlaneWithName(space, name);
  if (plane == nullptr) {
    plane = space->add_planes();
    plane->set_name(std::string(name));
  }
  return plane;
}

void SortXPlane(XPlane* plane) {
  for (XLine& line : *plane->mutable_lines()) {
    auto& events = *line.mutable_events();
    std::sort(events.pointer_begin(), events.pointer_end(),
              XEventsComparator());
  }
  std::sort(plane->mutable_lines()->pointer_begin(),
            plane->mutable_lines()->pointer_end(), XLinesComparator());
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
    if (line.timestamp_ns() >= static_cast<long int>(start_time_ns)) {
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
    XStat* new_stat = dst.FindOrAddMutableStat(stat_metadata->id());
    // Add or override the existing stat value except the metadata id.
    *new_stat = stat.RawStat();
    new_stat->set_metadata_id(stat_metadata->id());
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
        time_offset_ps = (line.TimestampNs() - dst_line.TimestampNs()) *
                         EnvTime::kNanosToPicos;
      }
      dst_line.SetNameIfEmpty(line.Name());
      if (!line.DisplayName().empty()) {
        dst_line.SetDisplayNameIfEmpty(line.DisplayName());
      }
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

void CreateTfFunctionCallEvent(XPlaneBuilder* plane_builder,
                               XLineBuilder* line_builder,
                               absl::string_view function_name, int64 offset_ps,
                               int64 duration_ps,
                               absl::string_view execution_mode,
                               int64 tracing_count) {
  XEventBuilder event_builder = CreateXEventWithStringViewMetadataValue(
      plane_builder, line_builder, function_name, offset_ps, duration_ps,
      {{StatType::kTfFunctionCall, execution_mode}});
  if (tracing_count >= 0) {
    // Adds the tracing_count stats only if tracing_count is valid.
    event_builder.AddStatValue(
        *plane_builder->GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kTfFunctionTracingCount)),
        tracing_count);
  }
}

}  // namespace profiler
}  // namespace tensorflow
