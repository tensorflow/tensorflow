/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/profiler/utils/derived_timeline.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

// TODO(profiler): Once we capture HLO protos for xla/gpu, we should use that
// to look up tensorflow op name from hlo_module/hlo_op.
absl::string_view DummySymbolResolver(absl::string_view hlo_module,
                                      absl::string_view hlo_op) {
  return absl::string_view();
}

// Helper for deriving an XLine from events in another XLine.
class DerivedXLineBuilder {
 public:
  DerivedXLineBuilder(XPlaneBuilder* plane, int64 line_id,
                      absl::string_view name, int64 timestamp_ns,
                      std::vector<DerivedXLineBuilder*> dependent_lines)
      : line_(plane->GetOrCreateLine(line_id)) {
    line_.SetName(name);
    line_.SetTimestampNs(timestamp_ns);
    dependent_lines_ = std::move(dependent_lines);
  }

  void ExpandOrAddEvents(const std::vector<XEvent>& event_per_level) {
    for (int level = 0; level < event_per_level.size(); ++level) {
      ExpandOrAddLevelEvent(event_per_level[level], level);
    }
  }

  void ExpandOrAddEvent(const XEvent& event) {
    ExpandOrAddLevelEvent(event, /*level=*/0);
  }

  // Reset last events lower than the given level.
  void ResetLastEvents(int level = -1) {
    for (int i = level + 1; i < last_event_by_level_.size(); ++i) {
      last_event_by_level_[i] = absl::nullopt;
    }
  }

 private:
  // If the last event of the given level has the same metadata, expands it to
  // include the time until the given event's (offset_ps + duration_ps).
  // Otherwise, adds a new event and clears last_event_by_level_ for the levels
  // below the given level and all levels of the dependent lines. Clearing
  // last_event_by_level_ prevents a nested event from growing larger than the
  // parent event(s).
  void ExpandOrAddLevelEvent(const XEvent& event, int level) {
    int64 offset_ps = event.offset_ps();
    int64 duration_ps = event.duration_ps();
    auto& last_event = last_event_by_level_[level];
    // If last_event is not nullptr, its offset must be less than or equal to
    // the given event's offset.
    DCHECK(!last_event || last_event->OffsetPs() <= offset_ps);
    if (last_event && last_event->MetadataId() == event.metadata_id()) {
      // If last_event is not nullptr and metadata is same, merge the given
      // event into last_event.
      last_event->SetDurationPs((offset_ps + duration_ps) -
                                last_event->OffsetPs());
    } else {
      // Otherwise, create a new event for the given level.
      last_event = line_.AddEvent(event);
      // Reset last events lower than the given level.
      ResetLastEvents(level);
      if (level == 0) ResetDependentLines();
    }
  }

  void ResetDependentLines() {
    for (DerivedXLineBuilder* line : dependent_lines_) {
      line->ResetLastEvents();
    }
  }

  XLineBuilder line_;
  absl::flat_hash_map<int, absl::optional<XEventBuilder>> last_event_by_level_;
  std::vector<DerivedXLineBuilder*> dependent_lines_;
};

const absl::string_view kDerivedLineSteps = "Steps";
const absl::string_view kDerivedLineTensorFlowNameScope =
    "TensorFlow Name Scope";
const absl::string_view kDerivedLineTensorFlowOps = "TensorFlow Ops";
const absl::string_view kDerivedLineXlaModules = "XLA Modules";
const absl::string_view kDerivedLineXlaOps = "XLA Ops";
const absl::string_view kDerivedLineKernelLaunch = "Launch Stats";
const absl::string_view kAnnotationDelimiter = "::";

XEvent CreateXEvent(const XEventVisitor& src_event_visitor,
                    const XEventMetadata& metadata,
                    int64 group_id_stat_metadata_id,
                    absl::optional<int64> group_id) {
  XEvent event;
  event.set_metadata_id(metadata.id());
  // TODO(b/150498419): Normalize with the line start time.
  event.set_offset_ps(src_event_visitor.OffsetPs());
  event.set_duration_ps(src_event_visitor.DurationPs());
  if (group_id) {
    XStat* stat = event.add_stats();
    stat->set_metadata_id(group_id_stat_metadata_id);
    stat->set_int64_value(*group_id);
  }
  return event;
}

void ProcessTfOpEvent(const XEventVisitor& event,
                      absl::string_view tf_op_full_name,
                      absl::optional<int64> group_id,
                      XPlaneBuilder* plane_builder,
                      DerivedXLineBuilder* tf_name_scope_line_builder,
                      DerivedXLineBuilder* tf_op_line_builder) {
  int64 group_id_stat_metadata_id =
      plane_builder->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId))
          ->id();
  TfOp tf_op = ParseTfOpFullname(tf_op_full_name);
  Category category = tf_op.category;
  if (category == Category::kTensorFlow || category == Category::kJax) {
    std::vector<XEvent> name_scope_event_per_level;
    for (const auto& tf_name_scope : ParseTfNameScopes(tf_op)) {
      name_scope_event_per_level.push_back(CreateXEvent(
          event, *plane_builder->GetOrCreateEventMetadata(tf_name_scope),
          group_id_stat_metadata_id, group_id));
    }
    tf_name_scope_line_builder->ExpandOrAddEvents(name_scope_event_per_level);
  }
  XEventMetadata* tf_op_event_metadata =
      plane_builder->GetOrCreateEventMetadata(tf_op_full_name);
  // Set the display name to op_type so that the events of the same op_type have
  // the same color in the trace viewer.
  tf_op_event_metadata->set_display_name(TfOpEventName(tf_op));
  tf_op_line_builder->ExpandOrAddEvent(CreateXEvent(
      event, *tf_op_event_metadata, group_id_stat_metadata_id, group_id));
}

}  // namespace

void DeriveEventsFromAnnotations(const SymbolResolver& symbol_resolver,
                                 const EventGroupNameMap& event_group_name_map,
                                 XPlane* device_trace, bool step_info_only) {
  // Merge and sort events by Timespan as they come from different lines.
  std::vector<XEventVisitor> events;
  uint64 start_timestamp_ns = 0;
  XPlaneVisitor device_plane = CreateTfXPlaneVisitor(device_trace);
  device_plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) return;  // Skip overhead line.
    start_timestamp_ns = line.TimestampNs();
    line.ForEachEvent(
        [&](const XEventVisitor& event) { events.push_back(event); });
  });
  absl::c_sort(events);

  XPlaneBuilder plane(device_trace);
  DerivedXLineBuilder tf_ops(&plane, kThreadIdTfOp, kDerivedLineTensorFlowOps,
                             start_timestamp_ns, {});
  DerivedXLineBuilder tf_name_scope(&plane, kThreadIdTfNameScope,
                                    kDerivedLineTensorFlowNameScope,
                                    start_timestamp_ns, {&tf_ops});
  DerivedXLineBuilder hlo_ops(&plane, kThreadIdHloOp, kDerivedLineXlaOps,
                              start_timestamp_ns, {});
  DerivedXLineBuilder hlo_modules(&plane, kThreadIdHloModule,
                                  kDerivedLineXlaModules, start_timestamp_ns,
                                  {&tf_ops, &tf_name_scope, &hlo_ops});
  DerivedXLineBuilder steps(&plane, kThreadIdStepInfo, kDerivedLineSteps,
                            start_timestamp_ns,
                            {&tf_ops, &tf_name_scope, &hlo_ops, &hlo_modules});
  int64 group_id_stat_metadata_id =
      plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId))->id();
  int64 step_name_stat_metadata_id =
      plane.GetOrCreateStatMetadata(GetStatTypeStr(StatType::kStepName))->id();

  // Process events in order by start time.
  for (const XEventVisitor& event : events) {
    absl::string_view tf_op_full_name;
    absl::string_view hlo_module_name;
    std::vector<absl::string_view> hlo_op_names;
    absl::optional<int64> group_id;
    bool is_kernel = false;
    event.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Type() == StatType::kGroupId) {
        group_id = stat.IntValue();
      } else if (stat.Type() == StatType::kLevel0) {
        tf_op_full_name = stat.StrOrRefValue();
      } else if (stat.Type() == StatType::kHloOp) {
        hlo_op_names =
            absl::StrSplit(stat.StrOrRefValue(), kAnnotationDelimiter);
      } else if (stat.Type() == StatType::kHloModule) {
        hlo_module_name = stat.StrOrRefValue();
      } else if (stat.Type() == StatType::kKernelDetails) {
        is_kernel = true;
      }
    });

    if (group_id) {
      XEvent step_event = CreateXEvent(
          event, *plane.GetOrCreateEventMetadata(absl::StrCat(*group_id)),
          group_id_stat_metadata_id, group_id);
      if (auto group_name = gtl::FindOrNull(event_group_name_map, *group_id)) {
        XStat* stat = step_event.add_stats();
        stat->set_metadata_id(step_name_stat_metadata_id);
        stat->set_str_value(*group_name);
      }
      steps.ExpandOrAddEvent(step_event);
    }

    if (step_info_only) continue;

    // For HLO/TF op lines, only use kernel events (i.e. excluding memcpy or
    // allocation events).
    if (!is_kernel) continue;

    if (!hlo_module_name.empty()) {
      hlo_modules.ExpandOrAddEvent(
          CreateXEvent(event, *plane.GetOrCreateEventMetadata(hlo_module_name),
                       group_id_stat_metadata_id, group_id));
    }

    if (!hlo_op_names.empty()) {  // GPU kernel compiled by XLA
      DCHECK(!hlo_module_name.empty());
      std::vector<XEvent> hlo_op_event_per_level;
      for (absl::string_view hlo_op_name : hlo_op_names) {
        DCHECK(!hlo_op_name.empty());
        hlo_op_event_per_level.push_back(
            CreateXEvent(event, *plane.GetOrCreateEventMetadata(hlo_op_name),
                         group_id_stat_metadata_id, group_id));
      }
      hlo_ops.ExpandOrAddEvents(hlo_op_event_per_level);
      auto tf_op_name = symbol_resolver(hlo_module_name, hlo_op_names.back());
      if (!tf_op_name.empty()) {
        ProcessTfOpEvent(event, tf_op_name, group_id, &plane, &tf_name_scope,
                         &tf_ops);
      }
    } else if (!tf_op_full_name.empty()) {  // GPU kernel not compiled by XLA
      ProcessTfOpEvent(event, tf_op_full_name, group_id, &plane, &tf_name_scope,
                       &tf_ops);
    }
  }
  RemoveEmptyLines(device_trace);
}

void DeriveEventsFromHostTrace(const XPlane* host_trace,
                               const EventGroupNameMap& event_group_name_map,
                               std::vector<XPlane*> device_traces) {
  struct GroupLaunchInfo {  // "Group" normally means step.
    Timespan timespan;
    int32 num_launches = 0;
    uint64 max_launch_time_ps = 0ULL;
    uint64 total_launch_time_ps = 0ULL;
  };
  typedef absl::flat_hash_map<uint64 /*group_id*/, GroupLaunchInfo>
      DeviceLaunchInfo;

  int num_devices = device_traces.size();
  std::vector<DeviceLaunchInfo> per_device_launch_info(num_devices);

  XPlaneVisitor host_plane = CreateTfXPlaneVisitor(host_trace);
  host_plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) return;
    line.ForEachEvent([&](const XEventVisitor& event) {
      absl::optional<int64> group_id;
      absl::optional<int64> device_id;
      absl::optional<int64> correlation_id;
      // Filter out API calls for cuEventRecord/cuEventQuery/cuCtxSynchronize
      // etc for now. TODO: find a better way to filter out only the memcpy and
      // kernel launch events.
      if (absl::StartsWith(event.Name(), "cu")) return;
      event.ForEachStat([&](const XStatVisitor& stat) {
        if (stat.Type() == StatType::kGroupId) {
          group_id = stat.IntValue();
        } else if (stat.Type() == StatType::kDeviceId) {
          device_id = stat.IntValue();
        } else if (stat.Type() == StatType::kCorrelationId) {
          correlation_id = stat.IntValue();
        }
      });
      if (group_id && device_id && correlation_id && *device_id >= 0 &&
          *device_id < num_devices) {
        // This is a launch event on a known device.
        GroupLaunchInfo& group_launch_info =
            per_device_launch_info[*device_id][*group_id];
        Timespan& group_span = group_launch_info.timespan;
        Timespan event_span = event.GetTimespan();
        if (group_launch_info.num_launches) {  // Existing group.
          uint64 begin_ps =
              std::min(group_span.begin_ps(), event_span.begin_ps());
          uint64 end_ps = std::max(group_span.end_ps(), event_span.end_ps());
          group_span = Timespan::FromEndPoints(begin_ps, end_ps);
        } else {
          group_span = event_span;
        }
        ++group_launch_info.num_launches;
        group_launch_info.max_launch_time_ps = std::max(
            group_launch_info.max_launch_time_ps, event_span.duration_ps());
        group_launch_info.total_launch_time_ps += event_span.duration_ps();
      }
    });
  });

  uint64 host_plane_start = GetStartTimestampNs(*host_trace);
  for (int i = 0; i < num_devices; ++i) {
    if (per_device_launch_info[i].empty()) continue;
    uint64 device_plane_start = GetStartTimestampNs(*device_traces[i]);
    XPlaneBuilder device_plane(device_traces[i]);
    XLineBuilder launch_line =
        device_plane.GetOrCreateLine(kThreadIdKernelLaunch);
    launch_line.SetName(kDerivedLineKernelLaunch);
    launch_line.SetTimestampNs(std::min(device_plane_start, host_plane_start));
    for (const auto& it : per_device_launch_info[i]) {
      uint64 group_id = it.first;
      const GroupLaunchInfo& group_info = it.second;
      if (auto group_name = gtl::FindOrNull(event_group_name_map, group_id)) {
        XEventBuilder device_event =
            launch_line.AddEvent(*device_plane.GetOrCreateEventMetadata(
                absl::StrCat("Launch Stats for ", *group_name)));
        device_event.SetTimestampNs(
            host_plane_start + PicosToNanos(group_info.timespan.begin_ps()));
        device_event.SetDurationPs(group_info.timespan.duration_ps());
        device_event.AddStatValue(*device_plane.GetOrCreateStatMetadata(
                                      GetStatTypeStr(StatType::kGroupId)),
                                  group_id);
        device_event.AddStatValue(
            *device_plane.GetOrCreateStatMetadata("num_launches"),
            group_info.num_launches);
        device_event.AddStatValue(
            *device_plane.GetOrCreateStatMetadata("max_launch_time_us"),
            PicosToMicros(group_info.max_launch_time_ps));
        device_event.AddStatValue(
            *device_plane.GetOrCreateStatMetadata("avg_launch_time_us"),
            PicosToMicros(group_info.total_launch_time_ps /
                          group_info.num_launches));
      }
    }
  }
}

void GenerateDerivedTimeLines(const EventGroupNameMap& event_group_name_map,
                              XSpace* space, bool step_info_only) {
  for (XPlane& plane : *space->mutable_planes()) {
    // Derived timelines only generated for device traces.
    if (plane.id() == kHostPlaneId) continue;
    DeriveEventsFromAnnotations(DummySymbolResolver, event_group_name_map,
                                &plane, step_info_only);
  }
}

void GenerateDerivedTimeLines(const EventGroupNameMap& event_group_name_map,
                              const std::vector<XPlane*>& device_traces,
                              bool step_info_only) {
  for (XPlane* plane : device_traces) {
    DeriveEventsFromAnnotations(DummySymbolResolver, event_group_name_map,
                                plane, step_info_only);
  }
}

}  // namespace profiler
}  // namespace tensorflow
