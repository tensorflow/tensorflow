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

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/xla_op_utils.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/gpu_event_stats.h"
#include "tensorflow/core/profiler/utils/group_events.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/timespan.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tensorflow/core/util/stats_calculator.h"

namespace tensorflow {
namespace profiler {

void ProcessTfOpEvent(absl::string_view tf_op_full_name,
                      absl::string_view low_level_event_name,
                      Timespan event_span, absl::optional<int64_t> group_id,
                      XPlaneBuilder* plane_builder,
                      DerivedXLineBuilder* tf_name_scope_line_builder,
                      DerivedXLineBuilder* tf_op_line_builder) {
  TfOp tf_op = ParseTfOpFullname(tf_op_full_name);
  Category category = tf_op.category;
  if (category == Category::kTensorFlow || category == Category::kJax) {
    tf_name_scope_line_builder->ExpandOrAddEvents(
        plane_builder->GetOrCreateEventsMetadata(ParseTfNameScopes(tf_op)),
        event_span, group_id, low_level_event_name);
  }
  XEventMetadata* tf_op_event_metadata =
      plane_builder->GetOrCreateEventMetadata(tf_op_full_name);
  // Set the display name to op_type so that the events of the same op_type have
  // the same color in the trace viewer.
  tf_op_event_metadata->set_display_name(TfOpEventName(tf_op));
  tf_op_line_builder->ExpandOrAddEvent(*tf_op_event_metadata, event_span,
                                       group_id, low_level_event_name);
}

DerivedXEventBuilder::DerivedXEventBuilder(
    XEventBuilder event, absl::optional<int64_t> group_id,
    absl::string_view low_level_event_name)
    : event_(std::move(event)), group_id_(group_id) {
  if (!low_level_event_name.empty()) {
    low_level_event_names_.insert(std::string(low_level_event_name));
  }
}

bool DerivedXEventBuilder::ShouldExpand(
    const XEventMetadata& event_metadata, absl::optional<int64_t> group_id,
    absl::string_view low_level_event_name) const {
  return event_.MetadataId() == event_metadata.id() && group_id_ == group_id &&
         !low_level_event_names_.contains(low_level_event_name);
}

void DerivedXEventBuilder::Expand(Timespan event_span,
                                  absl::string_view low_level_event_name) {
  Timespan timespan = event_.GetTimespan();
  DCHECK_LE(timespan.begin_ps(), event_span.begin_ps());
  timespan.ExpandToInclude(event_span);
  event_.SetTimespan(timespan);
  if (!low_level_event_name.empty()) {
    low_level_event_names_.insert(std::string(low_level_event_name));
  }
}

DerivedXLineBuilder::DerivedXLineBuilder(
    XPlaneBuilder* plane, int64_t line_id, absl::string_view name,
    int64_t timestamp_ns, std::vector<DerivedXLineBuilder*> dependent_lines)
    : group_id_stat_metadata_(
          plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId))),
      level_stat_metadata_(plane->GetOrCreateStatMetadata("l")),
      line_(plane->GetOrCreateLine(line_id)),
      dependent_lines_(std::move(dependent_lines)) {
  line_.SetName(name);
  line_.SetTimestampNs(timestamp_ns);
}

void DerivedXLineBuilder::ExpandOrAddLevelEvent(
    const XEventMetadata& event_metadata, Timespan event_span,
    absl::optional<int64_t> group_id, absl::string_view low_level_event_name,
    int level) {
  auto& last_event = last_event_by_level_[level];
  if (last_event && last_event->ShouldExpand(event_metadata, group_id,
                                             low_level_event_name)) {
    // Expand the last event to cover the given event.
    last_event->Expand(event_span, low_level_event_name);
  } else {
    // Otherwise, reset the last events lower than or equal to the given level.
    ResetLastEvents(level);
    // And create a new event for the given level.
    XEventBuilder event = line_.AddEvent(event_metadata);
    event.SetTimespan(event_span);
    if (group_id.has_value()) {
      event.AddStatValue(*group_id_stat_metadata_, *group_id);
    }
    event.AddStatValue(*level_stat_metadata_, level);
    last_event.emplace(std::move(event), group_id, low_level_event_name);
  }
}

void DerivedXLineBuilder::ResetLastEvents(int level) {
  for (int i = level, end = last_event_by_level_.size(); i < end; ++i) {
    last_event_by_level_[i].reset();
  }
  if (level == 0) {
    for (DerivedXLineBuilder* line : dependent_lines_) {
      line->ResetLastEvents(0);
    }
  }
}

void AddGroupMetadataToStepEvents(const GroupMetadataMap& group_metadata_map,
                                  XLineBuilder& line) {
  if (group_metadata_map.empty()) return;
  XPlaneBuilder* plane = line.Plane();
  const XStatMetadata* group_id_stat_metadata =
      plane->GetStatMetadata(GetStatTypeStr(StatType::kGroupId));
  if (group_id_stat_metadata == nullptr) return;
  const XStatMetadata* step_name_stat_metadata =
      plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kStepName));
  line.ForEachEvent([&](XEventBuilder event) {
    const XStat* group_id_stat = event.GetStat(*group_id_stat_metadata);
    if (group_id_stat != nullptr) {
      int64_t group_id = group_id_stat->int64_value();
      if (const GroupMetadata* group_metadata =
              gtl::FindOrNull(group_metadata_map, group_id)) {
        // TODO(b/160255693): Change the event name directly.
        event.AddStatValue(*step_name_stat_metadata, group_metadata->name);
      }
    }
  });
}

void DeriveEventsFromAnnotations(const SymbolResolver& symbol_resolver,
                                 const GroupMetadataMap& group_metadata_map,
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
  DerivedXLineBuilder tf_ops(&plane, kThreadIdTfOp, kTensorFlowOpLineName,
                             start_timestamp_ns, {});
  DerivedXLineBuilder tf_name_scope(&plane, kThreadIdTfNameScope,
                                    kTensorFlowNameScopeLineName,
                                    start_timestamp_ns, {&tf_ops});
  DerivedXLineBuilder hlo_ops(&plane, kThreadIdHloOp, kXlaOpLineName,
                              start_timestamp_ns, {});
  DerivedXLineBuilder hlo_modules(&plane, kThreadIdHloModule,
                                  kXlaModuleLineName, start_timestamp_ns,
                                  {&tf_name_scope, &hlo_ops});
  DerivedXLineBuilder steps(&plane, kThreadIdStepInfo, kStepLineName,
                            start_timestamp_ns, {&hlo_modules});
  DerivedXLineBuilder source(&plane, kThreadIdSource, kSourceLineName,
                             start_timestamp_ns, {});

  // Process events in order by start time.
  for (const XEventVisitor& event : events) {
    Timespan event_span = event.GetTimespan();
    GpuEventStats stats(&event);
    if (stats.group_id) {
      steps.ExpandOrAddEvent(
          *plane.GetOrCreateEventMetadata(absl::StrCat(*stats.group_id)),
          event_span, stats.group_id);
    }

    if (step_info_only) continue;

    // For HLO/TF op lines, only use kernel events (i.e. excluding memcpy or
    // allocation events).
    if (!stats.IsKernel()) continue;

    if (!stats.hlo_module_name.empty()) {
      std::string name = stats.program_id
                             ? HloModuleNameWithProgramId(stats.hlo_module_name,
                                                          *stats.program_id)
                             : std::string(stats.hlo_module_name);
      hlo_modules.ExpandOrAddEvent(
          *plane.GetOrCreateEventMetadata(std::move(name)), event_span,
          stats.group_id);
    }

    if (stats.IsXlaOp()) {
      DCHECK(!stats.hlo_module_name.empty());
      hlo_ops.ExpandOrAddEvents(
          plane.GetOrCreateEventsMetadata(stats.hlo_op_names), event_span,
          stats.group_id);
      auto symbol = symbol_resolver(stats.program_id, stats.hlo_module_name,
                                    stats.hlo_op_names.back());
      if (!symbol.tf_op_name.empty()) {
        ProcessTfOpEvent(symbol.tf_op_name,
                         /*low_level_event_name=*/event.Name(), event_span,
                         stats.group_id, &plane, &tf_name_scope, &tf_ops);
      }
      if (!symbol.source_info.empty()) {
        source.ExpandOrAddEvent(
            *plane.GetOrCreateEventMetadata(symbol.source_info), event_span,
            stats.group_id);
      }
    } else if (stats.IsTfOp()) {
      ProcessTfOpEvent(stats.tf_op_fullname,
                       /*low_level_event_name=*/event.Name(), event_span,
                       stats.group_id, &plane, &tf_name_scope, &tf_ops);
    }
  }
  AddGroupMetadataToStepEvents(group_metadata_map, steps.Line());
  RemoveEmptyLines(device_trace);
}

void DeriveEventsFromHostTrace(const XPlane* host_trace,
                               const GroupMetadataMap& group_metadata_map,
                               std::vector<XPlane*> device_traces) {
  struct GroupLaunchInfo {  // "Group" normally means step.
    Timespan timespan;
    Stat<uint64_t> stat;

    void AddEventTimespan(Timespan event_span) {
      if (stat.count() == 0) {
        timespan = event_span;
      } else {
        timespan.ExpandToInclude(event_span);
      }
      stat.UpdateStat(event_span.duration_ps());
    }
  };
  using DeviceLaunchInfo =
      absl::flat_hash_map<int64_t /*group_id*/, GroupLaunchInfo>;

  const int num_devices = device_traces.size();
  std::vector<DeviceLaunchInfo> per_device_launch_info(num_devices);

  XPlaneVisitor host_plane = CreateTfXPlaneVisitor(host_trace);
  host_plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) return;
    line.ForEachEvent([&](const XEventVisitor& event) {
      // Filter out API calls for cuEventRecord/cuEventQuery/cuCtxSynchronize
      // etc for now. TODO: find a better way to filter out only the memcpy and
      // kernel launch events.
      if (absl::StartsWith(event.Name(), "cu")) return;
      LaunchEventStats stats(&event);
      if (stats.group_id.has_value() && stats.IsLaunch() &&
          0 <= *stats.device_id && *stats.device_id < num_devices) {
        // This is a launch event on a known device.
        GroupLaunchInfo& group_launch_info =
            per_device_launch_info[*stats.device_id][*stats.group_id];
        group_launch_info.AddEventTimespan(event.GetTimespan());
      }
    });
  });

  uint64 host_plane_start = GetStartTimestampNs(*host_trace);
  for (int i = 0; i < num_devices; ++i) {
    if (per_device_launch_info[i].empty()) continue;
    uint64 device_plane_start = GetStartTimestampNs(*device_traces[i]);

    XPlaneBuilder device_plane(device_traces[i]);
    const XStatMetadata& group_id_stat_metadata =
        *device_plane.GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kGroupId));
    const XStatMetadata& num_launches_stat_metadata =
        *device_plane.GetOrCreateStatMetadata("num_launches");
    const XStatMetadata& max_launch_time_us_stat_metadata =
        *device_plane.GetOrCreateStatMetadata("max_launch_time_us");
    const XStatMetadata& avg_launch_time_us_stat_metadata =
        *device_plane.GetOrCreateStatMetadata("avg_launch_time_us");

    XLineBuilder launch_line =
        device_plane.GetOrCreateLine(kThreadIdKernelLaunch);
    launch_line.SetName(kKernelLaunchLineName);
    launch_line.SetTimestampNs(std::min(device_plane_start, host_plane_start));
    for (const auto& kv : per_device_launch_info[i]) {
      int64_t group_id = kv.first;
      const GroupLaunchInfo& group_info = kv.second;
      if (const GroupMetadata* group_metadata =
              gtl::FindOrNull(group_metadata_map, group_id)) {
        XEventBuilder device_event =
            launch_line.AddEvent(*device_plane.GetOrCreateEventMetadata(
                absl::StrCat("Launch Stats for ", group_metadata->name)));
        device_event.SetTimespan(group_info.timespan);
        device_event.AddStatValue(group_id_stat_metadata, group_id);
        device_event.AddStatValue(num_launches_stat_metadata,
                                  group_info.stat.count());
        device_event.AddStatValue(max_launch_time_us_stat_metadata,
                                  PicoToMicro(group_info.stat.max()));
        device_event.AddStatValue(avg_launch_time_us_stat_metadata,
                                  PicoToMicro(group_info.stat.avg()));
      }
    }
  }
}

void GenerateDerivedTimeLines(const GroupMetadataMap& group_metadata_map,
                              XSpace* space, bool step_info_only) {
  // TODO(profiler): Once we capture HLO protos for xla/gpu, we should use that
  // to look up tensorflow op name from hlo_module/hlo_op.
  auto dummy_symbol_resolver =
      [](absl::optional<uint64_t> program_id, absl::string_view hlo_module,
         absl::string_view hlo_op) { return Symbol(); };
  std::vector<XPlane*> device_traces =
      FindMutablePlanesWithPrefix(space, kGpuPlanePrefix);
  for (XPlane* plane : device_traces) {
    DeriveEventsFromAnnotations(dummy_symbol_resolver, group_metadata_map,
                                plane, step_info_only);
  }
}

}  // namespace profiler
}  // namespace tensorflow
