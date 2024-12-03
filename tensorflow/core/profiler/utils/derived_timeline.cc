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
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/profiler/convert/xla_op_utils.h"
#include "xla/tsl/profiler/utils/device_utils.h"
#include "xla/tsl/profiler/utils/group_events.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"
#include "xla/tsl/profiler/utils/trace_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_visitor.h"
#include "xla/tsl/util/stats_calculator.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/gpu_event_stats.h"
#include "tensorflow/core/profiler/utils/hlo_module_map.h"
#include "tensorflow/core/profiler/utils/hlo_proto_map.h"
#include "tensorflow/core/profiler/utils/host_offload_utils.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {
namespace {

using tsl::profiler::FindMutableTensorCorePlanes;

inline std::string HloModuleEventName(const GpuEventStats& stats) {
  return stats.program_id ? tsl::profiler::HloModuleNameWithProgramId(
                                stats.hlo_module_name, *stats.program_id)
                          : std::string(stats.hlo_module_name);
}

// Returns a prefix that uniquely identifies the HLO module.
inline std::string HloOpEventPrefix(const GpuEventStats& stats) {
  return stats.program_id ? absl::StrCat(*stats.program_id, "/")
                          : absl::StrCat(stats.hlo_module_name, "/");
}

std::vector<XEventMetadata*> GetOrCreateHloOpEventsMetadata(
    XPlaneBuilder& xplane, const GpuEventStats& stats, const Symbol symbol) {
  DCHECK(stats.IsXlaOp());
  std::vector<XEventMetadata*> hlo_op_events_metadata;
  hlo_op_events_metadata.reserve(stats.hlo_op_names.size());
  // Prepend an HLO module identifier so HLO operators with the same name but in
  // different modules have different metadata.
  std::string hlo_op_event_prefix = HloOpEventPrefix(stats);
  for (absl::string_view hlo_op_name : stats.hlo_op_names) {
    XEventMetadata* hlo_op_event_metadata = xplane.GetOrCreateEventMetadata(
        absl::StrCat(hlo_op_event_prefix, hlo_op_name));
    // Display the HLO name without the module name in tools.
    if (hlo_op_event_metadata->display_name().empty()) {
      hlo_op_event_metadata->set_display_name(std::string(hlo_op_name));
    }
    hlo_op_events_metadata.push_back(hlo_op_event_metadata);
    if (!symbol.hlo_text.empty()) {
      XStatsBuilder<XEventMetadata> event_stats(hlo_op_event_metadata, &xplane);
      event_stats.SetOrAddStatValue(*xplane.GetOrCreateStatMetadata("hlo_text"),
                                    symbol.hlo_text);
    }
  }
  return hlo_op_events_metadata;
}

// Get the derived line id for a given derived line in group which starts from
// first_derived_line_id.
// According to definition in trace_utils.h, the derived lines are:
// kThreadIdTfNameScope to kThreadIdSource. Keep the line id sequence in each
// group as this original group..
inline int64_t GetDerivedLineId(int64_t first_derived_line_id,
                                int64_t target_line_id) {
  return first_derived_line_id + (target_line_id - kThreadIdTfNameScope);
}

// Get the derived line name for a given derived line in group which starts from
// first_derived_line_id.
std::string GetDerivedLineName(int64_t first_derived_line_id,
                               int64_t target_line_id,
                               absl::Span<const int64_t> source_line_ids) {
  int64_t offset = target_line_id - kThreadIdTfNameScope;
  std::string suffix;
  if (first_derived_line_id != kThreadIdTfNameScope &&
      !source_line_ids.empty()) {
    suffix = absl::StrCat(" - from #", source_line_ids[0]);
  }
  switch (offset) {
    case kThreadIdTfNameScope - kThreadIdTfNameScope:
      return absl::StrCat(kTensorFlowNameScopeLineName, suffix);
    case kThreadIdHloOp - kThreadIdTfNameScope:
      return absl::StrCat(kXlaOpLineName, suffix);
    case kThreadIdHloModule - kThreadIdTfNameScope:
      return absl::StrCat(kXlaModuleLineName, suffix);
    case kThreadIdTfOp - kThreadIdTfNameScope:
      return absl::StrCat(kTensorFlowOpLineName, suffix);
    case kThreadIdSource - kThreadIdTfNameScope:
      return absl::StrCat(kSourceLineName, suffix);
    default:
      LOG(ERROR) << "Invalid target line id: " << target_line_id
                 << " for first_derived_line_id: " << first_derived_line_id;
      return absl::StrCat("UnknownDerived#", first_derived_line_id + offset);
  }
}

// Derive events from the given line ids using annotations.
// Returns the derived line ids in the order of tf_name_scope, tf_op, hlo_op,
// hlo_module, source. Where the derived line id for tf_name_scope is
// first_derived_line_id.
std::vector<int64_t> DeriveEventsFromAnnotationsForLines(
    const SymbolResolver& symbol_resolver, XPlane* device_trace,
    absl::Span<const int64_t> line_ids, int64_t first_derived_line_id,
    const ScopeRangeIdTree* scope_range_id_tree = nullptr) {
  XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(device_trace);

  XPlaneBuilder plane_builder(device_trace);
  int64_t start_timestamp_ns = GetStartTimestampNs(*device_trace);
  DerivedXLineBuilder tf_ops(
      &plane_builder, GetDerivedLineId(first_derived_line_id, kThreadIdTfOp),
      GetDerivedLineName(first_derived_line_id, kThreadIdTfOp, line_ids),
      start_timestamp_ns, {});
  DerivedXLineBuilder tf_name_scope(
      &plane_builder,
      GetDerivedLineId(first_derived_line_id, kThreadIdTfNameScope),
      GetDerivedLineName(first_derived_line_id, kThreadIdTfNameScope, line_ids),
      start_timestamp_ns, {&tf_ops});
  DerivedXLineBuilder hlo_ops(
      &plane_builder, GetDerivedLineId(first_derived_line_id, kThreadIdHloOp),
      GetDerivedLineName(first_derived_line_id, kThreadIdHloOp, line_ids),
      start_timestamp_ns, {});
  DerivedXLineBuilder hlo_modules(
      &plane_builder,
      GetDerivedLineId(first_derived_line_id, kThreadIdHloModule),
      GetDerivedLineName(first_derived_line_id, kThreadIdHloModule, line_ids),
      start_timestamp_ns, {&tf_name_scope, &hlo_ops});
  DerivedXLineBuilder source(
      &plane_builder, GetDerivedLineId(first_derived_line_id, kThreadIdSource),
      GetDerivedLineName(first_derived_line_id, kThreadIdSource, line_ids),
      start_timestamp_ns, {});

  // Declare this vector here so that its memory will be reused during the loop,
  // instead of being allocated and deallocated for each iteration.
  std::vector<std::optional<int64_t>> level_range_ids;
  for (const XEventVisitor& event :
       GetSortedEvents<XEventVisitor>(plane_visitor, false, line_ids)) {
    GpuEventStats stats(&event);
    // For HLO/TF op lines, only use kernel events (i.e. excluding memcpy or
    // allocation events). Also CudaGraph executions are also treated as
    // kernel events.
    if (!stats.IsKernel() && !stats.IsCudaGraphExecution()) continue;
    tsl::profiler::Timespan event_span = event.GetTimespan();

    if ((!stats.hlo_module_name.empty() || stats.IsXlaOp())) {
      level_range_ids.clear();
      if (stats.scope_range_id.has_value()) {
        level_range_ids.push_back(stats.scope_range_id);
        if (scope_range_id_tree) {
          for (auto it = scope_range_id_tree->find(*stats.scope_range_id);
               it != scope_range_id_tree->end();
               it = scope_range_id_tree->find(it->second)) {
            level_range_ids.push_back(it->second);
          }
        }
      }
      // Now, level_range_ids looks like:
      // [child_level_n, child_level_n-1, ..., child_level_1, root_level]
    }

    if (!stats.hlo_module_name.empty()) {
      // back() of the level_range_ids, i.e. root_level in above comment,
      // is the scope range id of HLO module.
      hlo_modules.ExpandOrAddEvent(
          *plane_builder.GetOrCreateEventMetadata(HloModuleEventName(stats)),
          event_span, stats.group_id,
          level_range_ids.empty() ? std::nullopt : level_range_ids.back());
    }

    if (stats.IsXlaOp()) {
      auto symbol = symbol_resolver(stats.program_id, stats.hlo_module_name,
                                    stats.hlo_op_names.back());
      auto hlo_events_metadata =
          GetOrCreateHloOpEventsMetadata(plane_builder, stats, symbol);
      // level_range_ids, if not empty, should be of same size as
      // hlo_events_metadata. If not of same size, do not use those ids.
      absl::Span<std::optional<int64_t>> xla_op_level_range_ids = {};
      if (level_range_ids.size() == hlo_events_metadata.size()) {
        std::reverse(level_range_ids.begin(), level_range_ids.end());
        // after reverse, the level_range_ids looks like:
        // [root_level, child_level_1, ..., child_level_n-1, child_level_n]
        xla_op_level_range_ids = absl::MakeSpan(level_range_ids);
      }
      hlo_ops.ExpandOrAddEvents(hlo_events_metadata, event_span, stats.group_id,
                                xla_op_level_range_ids);

      // If the kernel event is nodes of a CudaGraph or a whole cuda graph
      // exec, try to mark extra stats to to corresponding XLA op event here.
      if (stats.cuda_graph_id_for_inner_node.has_value() &&
          *stats.cuda_graph_id_for_inner_node != 0) {
        int level = static_cast<int>(hlo_events_metadata.size()) - 1;
        if (level >= 0) {
          hlo_ops.AddStatToLevelEvent(level, *hlo_ops.GetCudaGraphIdMetadata(),
                                      *stats.cuda_graph_id_for_inner_node);
          if (stats.correlation_id.has_value()) {
            hlo_ops.AddStatToLevelEvent(level,
                                        *hlo_ops.GetCorrelationIdMetadata(),
                                        *stats.correlation_id);
          }
        }
      }

      if (!symbol.tf_op_name.empty()) {
        ProcessTfOpEvent(symbol.tf_op_name, event_span, stats.group_id,
                         plane_builder, tf_name_scope, tf_ops);
      }
      if (!symbol.source_info.empty()) {
        source.ExpandOrAddEvent(
            *plane_builder.GetOrCreateEventMetadata(symbol.source_info),
            event_span, stats.group_id);
      }
    } else if (stats.IsTfOp()) {
      ProcessTfOpEvent(stats.tf_op_fullname, event_span, stats.group_id,
                       plane_builder, tf_name_scope, tf_ops);
    }
  }
  return {tf_name_scope.Line().Id(), tf_ops.Line().Id(),
          hlo_modules.Line().Id(), hlo_ops.Line().Id(), source.Line().Id()};
}

}  // namespace

void ProcessTfOpEvent(absl::string_view tf_op_full_name,
                      tsl::profiler::Timespan event_span,
                      std::optional<int64_t> group_id,
                      XPlaneBuilder& plane_builder,
                      DerivedXLineBuilder& tf_name_scope_line_builder,
                      DerivedXLineBuilder& tf_op_line_builder) {
  tsl::profiler::TfOp tf_op = tsl::profiler::ParseTfOpFullname(tf_op_full_name);
  tsl::profiler::Category category = tf_op.category;
  if (category == tsl::profiler::Category::kTensorFlow ||
      category == tsl::profiler::Category::kJax) {
    tf_name_scope_line_builder.ExpandOrAddEvents(
        plane_builder.GetOrCreateEventsMetadata(
            tsl::profiler::ParseTfNameScopes(tf_op)),
        event_span, group_id);
  }
  XEventMetadata* tf_op_event_metadata =
      plane_builder.GetOrCreateEventMetadata(tf_op_full_name);
  // Set the display name to op_type so that the events of the same op_type have
  // the same color in the trace viewer.
  if (tf_op_event_metadata->display_name().empty()) {
    tf_op_event_metadata->set_display_name(tsl::profiler::TfOpEventName(tf_op));
  }
  tf_op_line_builder.ExpandOrAddEvent(*tf_op_event_metadata, event_span,
                                      group_id);
}

DerivedXEventBuilder::DerivedXEventBuilder(
    XEventBuilder event, std::optional<int64_t> group_id,
    std::optional<int64_t> scope_range_id)
    : event_(std::move(event)),
      group_id_(group_id),
      scope_range_id_(scope_range_id) {}

bool DerivedXEventBuilder::ShouldExpand(
    const XEventMetadata& event_metadata, std::optional<int64_t> group_id,
    std::optional<int64_t> scope_range_id) const {
  return event_.MetadataId() == event_metadata.id() && group_id_ == group_id &&
         (!scope_range_id.has_value() || !scope_range_id_.has_value() ||
          scope_range_id_ == scope_range_id);
}

void DerivedXEventBuilder::Expand(tsl::profiler::Timespan event_span) {
  tsl::profiler::Timespan timespan = event_.GetTimespan();
  DCHECK_LE(timespan.begin_ps(), event_span.begin_ps());
  timespan.ExpandToInclude(event_span);
  event_.SetTimespan(timespan);
}

DerivedXLineBuilder::DerivedXLineBuilder(
    XPlaneBuilder* plane, int64_t line_id, absl::string_view name,
    int64_t timestamp_ns, std::vector<DerivedXLineBuilder*> dependent_lines)
    : group_id_stat_metadata_(
          plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId))),
      correlation_id_metadata_(plane->GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kCorrelationId))),
      cuda_graph_id_metadata_(plane->GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kCudaGraphId))),
      line_(plane->GetOrCreateLine(line_id)),
      dependent_lines_(std::move(dependent_lines)) {
  line_.SetName(name);
  line_.SetTimestampNs(timestamp_ns);
}

void DerivedXLineBuilder::ExpandOrAddEvent(
    const XEventMetadata& event_metadata, tsl::profiler::Timespan event_span,
    std::optional<int64_t> group_id, std::optional<int64_t> scope_range_id) {
  ExpandOrAddLevelEvent(event_metadata, event_span, group_id, scope_range_id,
                        /*level=*/0);
}

void DerivedXLineBuilder::ExpandOrAddEvents(
    const std::vector<XEventMetadata*>& events_metadata_per_level,
    tsl::profiler::Timespan event_span, std::optional<int64_t> group_id,
    absl::Span<std::optional<int64_t>> scope_range_ids) {
  if (events_metadata_per_level.empty()) return;

  size_t current_nested_level = events_metadata_per_level.size();
  for (size_t level = 0; level < current_nested_level; ++level) {
    ExpandOrAddLevelEvent(
        *events_metadata_per_level[level], event_span, group_id,
        level < scope_range_ids.size() ? scope_range_ids[level] : std::nullopt,
        level);
  }
  ResetLastEvents(current_nested_level);
}

void DerivedXLineBuilder::ExpandOrAddLevelEvent(
    const XEventMetadata& event_metadata, tsl::profiler::Timespan event_span,
    std::optional<int64_t> group_id, std::optional<int64_t> scope_range_id,
    int level) {
  auto& last_event = last_event_by_level_[level];
  if (last_event &&
      last_event->ShouldExpand(event_metadata, group_id, scope_range_id)) {
    // Expand the last event to cover the given event.
    last_event->Expand(event_span);
  } else {
    // Otherwise, reset the last events lower than or equal to the given level.
    ResetLastEvents(level);
    // And create a new event for the given level.
    XEventBuilder event = line_.AddEvent(event_metadata);
    event.SetTimespan(event_span);
    if (group_id.has_value()) {
      event.AddStatValue(*group_id_stat_metadata_, *group_id);
    }
    last_event.emplace(std::move(event), group_id, scope_range_id);
  }
}

void DerivedXLineBuilder::AddStatToLevelEvent(int level,
                                              const XStatMetadata& metadata,
                                              int64_t value) {
  if (auto it = last_event_by_level_.find(level);
      it != last_event_by_level_.end() && it->second.has_value()) {
    it->second->SetOrAddStatValue(metadata, value);
  }
}

void DerivedXLineBuilder::AddStatToLevelEvent(int level,
                                              const XStatMetadata& metadata,
                                              uint64_t value) {
  if (auto it = last_event_by_level_.find(level);
      it != last_event_by_level_.end() && it->second.has_value()) {
    it->second->SetOrAddStatValue(metadata, value);
  }
}

// When deriving a bunch of events with the same timespan, there could be
// indeterministic behavior of how trace viewer stacking these events.
// This function will shrink the stack of events with the same timespan when
// necessary. Event at top of stack might shrink more than event at the
// bottom. Because the time unit in trace viewer is nanosecond, therefore the
// minimum difference is 1ns. However to prevent shrink induced inconsitency,
// we can not shrink more than the duration of event at the top of the stack.
void DerivedXLineBuilder::AdjustDurationForTraceViewer(int level) {
  if (level >= last_event_by_level_.size() || !last_event_by_level_[level])
    return;

  int max_level = level;
  for (; max_level < last_event_by_level_.size(); ++max_level) {
    if (!last_event_by_level_[max_level].has_value()) {
      break;
    }
  }
  --max_level;
  if (max_level <= level) return;
  auto& event_on_top_stack = *last_event_by_level_[max_level];
  tsl::profiler::Timespan timespan = event_on_top_stack.GetTimespan();
  // We will at most shrink the top of the stack to 1ns.
  int64_t max_shrink_ns = timespan.duration_ps() / 1000 - 1;
  int64_t shrink_ns = 0;
  std::optional<tsl::profiler::Timespan> last_level_timespan;
  for (int i = level; i <= max_level; ++i) {
    auto& current_event = *last_event_by_level_[i];
    if (shrink_ns < max_shrink_ns &&
        last_level_timespan == current_event.GetTimespan()) {
      shrink_ns++;
    }
    last_level_timespan = current_event.GetTimespan();
    if (shrink_ns) {
      current_event.SetTimespan(tsl::profiler::Timespan::FromEndPoints(
          last_level_timespan->begin_ps(),
          last_level_timespan->end_ps() - 1000 * shrink_ns));
    }
  }
}

void DerivedXLineBuilder::ResetLastEvents(int level) {
  AdjustDurationForTraceViewer(level);
  for (int i = level, end = last_event_by_level_.size(); i < end; ++i) {
    last_event_by_level_[i].reset();
  }
  if (level == 0) {
    for (DerivedXLineBuilder* line : dependent_lines_) {
      line->ResetLastEvents(0);
    }
  }
}

void DeriveStepEventsFromGroups(
    const tsl::profiler::GroupMetadataMap& group_metadata_map,
    XPlane* device_trace) {
  XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(device_trace);
  const XStatMetadata* group_id_stat_metadata =
      plane_visitor.GetStatMetadataByType(StatType::kGroupId);
  if (group_id_stat_metadata == nullptr) return;
  XPlaneBuilder plane_builder(device_trace);
  int64_t start_timestamp_ns = GetStartTimestampNs(*device_trace);
  DerivedXLineBuilder steps(&plane_builder, kThreadIdStepInfo, kStepLineName,
                            start_timestamp_ns, {});
  for (const XEventVisitor& event_visitor :
       GetSortedEvents<XEventVisitor>(plane_visitor)) {
    std::optional<XStatVisitor> group_id_stat =
        event_visitor.GetStat(StatType::kGroupId, *group_id_stat_metadata);
    if (group_id_stat.has_value()) {
      int64_t group_id = group_id_stat->IntValue();
      steps.ExpandOrAddEvent(
          *plane_builder.GetOrCreateEventMetadata(absl::StrCat(group_id)),
          event_visitor.GetTimespan(), group_id);
    }
  }
  AddGroupMetadataToStepEvents(group_metadata_map, steps.Line());
}

void DeriveEventsFromAnnotations(const SymbolResolver& symbol_resolver,
                                 XPlane* device_trace,
                                 const ScopeRangeIdTree* scope_range_id_tree) {
  if (tsl::profiler::GetDeviceType(*device_trace) !=
      tsl::profiler::DeviceType::kGpu) {
    DeriveEventsFromAnnotationsForLines(symbol_resolver, device_trace, {},
                                        kThreadIdTfNameScope);
  } else {
    // TODO: Currently we derive events only from the line with the most number
    // of events. We should consider deriving events from all lines in the
    // future, also then we need to utilize the derived relation provided by
    // DeriveEventsFromAnnotationsForLines(), and find solid way to sort all
    // lines.
    int64_t line_id_with_most_events = -1;
    int64_t max_num_events_per_line = -1;
    {
      XPlaneVisitor plane_visitor =
          tsl::profiler::CreateTfXPlaneVisitor(device_trace);
      plane_visitor.ForEachLine([&](const XLineVisitor& line) {
        if (IsDerivedThreadId(line.Id())) return;
        int num_events = line.NumEvents();
        // make sure strong ordering
        if (num_events > max_num_events_per_line ||
            (num_events == max_num_events_per_line &&
             line.Id() < line_id_with_most_events)) {
          max_num_events_per_line = num_events;
          line_id_with_most_events = line.Id();
        }
      });
    }

    if (line_id_with_most_events >= 0) {
      DeriveEventsFromAnnotationsForLines(
          symbol_resolver, device_trace, {line_id_with_most_events},
          kThreadIdTfNameScope, scope_range_id_tree);
    }
  }
  RemoveEmptyLines(device_trace);
}

void DeriveEventsFromHostTrace(
    const XPlane* host_trace,
    const tsl::profiler::GroupMetadataMap& group_metadata_map,
    std::vector<XPlane*> device_traces) {
  struct GroupLaunchInfo {  // "Group" normally means step.
    tsl::profiler::Timespan timespan;
    tsl::Stat<uint64_t> stat;

    void AddEventTimespan(tsl::profiler::Timespan event_span) {
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

  XPlaneVisitor host_plane = tsl::profiler::CreateTfXPlaneVisitor(host_trace);
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

  int64_t host_plane_start = GetStartTimestampNs(*host_trace);
  for (int i = 0; i < num_devices; ++i) {
    if (per_device_launch_info[i].empty()) continue;
    int64_t device_plane_start = GetStartTimestampNs(*device_traces[i]);

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
      if (const tsl::profiler::GroupMetadata* group_metadata =
              gtl::FindOrNull(group_metadata_map, group_id)) {
        XEventBuilder device_event =
            launch_line.AddEvent(*device_plane.GetOrCreateEventMetadata(
                absl::StrCat("Launch Stats for ", group_metadata->name)));
        device_event.SetTimespan(group_info.timespan);
        device_event.AddStatValue(group_id_stat_metadata, group_id);
        device_event.AddStatValue(num_launches_stat_metadata,
                                  group_info.stat.count());
        device_event.AddStatValue(
            max_launch_time_us_stat_metadata,
            tsl::profiler::PicoToMicro(group_info.stat.max()));
        device_event.AddStatValue(
            avg_launch_time_us_stat_metadata,
            tsl::profiler::PicoToMicro(group_info.stat.avg()));
      }
    }
  }
}

void GenerateDerivedTimeLines(
    const tsl::profiler::GroupMetadataMap& group_metadata_map, XSpace* space) {
  HloModuleMap hlo_module_map;
  {
    HloProtoMap hlo_proto_map;
    hlo_proto_map.AddHloProtosFromXSpace(*space);
    for (const auto& [program_id, hlo_proto] : hlo_proto_map) {
      AddHloProto(hlo_module_map, program_id, *hlo_proto);
    }
  }

  auto symbol_resolver = [&](absl::optional<uint64_t> program_id,
                             absl::string_view hlo_module,
                             absl::string_view hlo_op) -> Symbol {
    Symbol output;
    const auto* hlo_instruction =
        GetHloInstruction(hlo_module_map, program_id, hlo_op);
    if (hlo_instruction != nullptr) {
      output.tf_op_name = hlo_instruction->op_full_name();
      output.source_info = std::string(hlo_instruction->source_info());
    }
    return output;
  };

  ScopeRangeIdTree scope_range_id_tree;
  const XPlane* namespace_tree_plane =
      FindPlaneWithName(*space, tsl::profiler::kScopeRangeIdTreePlaneName);
  if (namespace_tree_plane) {
    XPlaneVisitor namespace_tree_visitor =
        tsl::profiler::CreateTfXPlaneVisitor(namespace_tree_plane);
    namespace_tree_visitor.ForEachStat([&](const XStatVisitor& stat) {
      scope_range_id_tree.emplace(stat.Id(), stat.IntValue());
    });
  }

  std::vector<XPlane*> device_planes =
      FindMutablePlanesWithPrefix(space, kGpuPlanePrefix);
  for (XPlane* plane : device_planes) {
    DeriveStepEventsFromGroups(group_metadata_map, plane);
    DeriveEventsFromAnnotations(symbol_resolver, plane, &scope_range_id_tree);
  }

  const XPlane* host_plane = FindPlaneWithName(*space, kHostThreadsPlaneName);
  if (host_plane) {
    DeriveEventsFromHostTrace(host_plane, group_metadata_map, device_planes);
  }
  for (XPlane* plane : FindMutableTensorCorePlanes(space)) {
    DeriveLinesFromStats(plane);
    SortXPlane(plane);
  }
}

void DeriveLinesFromStats(XPlane* device_trace) {
  XPlaneVisitor plane_visitor =
      tsl::profiler::CreateTfXPlaneVisitor(device_trace);
  XPlaneBuilder plane_builder(device_trace);
  int64_t start_timestamp_ns = GetStartTimestampNs(*device_trace);
  DerivedXLineBuilder tf_ops(
      &plane_builder, tensorflow::profiler::kThreadIdTfOp,
      tensorflow::profiler::kTensorFlowOpLineName, start_timestamp_ns, {});
  DerivedXLineBuilder tf_name_scope(
      &plane_builder, tensorflow::profiler::kThreadIdTfNameScope,
      tensorflow::profiler::kTensorFlowNameScopeLineName, start_timestamp_ns,
      {&tf_ops});
  DerivedXLineBuilder source(
      &plane_builder, tensorflow::profiler::kThreadIdSource,
      tensorflow::profiler::kSourceLineName, start_timestamp_ns, {});

  HostOffloadEventProcessor host_offload_event_processor(&plane_builder,
                                                         start_timestamp_ns);

  for (const XEventVisitor& event :
       GetSortedEvents<XEventVisitor>(plane_visitor, true)) {
    tsl::profiler::Timespan event_span = event.GetTimespan();
    std::optional<absl::string_view> tf_op_name;
    std::optional<absl::string_view> source_info;
    std::optional<uint64_t> group_id;
    std::optional<uint64_t> is_async;
    auto for_each_stat = [&](const XStatVisitor& stat) {
      if (stat.Type() == StatType::kTfOp) {
        tf_op_name = stat.StrOrRefValue();
      } else if (stat.Type() == StatType::kGroupId) {
        group_id = stat.IntOrUintValue();
      } else if (stat.Type() == StatType::kSourceInfo) {
        source_info = stat.StrOrRefValue();
      } else if (stat.Type() == StatType::kIsAsync) {
        is_async = stat.IntOrUintValue();
      }
    };
    event.Metadata().ForEachStat(for_each_stat);
    event.ForEachStat(for_each_stat);

    if (is_async && *is_async) continue;  // Disregard asynchronous events.

    if (tf_op_name && !tf_op_name->empty()) {
      ProcessTfOpEvent(*tf_op_name, event_span, group_id, plane_builder,
                       tf_name_scope, tf_ops);
    }
    if (source_info && !source_info->empty()) {
      source.ExpandOrAddEvent(
          *plane_builder.GetOrCreateEventMetadata(*source_info), event_span,
          group_id);
    }
    if (host_offload_event_processor.IsHostOffloadOpName(event)) {
      host_offload_event_processor.ProcessHostOffloadOpEvent(event, group_id);
    }
  }

  RemoveEmptyLines(device_trace);
}

void DeriveLinesForXlaCpuOps(XPlane* host_trace) {
  if (host_trace == nullptr ||
      !absl::StartsWith(host_trace->name(), kHostThreadsPlaneName))
    return;
  XPlaneVisitor visitor = tsl::profiler::CreateTfXPlaneVisitor(host_trace);
  XPlane destination_plane;
  XPlaneBuilder plane_builder(&destination_plane);
  int64_t line_id = tsl::profiler::kThreadIdHostXlaRegionStart;
  visitor.ForEachLine([&](const XLineVisitor& line) {
    int64_t start_timestamp_ns = line.TimestampNs();
    DerivedXLineBuilder tf_ops(
        &plane_builder, line_id++,
        absl::StrCat(line.Name(), "-",
                     tensorflow::profiler::kTensorFlowOpLineName),
        start_timestamp_ns, {});
    DerivedXLineBuilder tf_name_scope(
        &plane_builder, line_id++,
        absl::StrCat(line.Name(), "-",
                     tensorflow::profiler::kTensorFlowNameScopeLineName),
        start_timestamp_ns, {&tf_ops});
    DerivedXLineBuilder xla_cpu_ops(
        &plane_builder, line_id++,
        absl::StrCat(line.Name(), "-", tsl::profiler::kXlaModuleLineName),
        start_timestamp_ns, {});
    line.ForEachEvent([&](const XEventVisitor& event) {
      std::optional<std::string> hlo_module_name;
      std::optional<std::string> framework_op_name;
      event.ForEachStat([&](const XStatVisitor& stat) {
        if (!stat.Type().has_value()) return;
        // TODO: Add additional stats for framework ops.
        switch (stat.Type().value()) {
          case StatType::kHloModule:
            hlo_module_name = stat.StrOrRefValue();
            break;
          case StatType::kTfOp:
            framework_op_name = stat.StrOrRefValue();
            break;
        }
      });
      if (hlo_module_name.has_value()) {
        xla_cpu_ops.ExpandOrAddEvent(
            *plane_builder.GetOrCreateEventMetadata(*hlo_module_name),
            event.GetTimespan(), std::nullopt);
        if (framework_op_name.has_value()) {
          ProcessTfOpEvent(*framework_op_name, event.GetTimespan(),
                           std::nullopt, plane_builder, tf_name_scope, tf_ops);
        }
      }
    });
  });
  RemoveEmptyLines(&destination_plane);
  MergePlanes(destination_plane, host_trace);
}

}  // namespace profiler
}  // namespace tensorflow
