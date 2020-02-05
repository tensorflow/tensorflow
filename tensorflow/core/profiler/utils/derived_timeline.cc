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

#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

// Helper for deriving an XLine from events in another XLine.
// Merges consecutive events with the same metadata.
class DerivedXLineBuilder {
 public:
  DerivedXLineBuilder(XPlaneBuilder* plane, int64 line_id,
                      absl::string_view name, int64 timestamp_ns)
      : line_(plane->GetOrCreateLine(line_id)),
        group_id_stats_(plane->GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kGroupId))) {
    line_.SetName(std::string(name));
    line_.SetTimestampNs(timestamp_ns);
  }

  // If the last event of the given level has the same metadata and group_id,
  // expands it to include the time until (offset_ps + duration_ps). Otherwise,
  // adds a new event and clears last_event_by_level_ for the levels below the
  // given level. Clearing last_event_by_level_ prevents a nested event from
  // growing larger than the parent event(s).
  void ExpandOrAddEvent(const XEventMetadata& event_metadata,
                        const XEventVisitor& event,
                        absl::optional<int64> group_id, int level = 0) {
    int64 offset_ps = event.OffsetPs(), duration_ps = event.DurationPs();
    auto& last_event = last_event_by_level_[level];
    DCHECK(!last_event || last_event->OffsetPs() <= offset_ps);
    if (last_event && last_event->MetadataId() == event_metadata.id() &&
        last_group_id_ == group_id) {
      last_event->SetDurationPs((offset_ps + duration_ps) -
                                last_event->OffsetPs());
    } else {
      last_event = line_.AddEvent(event_metadata);
      last_event->SetOffsetPs(offset_ps);
      last_event->SetDurationPs(duration_ps);
      last_group_id_ = group_id;
      if (group_id) last_event->AddStatValue(*group_id_stats_, *group_id);
      for (int i = level + 1; i < last_event_by_level_.size(); ++i) {
        last_event_by_level_[i] = absl::nullopt;
      }
    }
  }

 private:
  XLineBuilder line_;
  absl::flat_hash_map<int, absl::optional<XEventBuilder>> last_event_by_level_;
  absl::optional<int64> last_group_id_;
  XStatMetadata* group_id_stats_;
};

constexpr absl::string_view kDerivedLineSteps = "Steps";
constexpr absl::string_view kDerivedLineTensorFlowOps = "TensorFlow Ops";
constexpr absl::string_view kDerivedLineXlaModules = "XLA Modules";
constexpr absl::string_view kDerivedLineXlaOps = "XLA Ops";
constexpr absl::string_view kAnnotationDelimiter = "::";

}  // namespace

void DeriveEventsFromAnnotations(const SymbolResolver& symbol_resolver,
                                 const EventGroupNameMap& event_group_name_map,
                                 XPlane* device_trace) {
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
  DerivedXLineBuilder steps(&plane, kThreadIdStepInfo, kDerivedLineSteps,
                            start_timestamp_ns);
  DerivedXLineBuilder tf_ops(&plane, kThreadIdTfOp, kDerivedLineTensorFlowOps,
                             start_timestamp_ns);
  DerivedXLineBuilder hlo_ops(&plane, kThreadIdHloOp, kDerivedLineXlaModules,
                              start_timestamp_ns);
  DerivedXLineBuilder hlo_modules(&plane, kThreadIdHloModule,
                                  kDerivedLineXlaOps, start_timestamp_ns);

  // Process events in order by start time.
  for (const XEventVisitor& event : events) {
    absl::string_view tf_op_fullname;
    absl::string_view hlo_module_name;
    std::vector<absl::string_view> hlo_op_names;
    absl::optional<int64> group_id;
    bool is_kernel = false;
    event.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Type() == StatType::kGroupId) {
        group_id = stat.IntValue();
      } else if (stat.Type() == StatType::kLevel0) {
        tf_op_fullname = stat.StrValue();
      } else if (stat.Type() == StatType::kHloOp) {
        hlo_op_names = absl::StrSplit(stat.StrValue(), kAnnotationDelimiter);
      } else if (stat.Type() == StatType::kHloModule) {
        hlo_module_name = stat.StrValue();
      } else if (stat.Type() == StatType::kKernelDetails) {
        is_kernel = true;
      }
    });

    if (group_id) {
      if (auto group_name = gtl::FindOrNull(event_group_name_map, *group_id)) {
        steps.ExpandOrAddEvent(*plane.GetOrCreateEventMetadata(*group_name),
                               event, group_id);
      }
    }

    if (!is_kernel) {
      // For HLO/TF op lines, only use kernel events, (i.e. excluding memcpy or
      // allocation events).
      continue;
    }

    if (!hlo_module_name.empty()) {
      hlo_modules.ExpandOrAddEvent(
          *plane.GetOrCreateEventMetadata(hlo_module_name), event, group_id);
    }

    if (!hlo_op_names.empty()) {  // GPU kernel compiled by XLA
      DCHECK(!hlo_module_name.empty());
      int level = 0;
      for (absl::string_view hlo_op_name : hlo_op_names) {
        DCHECK(!hlo_op_name.empty());
        hlo_ops.ExpandOrAddEvent(*plane.GetOrCreateEventMetadata(hlo_op_name),
                                 event, group_id, level);
        auto tf_op_name = symbol_resolver(hlo_module_name, hlo_op_name);
        if (!tf_op_name.empty()) {
          tf_ops.ExpandOrAddEvent(*plane.GetOrCreateEventMetadata(tf_op_name),
                                  event, group_id, level);
        }
        ++level;
      }
    } else if (!tf_op_fullname.empty()) {  // GPU kernel not compiled by XLA
      tf_ops.ExpandOrAddEvent(*plane.GetOrCreateEventMetadata(tf_op_fullname),
                              event, group_id);
    }
  }
  RemoveEmptyLines(device_trace);
}

}  // namespace profiler
}  // namespace tensorflow
