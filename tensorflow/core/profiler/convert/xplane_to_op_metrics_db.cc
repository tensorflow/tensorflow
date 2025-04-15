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

#include "tensorflow/core/profiler/convert/xplane_to_op_metrics_db.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/op_stack.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "plugin/tensorboard_plugin_profile/protobuf/op_metrics.pb.h"  // from @org_xprof
#include "xprof/utils/cost_utils.h"  // from @org_xprof
#include "xprof/utils/gpu_event_stats.h"  // from @org_xprof
#include "xprof/utils/hlo_module_map.h"  // from @org_xprof
#include "xprof/utils/op_metrics_db_utils.h"  // from @org_xprof
#include "xprof/utils/op_utils.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {
namespace {

using ::tensorflow::profiler::GpuEventStats;
using tsl::profiler::GetDeviceEventTimespan;

struct HLOTracker {
  uint64_t duration = 0;
  uint64_t program_id = 0;
  uint64_t group_id = 0;
  bool is_eager;
  const HloInstructionWrapper* hlo_instruction = nullptr;
  std::string hlo_op_name;

  void Reset() {
    duration = program_id = group_id = 0;
    hlo_op_name.clear();
    hlo_instruction = nullptr;
  }
};

// Type of a TensorFlow Op activity, which is either beginning or ending an Op.
enum TfActivityType { kTfOpBegin, kTfOpEnd };

// Instant activity representing the begin or end of a host-side TF Op.
struct TfActivity {
  // The timestamp in picoseconds when this activity happened.
  uint64 timestamp_ps;
  // The ID of this Op.
  uint32 tf_op_id;
  // Type of this activity.
  TfActivityType activity_type;
  // Full TF op name and type of this activity (backed by XEvent::name).
  tsl::profiler::TfOp tf_op;
  // Whether it is eagerly executed.
  bool is_eager;
};

// TF Op metrics stored as element in OpStack.
struct TfOpInfo {
  explicit TfOpInfo(uint64 ts) : start_timestamp_ps(ts) {}

  // Start timestamp in picoseconds.
  uint64 start_timestamp_ps;
  // Children duration in picoseconds.
  uint64 children_duration_ps = 0;
};

// Processes a TF-activity on particular core.
void ProcessOneTfActivity(const TfActivity& activity,
                          OpStack<TfOpInfo>* tf_op_stack,
                          TfMetricsDbData* tf_metrics_data) {
  uint32 tf_op_id = activity.tf_op_id;
  switch (activity.activity_type) {
    case kTfOpBegin: {
      tf_op_stack->Push(tf_op_id,
                        std::make_unique<TfOpInfo>(activity.timestamp_ps));
      break;
    }
    case kTfOpEnd: {
      std::unique_ptr<TfOpInfo> info = tf_op_stack->Pop(tf_op_id);
      if (info == nullptr) {
        // This happens if TraceMes overlap.
        VLOG(1) << "No begin event found for TF activity id=" << tf_op_id
                << " name=" << activity.tf_op.name
                << " type=" << activity.tf_op.type;
        break;
      }
      tsl::profiler::Timespan tf_op_span = tsl::profiler::PicoSpan(
          info->start_timestamp_ps, activity.timestamp_ps);
      tf_metrics_data->tf_metrics_db_builder.EnterOp(
          activity.tf_op.name, activity.tf_op.type, activity.is_eager,
          tf_op_span.duration_ps(), info->children_duration_ps);
      TfOpInfo* parent_info = tf_op_stack->Top();
      if (parent_info != nullptr) {
        parent_info->children_duration_ps += tf_op_span.duration_ps();
      }
      if (tsl::profiler::IsInfeedEnqueueOp(activity.tf_op.type)) {
        tf_metrics_data->tf_metrics_db_builder.EnterHostInfeedEnqueue(
            tf_op_span);
      }
      break;
    }
  }
}

// Processes all TF-activities on the given core.
void ProcessTfActivities(std::vector<TfActivity>* tf_activities,
                         TfMetricsDbData* tf_metrics_db_data) {
  if (tf_activities->empty()) return;
  absl::c_stable_sort(*tf_activities,
                      [](const TfActivity& a, const TfActivity& b) {
                        return a.timestamp_ps < b.timestamp_ps;
                      });
  OpStack<TfOpInfo> tf_op_stack;
  for (const auto& tf_activity : *tf_activities) {
    ProcessOneTfActivity(tf_activity, &tf_op_stack, tf_metrics_db_data);
  }
  SetTotalTimePs(
      tf_metrics_db_data->tf_metrics_db,
      tf_activities->back().timestamp_ps - tf_activities->front().timestamp_ps);
}

void CollectTfActivities(
    const XLineVisitor& line,
    const absl::flat_hash_map<int64_t, tsl::profiler::TfOp>& tf_ops,
    std::vector<TfActivity>* tf_activities) {
  uint32 tf_op_id = 0;
  if (IsDerivedThreadId(line.Id())) return;
  tf_activities->reserve(line.NumEvents() * 2);
  line.ForEachEvent(
      [&tf_ops, &tf_op_id, &tf_activities](const XEventVisitor& event) {
        const tsl::profiler::TfOp* tf_op = gtl::FindOrNull(tf_ops, event.Id());
        if (tf_op != nullptr) {
          ++tf_op_id;
          bool is_eager = false;
          if (std::optional<XStatVisitor> stat =
                  event.GetStat(StatType::kIsEager)) {
            is_eager = stat->IntValue();
          }
          tsl::profiler::Timespan span = event.GetTimespan();
          tf_activities->push_back(
              {span.begin_ps(), tf_op_id, kTfOpBegin, *tf_op, is_eager});
          tf_activities->push_back(
              {span.end_ps(), tf_op_id, kTfOpEnd, *tf_op, is_eager});
        }
        if (auto tf_op_stat = event.GetStat(StatType::kTfOp);
            tf_op_stat.has_value()) {
          ++tf_op_id;
          tsl::profiler::TfOp tf_op =
              tsl::profiler::ParseTfOpFullname(tf_op_stat->StrOrRefValue());
          tsl::profiler::Timespan span = event.GetTimespan();
          tf_activities->push_back(
              {span.begin_ps(), tf_op_id, kTfOpBegin, tf_op, false});
          tf_activities->push_back(
              {span.end_ps(), tf_op_id, kTfOpEnd, tf_op, false});
        }
      });
}

}  // namespace

absl::flat_hash_map<int64_t, tsl::profiler::TfOp>
CollectTfOpsFromHostThreadsXPlane(const XPlane& host_trace) {
  absl::flat_hash_map<int64_t, tsl::profiler::TfOp> tf_ops;
  for (const auto& id_metadata : host_trace.event_metadata()) {
    const XEventMetadata& metadata = id_metadata.second;
    // On the host, we have added some user-specified TraceMe's in addition to
    // the TraceMe's added to every TensorFlow op by the system. These
    // user-inserted TraceMe's have "unknown" type. We don't count them in
    // Tf-stats.
    tsl::profiler::TfOp tf_op =
        tsl::profiler::ParseTfOpFullname(metadata.name());
    if (tf_op.category != tsl::profiler::Category::kUnknown) {
      tf_ops.try_emplace(metadata.id(), tf_op);
    }
  }
  return tf_ops;
}

TfMetricsDbData ConvertHostThreadsXLineToTfMetricsDbData(
    const XLineVisitor& line,
    const absl::flat_hash_map<int64_t, tsl::profiler::TfOp>& tf_ops) {
  TfMetricsDbData tf_metrics_db_data;
  std::vector<TfActivity> tf_activities;
  CollectTfActivities(line, tf_ops, &tf_activities);
  ProcessTfActivities(&tf_activities, &tf_metrics_db_data);
  return tf_metrics_db_data;
}

void ConsumeTfMetricsDbData(TfMetricsDbData src, OpMetricsDbCombiner* dst) {
  AddIdleOp(src.tf_metrics_db);
  // Host OpMetricsDb does not need to update the number of cores a certain op
  // occurs.
  dst->Combine(src.tf_metrics_db, /*update_num_cores=*/false);
  src.tf_metrics_db.Clear();
}

OpMetricsDb ConvertHostThreadsXPlaneToOpMetricsDb(const XPlane& host_trace) {
  absl::flat_hash_map<int64_t, tsl::profiler::TfOp> tf_ops =
      CollectTfOpsFromHostThreadsXPlane(host_trace);
  OpMetricsDb result;
  OpMetricsDbCombiner combiner(&result);
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&host_trace);
  plane.ForEachLine([&tf_ops, &combiner](const XLineVisitor& line) {
    ConsumeTfMetricsDbData(
        ConvertHostThreadsXLineToTfMetricsDbData(line, tf_ops), &combiner);
  });
  return result;
}

OpMetricsDb ConvertTpuDeviceTraceXPlaneToOpMetricsDb(
    const XPlane& device_trace) {
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  XEventsOpMetricsDbBuilder builder;
  uint64_t first_op_timestamp_ps = std::numeric_limits<uint64_t>::max();
  uint64_t last_op_timestamp_ps = 0;

  struct ParentReference {
    const XEventVisitor event;
    tsl::profiler::Timespan device_timespan;
    uint64_t children_duration_ps = 0;
  };

  tsl::profiler::AncestorStack<ParentReference> event_stack(
      [&](const ParentReference& parent) {
        OpMetrics op_metrics = FromXEvent(parent.event);
        op_metrics.set_time_ps(parent.device_timespan.duration_ps());
        op_metrics.set_self_time_ps(op_metrics.time_ps() -
                                    parent.children_duration_ps);
        builder.AddOpMetric(op_metrics, GetOpKeyFromXEvent(parent.event));
      },
      [](const ParentReference& parent, const ParentReference& child) {
        return parent.device_timespan.Includes(child.device_timespan);
      },
      [](ParentReference& parent, ParentReference& child) {
        parent.children_duration_ps += child.device_timespan.duration_ps();
      });

  auto track_first_and_last_op_timestamps = [&](const XEventVisitor& event) {
    tsl::profiler::Timespan timespan = GetDeviceEventTimespan(event);
    first_op_timestamp_ps =
        std::min(first_op_timestamp_ps, timespan.begin_ps());
    last_op_timestamp_ps = std::max(last_op_timestamp_ps, timespan.end_ps());
  };

  plane.ForEachLine([&](const XLineVisitor& line) {
    if (line.Name() == tsl::profiler::kSparseCoreStepLineName ||
        line.Name() == tsl::profiler::kStepLineName) {
      line.ForEachEvent(track_first_and_last_op_timestamps);
    }
    if (!tsl::profiler::IsOpLineName(line.Name())) return;
    line.ForEachEvent([&](const XEventVisitor& event) {
      tsl::profiler::Timespan timespan = GetDeviceEventTimespan(event);
      track_first_and_last_op_timestamps(event);

      event_stack.Push({.event = event, .device_timespan = timespan});
    });
    event_stack.Flush();
  });

  return builder.Finalize(last_op_timestamp_ps - first_op_timestamp_ps);
}

void AggregateHloFunc(HLOTracker& current, DeviceOpMetricsDbBuilder& metricDb) {
  if (current.hlo_instruction == nullptr) return;
  auto performance_info_wrapper =
      current.hlo_instruction->GetPerformanceInfoWrapper();
  auto flops = 0;
  auto bytes_accessed = 0;
  if (performance_info_wrapper != nullptr) {
    flops = performance_info_wrapper->flops();
    bytes_accessed = performance_info_wrapper->bytes_accessed();
  }
  metricDb.EnterOp(
      current.program_id, current.hlo_op_name,
      current.hlo_instruction->Category(), current.hlo_instruction->TfOpName(),
      current.hlo_instruction->DeduplicatedName(), current.is_eager, 1,
      current.duration, 0, performance_info_wrapper->DeviceFlops(),
      performance_info_wrapper->bytes_accessed(),
      ConvertPerformanceInfo(
          performance_info_wrapper->memory_accessed_breakdown(), 1),
      performance_info_wrapper->ModelFlops(),
      current.hlo_instruction->Expression());
  current.Reset();
}

OpMetricsDb ConvertDeviceTraceXPlaneToOpMetricsDb(
    const XPlane& device_trace, const HloModuleMap& hlo_module_map) {
  OpMetricsDb result;
  DeviceOpMetricsDbBuilder device_op_metrics_db_builder(&result);

  int64_t first_op_offset_ps = kint64max;
  int64_t last_op_offset_ps = 0;

  TfOpRoofLineCostEstimator op_level_cost_estimator;
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  HLOTracker current;
  plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) return;
    line.ForEachEvent([&](const XEventVisitor& event) {
      first_op_offset_ps = std::min(first_op_offset_ps, event.OffsetPs());
      last_op_offset_ps = std::max(last_op_offset_ps, event.EndOffsetPs());

      GpuEventStats stats(&event);
      if (stats.IsXlaOp()) {
        const auto* hlo_instruction = GetHloInstruction(
            hlo_module_map, stats.program_id, stats.hlo_op_names.back());
        if (hlo_instruction != nullptr) {
          if (stats.hlo_op_names.back() != current.hlo_op_name ||
              stats.group_id != current.group_id) {
            AggregateHloFunc(current, device_op_metrics_db_builder);
          }
          // Merge identical and contiguous HLOs.
          current.hlo_instruction = hlo_instruction;
          current.hlo_op_name = stats.hlo_op_names.back();
          current.duration += event.DurationPs();
          current.is_eager = stats.is_eager;
          current.program_id = *stats.program_id;
          if (stats.group_id.has_value()) {
            current.group_id = *stats.group_id;
          }
        }
      } else if (stats.IsTfOp()) {
        AggregateHloFunc(current, device_op_metrics_db_builder);
        tsl::profiler::TfOp tf_op =
            tsl::profiler::ParseTfOpFullname(stats.tf_op_fullname);
        PerformanceInfo perf_info;
        if (tf_op.category != tsl::profiler::Category::kUnknown) {
          auto costs = op_level_cost_estimator.Predict(event);
          // NOTE: events are per kernel, but costs are per tf-ops.
          perf_info.set_flops(costs.flops);
          perf_info.set_bytes_accessed(costs.bytes_accessed);
        }
        std::string name = absl::StrCat(tf_op.name, "/", event.Name());
        device_op_metrics_db_builder.EnterOp(
            /*program_id=*/0,
            /**name=*/name,
            /**category=*/tf_op.type,
            /*provenance=*/stats.tf_op_fullname, "", stats.is_eager,
            /*occurrences=*/1, event.DurationPs(),
            /*children_time_ps=*/0, perf_info.flops(),
            perf_info.bytes_accessed());
      }
    });
    AggregateHloFunc(current, device_op_metrics_db_builder);
  });
  SetTotalTimePs(
      result, last_op_offset_ps ? last_op_offset_ps - first_op_offset_ps : 0);
  AddIdleOp(result);
  return result;
}

}  // namespace profiler
}  // namespace tensorflow
