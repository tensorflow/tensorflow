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

#include "tensorflow/core/profiler/convert/xplane_to_step_events.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/profiler/utils/tf_op_utils.h"
#include "xla/tsl/profiler/utils/tf_xplane_visitor.h"
#include "xla/tsl/profiler/utils/timespan.h"
#include "xla/tsl/profiler/utils/tpu_xplane_utils.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"
#include "tsl/profiler/protobuf/xplane.pb.h"
#include "xprof/utils/event_span.h"  // from @org_xprof
#include "xprof/utils/op_metrics_db_utils.h"  // from @org_xprof

namespace tensorflow {
namespace profiler {
namespace {

inline AllReduceInfo GetAllReduceInfo(const XEventVisitor& event,
                                      uint64_t all_reduce_unique_id) {
  AllReduceInfo collective_ops;
  collective_ops.set_id(all_reduce_unique_id);
  collective_ops.set_start_time_ps(event.TimestampPs());
  if (auto device_offset_ps_stat = event.GetStat(StatType::kDeviceOffsetPs)) {
    collective_ops.set_start_time_ps(device_offset_ps_stat->IntOrUintValue());
  }
  collective_ops.set_end_time_ps(event.EndTimestampPs());
  if (auto device_duration_ps_stat =
          event.GetStat(StatType::kDeviceDurationPs)) {
    collective_ops.set_end_time_ps(collective_ops.start_time_ps() +
                                   device_duration_ps_stat->IntOrUintValue());
  }
  if (auto all_reduce_id_stat = event.GetStat(StatType::kAllReduceId)) {
    collective_ops.set_all_reduce_id(all_reduce_id_stat->IntOrUintValue());
  }
  if (auto bytes_accessed_stat =
          event.Metadata().GetStat(StatType::kBytesAccessed)) {
    collective_ops.set_byte_size(bytes_accessed_stat->IntOrUintValue());
  }
  return collective_ops;
}

inline bool IsExplicitHostStepMarker(absl::string_view event_name) {
  return (absl::StartsWith(event_name, "train") ||
          absl::StartsWith(event_name, "test") ||
          absl::StartsWith(event_name, "TraceContext")) &&
         !absl::StrContains(event_name, "/");
}

// Returns true if the given event_name should be considered as real computation
// on CPU.
inline bool IsRealCpuCompute(absl::string_view event_name) {
  bool not_real = absl::StartsWith(event_name, "EagerExecute") ||
                  absl::StartsWith(event_name, "EagerLocalExecute") ||
                  absl::StartsWith(event_name, "EagerKernelExecute") ||
                  absl::StartsWith(event_name, "FunctionRun") ||
                  IsExplicitHostStepMarker(event_name);
  return !not_real;
}

uint64 ParseNumBytesFromMemcpyDetail(absl::string_view memcpy_detail) {
  const std::vector<absl::string_view> params =
      absl::StrSplit(memcpy_detail, absl::ByAnyChar(":\n"));

  // Processes value pairs.
  for (uint32 ii = 0; ii < params.size(); ii += 2) {
    if (params[ii] != "num_bytes") continue;
    uint64 value = 0;
    if (absl::SimpleAtoi(params[ii + 1], &value)) return value;
    break;
  }
  return 0ULL;
}

EventType ClassifyGpuCompute(absl::string_view event_name,
                             absl::string_view tensor_shapes) {
  if (tensor_shapes.empty()) {
    // Deduces the precision from the name.
    return (absl::StrContains(event_name, "half") ||
            absl::StrContains(event_name, "fp16"))
               ? DEVICE_COMPUTE_16
               : DEVICE_COMPUTE_32;
  } else {
    // Deduces the precision from the shapes.
    return (absl::StrContains(tensor_shapes, "half")) ? DEVICE_COMPUTE_16
                                                      : DEVICE_COMPUTE_32;
  }
}

EventType ClassifyGpuEvent(absl::string_view event_name,
                           absl::string_view tensor_shapes) {
  tsl::profiler::TfOp tf_op = tsl::profiler::ParseTfOpFullname(event_name);
  if (tsl::profiler::IsMemcpyHToDOp(tf_op)) {
    return HOST_TO_DEVICE;
  } else if (tsl::profiler::IsMemcpyDToHOp(tf_op)) {
    return DEVICE_TO_HOST;
  } else if (tsl::profiler::IsMemcpyDToDOp(tf_op)) {
    return DEVICE_TO_DEVICE;
  } else if (absl::StartsWithIgnoreCase(event_name, "nccl")) {
    return DEVICE_COLLECTIVES;
  } else {
    return ClassifyGpuCompute(event_name, tensor_shapes);
  }
}

EventType ClassifyCpuEvent(absl::string_view event_name, bool has_device,
                           bool has_correlation_id) {
  tsl::profiler::TfOp tf_op = tsl::profiler::ParseTfOpFullname(event_name);
  if (tsl::profiler::IsInfeedEnqueueOp(tf_op) ||
      tsl::profiler::IsMemcpyHToDOp(tf_op)) {
    return HOST_TO_DEVICE;
  } else if (tsl::profiler::IsMemcpyHToHOp(tf_op)) {
    return HOST_TO_HOST;
  } else if (has_device && (has_correlation_id ||
                            absl::StartsWithIgnoreCase(
                                event_name, "ExecutorState::Process"))) {
    // TODO(b/150420972): Separate runtime overhead from actual compute for
    // CPU-only.
    return HOST_PREPARE;
  } else if (absl::StartsWithIgnoreCase(event_name, "IteratorGetNext")) {
    return HOST_WAIT_INPUT;
  } else {
    return HOST_COMPUTE;
  }
}

}  // namespace

StepEvents ConvertHostThreadsXLineToStepEvents(
    const XLineVisitor& line, const StepEvents* device_step_events) {
  StepEvents result;
  line.ForEachEvent([&](const XEventVisitor& event) {
    int64_t correlation_id = -1;
    int64_t group_id = -1;
    absl::string_view step_name;
    event.ForEachStat([&](const XStatVisitor& stat) {
      if (!stat.Type().has_value()) return;
      switch (stat.Type().value()) {
        case StatType::kCorrelationId:
          correlation_id = stat.IntValue();
          break;
        case StatType::kGroupId:
          group_id = stat.IntValue();
          break;
        case StatType::kStepName:
          step_name = stat.StrOrRefValue();
          break;
      }
    });
    if (group_id < 0) return;
    // Don't add CPU events when (1) it includes device step events and (2) it
    // doesn't have a device and that the group_id (i.e. step number) already
    // appears on the device. This will filter out all cpu events that do not
    // correspond to any steps executed on the device.
    bool has_device = (device_step_events != nullptr);
    if (has_device && !device_step_events->contains(group_id)) return;
    if (IsExplicitHostStepMarker(event.Name())) {
      result[group_id].AddMarker(
          StepMarker(StepMarkerType::kExplicitHostStepMarker, event.Name(),
                     event.GetTimespan()));
    } else if (!step_name.empty()) {
      // Grouping adds a step_name stat to implicit host step markers.
      result[group_id].AddMarker(
          StepMarker(StepMarkerType::kImplicitHostStepMarker, event.Name(),
                     event.GetTimespan()));
    } else if (IsRealCpuCompute(event.Name())) {
      result[group_id].AddEvent(EventTypeSpan(
          ClassifyCpuEvent(event.Name(), has_device, correlation_id >= 0),
          event.GetTimespan()));
    }
    if (!step_name.empty()) {
      result[group_id].SetStepName(std::string(step_name));
    }
  });
  return result;
}

StepEvents ConvertHostThreadsXPlaneToStepEvents(
    const XPlane& host_trace, const StepEvents* device_step_events) {
  StepEvents host_step_events;
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&host_trace);
  plane.ForEachLine([&](const XLineVisitor& line) {
    StepEvents thread_step_events =
        ConvertHostThreadsXLineToStepEvents(line, device_step_events);
    UnionCombineStepEvents(thread_step_events, &host_step_events);
  });
  return host_step_events;
}

StepEvents ConvertDeviceStepInfoToStepMarkers(const XLineVisitor& line) {
  StepEvents result;
  line.ForEachEvent([&](const XEventVisitor& event) {
    if (std::optional<XStatVisitor> stat = event.GetStat(StatType::kGroupId)) {
      result[stat->IntValue()].AddMarker(
          StepMarker(StepMarkerType::kDeviceStepMarker, event.Name(),
                     event.GetTimespan()));
    }
  });
  return result;
}

StepEvents ConvertDeviceTraceXLineToStepEvents(const uint64 device_id,
                                               const XLineVisitor& line) {
  StepEvents result;
  line.ForEachEvent([&](const XEventVisitor& event) {
    int64_t correlation_id = -1;
    int64_t group_id = -1;
    absl::string_view tensor_shapes;
    absl::string_view memcpy_details;
    event.ForEachStat([&](const XStatVisitor& stat) {
      if (!stat.Type().has_value()) return;
      switch (stat.Type().value()) {
        case StatType::kCorrelationId:
          correlation_id = stat.IntValue();
          break;
        case StatType::kGroupId:
          group_id = stat.IntValue();
          break;
        case StatType::kTensorShapes:
          tensor_shapes = stat.StrOrRefValue();
          break;
        case StatType::kMemcpyDetails:
          memcpy_details = stat.StrOrRefValue();
          break;
      }
    });

    if (correlation_id >= 0 && group_id >= 0) {
      EventType event_type = ClassifyGpuEvent(event.Name(), tensor_shapes);
      EventTypeSpan event_type_span(event_type, event.GetTimespan());
      result[group_id].AddEvent(event_type_span);
      switch (event_type) {
        case DEVICE_COLLECTIVES: {
          AllReduceInfo collective_ops;
          collective_ops.set_start_time_ps(event.TimestampPs());
          collective_ops.set_end_time_ps(event.EndOffsetPs());
          // TODO(jiesun): figure out how to get size info etc.
          result[group_id].AddCollectiveOpEvent(device_id, collective_ops);
          break;
        }
        case HOST_TO_DEVICE:
        case DEVICE_TO_DEVICE:
        case DEVICE_TO_HOST: {
          // TODO(jiesun): not all memcpy events are grouped, figure out a
          // better way to attribute them to steps.
          uint64 bytes_transferred =
              ParseNumBytesFromMemcpyDetail(memcpy_details);
          result[group_id].AddDeviceMemoryTransferEvent(
              event_type, event.GetTimespan(), bytes_transferred);
          break;
        }
        default:
          return;
      }
    }
  });
  return result;
}

StepEvents ConvertTpuDeviceTraceXLineToStepEvents(const uint64 device_id,
                                                  const XLineVisitor& line) {
  StepEvents result;
  absl::flat_hash_map</*group_id=*/int64_t, XEventsOpMetricsDbBuilder>
      op_metrics_builder;
  struct ParentRef {
    const XEventVisitor event;
    tsl::profiler::Timespan device_timespan;
    uint64_t children_duration_ps = 0;
    int64_t group_id = -1;
  };
  tsl::profiler::AncestorStack<ParentRef> event_stack(
      // Adds an OpMetric to the builder based on the provided parent reference.
      [&](const ParentRef& parent) {
        OpMetrics op_metrics = FromXEvent(parent.event);
        op_metrics.set_time_ps(parent.device_timespan.duration_ps());
        // TODO(b/397774568): Remove this once the SparseCore OpMetricsDb is
        // implemented.
        if (device_id < kSparseCoreIndexStart) {
          op_metrics.set_self_time_ps(op_metrics.time_ps() -
                                      parent.children_duration_ps);
        }
        op_metrics_builder[parent.group_id].AddOpMetric(
            op_metrics, GetOpKeyFromXEvent(parent.event));
      },
      // Checks if the child event is a child of the parent event.
      [](const ParentRef& parent, const ParentRef& child) {
        return parent.device_timespan.Includes(child.device_timespan);
      },
      // Adds the child duration to the parent.
      [](ParentRef& parent, ParentRef& child) {
        parent.children_duration_ps += child.device_timespan.duration_ps();
      });
  line.ForEachEvent([&](const XEventVisitor& event) {
    auto group_id_stat = event.GetStat(StatType::kGroupId);
    if (!group_id_stat.has_value()) return;
    int64_t group_id = group_id_stat->IntOrUintValue();
    event_stack.Push(ParentRef{
        .event = event,
        .device_timespan = tsl::profiler::GetDeviceEventTimespan(event),
        .group_id = group_id,
    });

    if (auto all_reduce_unique_id_stat =
            event.GetStat(StatType::kAllReduceUniqueId)) {
      result[group_id].AddCollectiveOpEvent(
          device_id,
          GetAllReduceInfo(event, all_reduce_unique_id_stat->IntOrUintValue()));
    }
  });
  event_stack.Flush();
  for (auto& [group_id, builder] : op_metrics_builder) {
    // Finalize Without the step time now.
    result[group_id].SetPerCoreOpMetricsDb(builder.Finalize(), device_id);
  }
  return result;
}

StepEvents ConvertDeviceTraceXPlaneToStepEvents(const XPlane& device_trace) {
  StepEvents device_step_events;
  XPlaneVisitor plane = tsl::profiler::CreateTfXPlaneVisitor(&device_trace);
  std::optional<int> tpu_core_id = tsl::profiler::GetTensorCoreId(plane.Name());
  std::optional<int> sc_core_id = tsl::profiler::GetSparseCoreId(plane.Name());
  plane.ForEachLine([&](const XLineVisitor& line) {
    int64_t line_id = line.Id();
    if (line_id == kThreadIdStepInfo ||
        (tpu_core_id.has_value() &&
         line.Name() == tsl::profiler::kStepLineName)) {
      // TODO(b/397774568): Re-add processing of SparseCore steps once the
      // SparseCore OpMetricsDb is implemented.
      StepEvents step_marker_events = ConvertDeviceStepInfoToStepMarkers(line);
      UnionCombineStepEvents(step_marker_events, &device_step_events);
    } else if (IsDerivedThreadId(line_id)) {
      return;
    } else {
      StepEvents stream_step_events;
      if (tpu_core_id.has_value()) {
        if (!tsl::profiler::IsOpLineName(line.Name())) return;
        // In TPU sampling mode, the profiling session could stop in the middle
        //  of a training step. In this case, the "XLA Ops" line will have
        // one more step than the "Step" line. We need to intersect them to get
        // the common step numbers.
        stream_step_events =
            ConvertTpuDeviceTraceXLineToStepEvents(plane.Id(), line);
        IntersectCombineStepEvents(stream_step_events, &device_step_events);
      } else if (sc_core_id.has_value()) {
        // TODO(b/397774568): Switch to IsOpLineName once SparseCore OpMetricsDb
        // is implemented.
        if (line.Name() != tsl::profiler::kSparseCoreStepLineName) return;
        stream_step_events = ConvertTpuDeviceTraceXLineToStepEvents(
            kSparseCoreIndexStart + plane.Id(), line);
        IntersectCombineStepEvents(stream_step_events, &device_step_events);
      } else {
        stream_step_events =
            ConvertDeviceTraceXLineToStepEvents(plane.Id(), line);
        UnionCombineStepEvents(stream_step_events, &device_step_events);
      }
    }
  });
  return device_step_events;
}

}  // namespace profiler
}  // namespace tensorflow
