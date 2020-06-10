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

#include "tensorflow/core/profiler/convert/xplane_to_op_stats.h"

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/step_events_to_steps_db.h"
#include "tensorflow/core/profiler/convert/xplane_to_kernel_stats_db.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_metrics_db.h"
#include "tensorflow/core/profiler/convert/xplane_to_step_events.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_function.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

DeviceCapabilities GetDeviceCapFromXPlane(const XPlane& device_plane) {
  DeviceCapabilities cap;
  XPlaneVisitor plane = CreateTfXPlaneVisitor(&device_plane);
  plane.ForEachStat([&cap](const XStatVisitor& stat) {
    if (!stat.Type().has_value()) return;
    switch (stat.Type().value()) {
      case kDevCapClockRateKHz:
        cap.set_clock_rate_in_ghz(stat.IntValue() / 1000000.0);
        break;
      case kDevCapCoreCount:
        cap.set_num_cores(stat.IntValue());
        break;
      case kDevCapMemoryBandwidth:
        cap.set_memory_bandwidth(stat.IntValue());  // bytes/s
        break;
      case kDevCapMemorySize:
        cap.set_memory_size_in_bytes(stat.UintValue());
        break;
      case kDevCapComputeCapMajor:
        cap.mutable_compute_capability()->set_major(stat.IntValue());
        break;
      case kDevCapComputeCapMinor:
        cap.mutable_compute_capability()->set_minor(stat.IntValue());
        break;
    }
  });
  return cap;
}

PerfEnv GetPerfEnvFromXPlane(const XPlane& device_plane) {
  PerfEnv result;
  DeviceCapabilities cap = GetDeviceCapFromXPlane(device_plane);
  result.set_peak_tera_flops_per_second(GetFlopMaxThroughputPerSM(cap) / 1000 *
                                        cap.num_cores());
  result.set_peak_hbm_bw_giga_bytes_per_second(cap.memory_bandwidth() / 1e9);
  result.set_ridge_point(result.peak_tera_flops_per_second() * 1000 /
                         result.peak_hbm_bw_giga_bytes_per_second());
  return result;
}

void SetRunEnvironment(int32 accelerator_count, RunEnvironment* env) {
  // Currently, we only support profiling one host and one program.
  env->set_host_count(1);
  env->set_task_count(1);
  env->set_device_type(accelerator_count > 0 ? "GPU" : "CPU");
  env->set_device_core_count(accelerator_count);
}

void ProcessHostPlane(const XPlane* host_plane, bool use_device_step_events,
                      OpMetricsDb* op_metrics_db, StepEvents* step_events,
                      TfFunctionDb* tf_function_db) {
  absl::flat_hash_map<int64, TfOp> tf_ops =
      CollectTfOpsFromHostThreadsXPlane(*host_plane);
  OpMetricsDbCombiner combiner(op_metrics_db);
  XPlaneVisitor plane = CreateTfXPlaneVisitor(host_plane);
  plane.ForEachLine([&](const XLineVisitor& line) {
    ConsumeTfMetricsDbData(
        ConvertHostThreadsXLineToTfMetricsDbData(line, tf_ops), &combiner);
    CombineStepEvents(ConvertHostThreadsXLineToStepEvents(
                          line, use_device_step_events, *step_events),
                      step_events);
    CombineTfFunctionDb(ConvertHostThreadsXLineToTfFunctionDb(line),
                        tf_function_db);
  });
}

}  // namespace

void PropagateXSpaceErrorsToOpStats(const XSpace& space, OpStats* op_stats) {
  if (space.errors().empty()) return;
  absl::flat_hash_set<std::string> unique_errors;
  unique_errors.insert(space.errors().begin(), space.errors().end());
  *op_stats->mutable_errors() = {unique_errors.begin(), unique_errors.end()};
}

OpStats ConvertXSpaceToOpStats(const XSpace& space) {
  const XPlane* host_plane = FindPlaneWithName(space, kHostThreads);
  std::vector<const XPlane*> device_planes =
      FindPlanesWithPrefix(space, kGpuPlanePrefix);
  OpStats op_stats;
  StepEvents step_events;
  PropagateXSpaceErrorsToOpStats(space, &op_stats);
  // Convert device planes.
  OpMetricsDbCombiner op_metrics_db_combiner(
      op_stats.mutable_device_op_metrics_db());
  SetRunEnvironment(device_planes.size(), op_stats.mutable_run_environment());

  std::vector<KernelReport> reports;
  for (const XPlane* device_trace : device_planes) {
    if (!op_stats.has_perf_env()) {
      *op_stats.mutable_perf_env() = GetPerfEnvFromXPlane(*device_trace);
    }
    const PerfEnv& perf_env = op_stats.perf_env();
    OpMetricsDb device_op_metrics_db = ConvertDeviceTraceXPlaneToOpMetricsDb(
        *device_trace, perf_env.peak_tera_flops_per_second(),
        perf_env.peak_hbm_bw_giga_bytes_per_second());
    op_metrics_db_combiner.Combine(device_op_metrics_db);
    CombineStepEvents(ConvertDeviceTraceXPlaneToStepEvents(*device_trace),
                      &step_events);
    KernelStatsDb kernel_stats_db = ConvertDeviceTraceXPlaneToKernelStatsDb(
        *device_trace, /*on_kernel_fn=*/{});
    reports.insert(reports.begin(), kernel_stats_db.reports().begin(),
                   kernel_stats_db.reports().end());
  }
  GroupKernelReports(&reports, op_stats.mutable_kernel_stats_db());
  SortKernelsByTotalDurationDesc(op_stats.mutable_kernel_stats_db());
  // Convert a host plane.
  bool has_device = !device_planes.empty();
  if (host_plane) {
    ProcessHostPlane(host_plane, has_device,
                     op_stats.mutable_host_op_metrics_db(), &step_events,
                     op_stats.mutable_tf_function_db());
  }
  StepEvents nonoverlapped_step_events = ToNonOverlappedStepEvents(step_events);
  *op_stats.mutable_step_db() =
      ConvertStepEventsToStepDb(has_device, nonoverlapped_step_events);
  *op_stats.mutable_device_op_metrics_db()->mutable_precision_stats() =
      ComputePrecisionStats(nonoverlapped_step_events);
  return op_stats;
}

}  // namespace profiler
}  // namespace tensorflow
