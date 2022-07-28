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

#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/match.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/op_stats_combiner.h"
#include "tensorflow/core/profiler/convert/step_events_to_steps_db.h"
#include "tensorflow/core/profiler/convert/xplane_to_kernel_stats_db.h"
#include "tensorflow/core/profiler/convert/xplane_to_op_metrics_db.h"
#include "tensorflow/core/profiler/convert/xplane_to_step_events.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"
#include "tensorflow/core/profiler/protobuf/diagnostics.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/protobuf/tf_function.pb.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/device_caps_utils.h"
#include "tensorflow/core/profiler/utils/event_span.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/math_utils.h"
#include "tensorflow/core/profiler/utils/step_intersection.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/tpu_xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

std::string Hostname(const XSpace& space) {
  if (space.hostnames().empty()) return "localhost";
  DCHECK_EQ(space.hostnames_size(), 1);
  const std::string& hostname = space.hostnames(0);
  // This shouldn't be a taskname in host:port format.
  DCHECK(!absl::StrContains(hostname, ':'));
  return hostname;
}

}  // namespace

PerfEnv MakePerfEnv(double peak_tera_flops_per_second,
                    double peak_hbm_bw_giga_bytes_per_second) {
  PerfEnv result;
  result.set_peak_tera_flops_per_second(peak_tera_flops_per_second);
  result.set_peak_hbm_bw_giga_bytes_per_second(
      peak_hbm_bw_giga_bytes_per_second);
  result.set_ridge_point(TeraToGiga(peak_tera_flops_per_second) /
                         peak_hbm_bw_giga_bytes_per_second);
  return result;
}

PerfEnv GetPerfEnvFromXPlane(const XPlane& device_plane) {
  DeviceCapabilities cap = GetDeviceCaps(device_plane);
  if (!absl::StartsWith(device_plane.name(), kTpuPlanePrefix)) {
    return MakePerfEnv(
        GigaToTera(GetFlopMaxThroughputPerSM(cap)) * cap.num_cores(),
        UniToGiga(cap.memory_bandwidth()));
  } else {
    XPlaneVisitor visitor = CreateTfXPlaneVisitor(&device_plane);
    auto peak_tera_flops_per_second =
        visitor.GetStat(StatType::kDevCapPeakTeraflopsPerSecond);
    auto peak_hbm_bw_giga_bytes_per_second =
        visitor.GetStat(StatType::kDevCapPeakHbmBwGigabytesPerSecond);
    return MakePerfEnv(peak_tera_flops_per_second->DoubleValue(),
                       peak_hbm_bw_giga_bytes_per_second->DoubleValue());
  }
}

void SetRunEnvironment(const XSpace& space, RunEnvironment* env) {
  // Currently, we only support profiling one host and one program.
  env->set_host_count(1);
  env->set_task_count(1);
  env->mutable_hostnames()->insert({Hostname(space), true});

  std::vector<const XPlane*> gpu_planes =
      FindPlanesWithPrefix(space, kGpuPlanePrefix);
  if (!gpu_planes.empty()) {
    absl::string_view gpu_model =
        GpuModelName(GetDeviceCaps(*gpu_planes.front()));
    if (!gpu_model.empty()) {
      env->set_device_type(std::string(gpu_model));
    } else {
      env->set_device_type("GPU");
    }
    env->set_device_core_count(gpu_planes.size());
  } else if (std::vector<const XPlane*> tpu_planes =
                 FindTensorCorePlanes(space);
             !tpu_planes.empty()) {
    XPlaneVisitor visitor = CreateTfXPlaneVisitor(tpu_planes.at(0));
    auto xstat = visitor.GetStat(StatType::kDeviceTypeString);
    if (xstat.has_value()) {
      env->set_device_type(std::string(xstat->StrOrRefValue()));
    }
    env->set_device_core_count(tpu_planes.size());
  } else {
    env->set_device_type("CPU");
    env->set_device_core_count(0);
  }
}

void PropagateXSpaceDiagnosticsToOpStats(const XSpace& space,
                                         OpStats* op_stats) {
  if (!space.errors().empty()) {
    absl::flat_hash_set<std::string> unique_errors;
    unique_errors.insert(space.errors().begin(), space.errors().end());
    *op_stats->mutable_diagnostics()->mutable_errors() = {unique_errors.begin(),
                                                          unique_errors.end()};
  }
  if (!space.warnings().empty()) {
    absl::flat_hash_set<std::string> unique_warnings;
    unique_warnings.insert(space.warnings().begin(), space.warnings().end());
    *op_stats->mutable_diagnostics()->mutable_warnings() = {
        unique_warnings.begin(), unique_warnings.end()};
  }
}

OpStats ConvertXSpaceToOpStats(const XSpace& space,
                               const OpStatsOptions& options) {
  const XPlane* host_plane = FindPlaneWithName(space, kHostThreadsPlaneName);
  std::vector<const XPlane*> device_planes = FindTensorCorePlanes(space);
  bool is_gpu = device_planes.empty();
  if (is_gpu) {
    device_planes = FindPlanesWithPrefix(space, kGpuPlanePrefix);
  }

  OpStats op_stats;
  StepEvents step_events;
  PropagateXSpaceDiagnosticsToOpStats(space, &op_stats);
  // Convert device planes.
  OpMetricsDbCombiner op_metrics_db_combiner(
      op_stats.mutable_device_op_metrics_db());
  SetRunEnvironment(space, op_stats.mutable_run_environment());

  KernelReportMap reports;

  // TODO(b/161942993) parallelize XPlane processing per thread.
  for (const XPlane* device_trace : device_planes) {
    if (options.generate_op_metrics_db) {
      if (!op_stats.has_perf_env()) {
        *op_stats.mutable_perf_env() = GetPerfEnvFromXPlane(*device_trace);
      }
      OpMetricsDb device_op_metrics_db =
          ConvertDeviceTraceXPlaneToOpMetricsDb(*device_trace);
      op_metrics_db_combiner.Combine(device_op_metrics_db);
    }
    if (options.generate_step_db) {
      StepEvents device_step_events =
          ConvertDeviceTraceXPlaneToStepEvents(*device_trace);
      CombineStepEvents(device_step_events, &step_events);
    }
    if (options.generate_kernel_stats_db) {
      ConvertDeviceTraceXPlaneToKernelReports(*device_trace,
                                              /*on_kernel_fn=*/{}, &reports);
    }
  }

  // Combine into reports.
  if (options.generate_kernel_stats_db) {
    CopyTopKDurationKernelReportsToDb(reports,
                                      op_stats.mutable_kernel_stats_db());
  }

  bool has_device = !device_planes.empty();
  // Convert a host plane.
  if (host_plane) {
    if (options.generate_op_metrics_db) {
      *op_stats.mutable_host_op_metrics_db() =
          ConvertHostThreadsXPlaneToOpMetricsDb(*host_plane);
    }
    if (options.generate_step_db) {
      const StepEvents* device_step_events =
          has_device ? &step_events : nullptr;
      StepEvents host_step_events =
          ConvertHostThreadsXPlaneToStepEvents(*host_plane, device_step_events);
      CombineStepEvents(host_step_events, &step_events);
    }
  }
  if (options.generate_step_db) {
    StepEvents nonoverlapped_step_events =
        ToNonOverlappedStepEvents(step_events);
    *op_stats.mutable_step_db() = ConvertStepEventsToStepDb(
        has_device, options.maybe_drop_incomplete_steps,
        nonoverlapped_step_events);
    *op_stats.mutable_device_op_metrics_db()->mutable_precision_stats() =
        ComputePrecisionStats(nonoverlapped_step_events);
  }

  // TODO(bvandermoon): Add the TPU equivalent for setting core details hostname
  if (is_gpu) {
    CoreDetails& details =
        (*op_stats.mutable_core_id_to_details())[kDefaultGpuLocalCoreId];
    details.set_hostname(Hostname(space));
  }
  return op_stats;
}

Status ConvertMultiXSpacesToCombinedOpStats(const std::vector<XSpace>& xspaces,
                                            const OpStatsOptions& options,
                                            OpStats* combined_op_stats) {
  // A shortcut code path for a single XSpace. There is no need to merge OpStats
  // if there is only a single XSpace.
  if (xspaces.size() == 1) {
    *combined_op_stats = ConvertXSpaceToOpStats(xspaces[0], options);
    return OkStatus();
  }

  // Read multiple XSpaces and convert to multiple OpStats.
  std::vector<OpStats> all_op_stats;
  all_op_stats.reserve(xspaces.size());
  for (const XSpace& xspace : xspaces) {
    all_op_stats.push_back(ConvertXSpaceToOpStats(xspace, options));
  }

  // Combine OpStats.
  std::vector<OpStatsInfo> all_op_stats_info;
  all_op_stats_info.reserve(all_op_stats.size());
  for (int i = 0; i < all_op_stats.size(); i++) {
    all_op_stats_info.emplace_back(
        &all_op_stats[i],
        ParseHardwareType(all_op_stats[i].run_environment().device_type()), i);
  }

  // Do not limit the maximum number of steps during the merge of OpStats.
  StepIntersection step_intersection =
      ComputeStepIntersectionToMergeOpStats(all_op_stats_info, kuint32max);
  CombineAllOpStats(all_op_stats_info, step_intersection, combined_op_stats);

  return OkStatus();
}

}  // namespace profiler
}  // namespace tensorflow
