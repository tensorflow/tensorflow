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

#include "tensorflow/core/profiler/convert/op_stats_combiner.h"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"
#include "tensorflow/core/profiler/protobuf/diagnostics.pb.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/power_metrics.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/protobuf/topology.pb.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tensorflow/core/profiler/utils/kernel_stats_utils.h"
#include "tensorflow/core/profiler/utils/step_intersection.h"

namespace tensorflow {
namespace profiler {

namespace {

// Combines the src PerCoreStepInfo into the dst PerCoreStepInfo.
void CombinePerCoreStepInfo(
    int src_host_id, const PerCoreStepInfo& src, bool use_incomplete_step,
    PerCoreStepInfo* dst,
    OpMetricsDbCombiner* hlo_metrics_db_complete_steps_only_combiner,
    OpMetricsDbCombiner* hlo_metrics_db_per_step_combiner) {
  CombineCoreIdMap(src_host_id, src.step_info_per_core(),
                   dst->mutable_step_info_per_core());

  // Since we have assigned a new step number to the combined result, update
  // the step number on each core to this new step number.
  uint32 new_step_num = dst->step_num();
  for (auto& percore_stepinfo : *dst->mutable_step_info_per_core()) {
    auto& stepinfo = percore_stepinfo.second;
    stepinfo.set_step_num(new_step_num);
  }

  if (!use_incomplete_step) {
    hlo_metrics_db_complete_steps_only_combiner->Combine(src.hlo_metrics_db());
  }
  hlo_metrics_db_per_step_combiner->Combine(src.hlo_metrics_db());
  CombineCoreIdMap(src_host_id, src.all_reduce_db_per_core(),
                   dst->mutable_all_reduce_db_per_core());
  CombineCoreIdMap(src_host_id, src.core_id_to_replica_id_map(),
                   dst->mutable_core_id_to_replica_id_map());
}

void CombineStepDatabase(
    int src_host_id, const StepIntersection& step_intersection,
    const StepDatabaseResult& src, StepDatabaseResult* dst,
    OpMetricsDbCombiner* hlo_metrics_db_complete_steps_only_combiner,
    std::vector<OpMetricsDbCombiner>* hlo_metrics_db_per_step_combiners) {
  if (src.use_incomplete_step()) dst->set_use_incomplete_step(true);
  uint32 src_first_step_idx = step_intersection.FirstStepIndex(src_host_id);
  for (uint32 i = 0; i < step_intersection.NumSteps(); i++) {
    CombinePerCoreStepInfo(
        src_host_id, src.step_sequence(src_first_step_idx + i),
        src.use_incomplete_step(), dst->mutable_step_sequence(i),
        hlo_metrics_db_complete_steps_only_combiner,
        &(*hlo_metrics_db_per_step_combiners)[i]);
  }
}

void CombinePowerMetrics(const RunEnvironment& src, RunEnvironment* dst) {
  const size_t src_hosts = src.hostnames_size();
  const size_t dst_hosts = dst->hostnames_size();
  const double src_weight = src_hosts * 1.0 / (src_hosts + dst_hosts);
  const double dst_weight = dst_hosts * 1.0 / (src_hosts + dst_hosts);
  // Always assume src/dst have the same number of power components.
  for (const auto& src_metric : src.power_metrics().power_component_metrics()) {
    for (auto& dst_metric :
         *dst->mutable_power_metrics()->mutable_power_component_metrics()) {
      if (src_metric.component_name() != dst_metric.component_name()) continue;
      dst_metric.set_max_power(
          std::max(src_metric.max_power(), dst_metric.max_power()));
      dst_metric.set_avg_power(src_metric.avg_power() * src_weight +
                               dst_metric.avg_power() * dst_weight);
    }
  }
}

void CombineRunEnvironment(const RunEnvironment& src, RunEnvironment* dst) {
  dst->mutable_hostnames()->insert(src.hostnames().begin(),
                                   src.hostnames().end());
  dst->set_host_count(dst->hostnames_size());
  // Ignore CPU and Unknown Device type for device type selection if the
  // destination does not have a device type already.
  if (src.device_type() != "CPU" && src.device_type() != "Device") {
    dst->set_device_type(src.device_type());
    dst->set_device_core_count(src.device_core_count() +
                               dst->device_core_count());
    // Replica count and num cores per replica must be same for all copies.
    dst->set_replica_count(std::max(src.replica_count(), dst->replica_count()));
    dst->set_num_cores_per_replica(
        std::max(src.num_cores_per_replica(), dst->num_cores_per_replica()));
    *dst->mutable_system_topology() = src.system_topology();
  } else if (dst->device_type().empty()) {
    dst->set_device_type(src.device_type());
  }
  if (src.hardware_type() != dst->hardware_type()) {
    // Select the highest hardware type as TPU/GPU should override CPU_ONLY
    // (e.g. coordinator).
    dst->set_hardware_type(std::max(src.hardware_type(), dst->hardware_type()));
  }
  dst->set_task_count(src.task_count() + dst->task_count());
  // Only overwrite the dst if profile_duration_ms in dst is not defined or
  // is zero and profile_duration_ms in src is greater than zero.
  if (src.host_independent_job_info().profile_duration_ms() > 0) {
    (*dst->mutable_host_independent_job_info()) =
        src.host_independent_job_info();
  }
  for (const auto& job_info : src.host_dependent_job_info()) {
    *(dst->add_host_dependent_job_info()) = job_info;
  }
  dst->set_host_trace_level(src.host_trace_level());
  dst->set_is_training(src.is_training());
  CombinePowerMetrics(src, dst);
}

// Combines the src PerfEnv into the dst PerfEnv.
void CombinePerfEnv(const PerfEnv& src, PerfEnv* dst) {
  if (src.peak_tera_flops_per_second() > 0) {
    dst->set_peak_tera_flops_per_second(src.peak_tera_flops_per_second());
  }

  if (src.peak_bws_giga_bytes_per_second_size() > 0 &&
      dst->peak_bws_giga_bytes_per_second_size() == 0) {
    *dst->mutable_peak_bws_giga_bytes_per_second() =
        src.peak_bws_giga_bytes_per_second();
  }
  if (src.ridge_point() > 0) {
    dst->set_ridge_point(src.ridge_point());
  }
}

// Combines the src Diagnostics into the dst Diagnostics.
void CombineDiagnostics(const Diagnostics& src, Diagnostics* dst) {
  dst->mutable_info()->MergeFrom(src.info());
  dst->mutable_warnings()->MergeFrom(src.warnings());
  dst->mutable_errors()->MergeFrom(src.errors());
}

// Combine the src OpStats into the dst OpStats.
void CombineOpStats(
    bool no_accelerator_in_system, int src_host_id, HardwareType hardware_type,
    const StepIntersection& step_intersection, const OpStats& src, OpStats* dst,
    OpMetricsDbCombiner* host_op_metrics_db_combiner,
    OpMetricsDbCombiner* device_op_metrics_db_combiner,
    OpMetricsDbCombiner* hlo_metrics_db_complete_steps_only_combiner,
    std::vector<OpMetricsDbCombiner>* hlo_metrics_db_per_step_combiners) {
  // Combine host_metrics_db.
  // Host OpMetricsDb does not need to update the number of cores a certain op
  // occurs.
  host_op_metrics_db_combiner->Combine(src.host_op_metrics_db(),
                                       /*update_num_cores=*/false);
  // Combine device_metrics_db.
  device_op_metrics_db_combiner->Combine(src.device_op_metrics_db());

  // Combine step_db.
  if (!IsCoordinator(no_accelerator_in_system, hardware_type)) {
    CombineStepDatabase(src_host_id, step_intersection, src.step_db(),
                        dst->mutable_step_db(),
                        hlo_metrics_db_complete_steps_only_combiner,
                        hlo_metrics_db_per_step_combiners);
  }

  // Combine run environment info.
  CombineRunEnvironment(src.run_environment(), dst->mutable_run_environment());

  // Combine the perf environment info.
  CombinePerfEnv(src.perf_env(), dst->mutable_perf_env());

  // Combine diagnostics.
  CombineDiagnostics(src.diagnostics(), dst->mutable_diagnostics());

  // Combine kernel stats.
  dst->mutable_kernel_stats_db()->mutable_reports()->MergeFrom(
      src.kernel_stats_db().reports());

  // Combine tf-function stats.
  CombineTfFunctionDb(src.tf_function_db(), dst->mutable_tf_function_db());

  // Combine the mapping from core ID to details.
  CombineCoreIdMap(src_host_id, src.core_id_to_details(),
                   dst->mutable_core_id_to_details());

  // Combine performance counter result.
  dst->mutable_performance_counter_result()
      ->set_matrix_unit_utilization_percent(
          dst->performance_counter_result().matrix_unit_utilization_percent() +
          src.performance_counter_result().matrix_unit_utilization_percent());
}

}  // namespace

bool IsCoordinator(bool no_accelerator_in_system, HardwareType hardware_type) {
  // A host is a coordinator if:
  //   (1) The host doesn't have a device, and
  //   (2) The system does use accelerator (if not, it uses CPU only and so this
  //   host should be regarded as a worker as well).
  return !HasDevice(hardware_type) && !no_accelerator_in_system;
}

bool NoAcceleratorInSystem(const std::vector<OpStatsInfo>& all_op_stats_info) {
  for (const auto& op_stats_info : all_op_stats_info) {
    if (HasDevice(op_stats_info.hardware_type)) {
      return false;
    }
  }
  return true;
}

uint32 GlobalCoreId(int host_id, uint32 device_ordinal) {
  constexpr uint32 kMaxDevicesPerHost = 1000;  // power-of-10 for debuggability
  return host_id * kMaxDevicesPerHost + device_ordinal;
}

StepIntersection ComputeStepIntersectionToMergeOpStats(
    const std::vector<OpStatsInfo>& all_op_stats_info,
    uint32 max_step_per_host) {
  bool no_accelerator_in_system = NoAcceleratorInSystem(all_op_stats_info);

  absl::flat_hash_map<uint32, const StepDatabaseResult*> per_host_step_db;
  for (const auto& op_stats_info : all_op_stats_info) {
    if (IsCoordinator(no_accelerator_in_system, op_stats_info.hardware_type))
      continue;
    // Includes only workers in per_host_step_db.
    per_host_step_db[op_stats_info.src_host_id] =
        &op_stats_info.op_stats->step_db();
  }

  return StepIntersection(max_step_per_host, per_host_step_db);
}

void CombineAllOpStats(const std::vector<OpStatsInfo>& all_op_stats_info,
                       const StepIntersection& step_intersection,
                       OpStats* combined_op_stats) {
  // A shortcut code path for a single OpStats. There is no need to merge.
  if (all_op_stats_info.size() == 1) {
    *combined_op_stats = *all_op_stats_info[0].op_stats;
    return;
  }

  StepDatabaseResult* combined_step_db = combined_op_stats->mutable_step_db();
  // Initialize the StepDatabaseResult field that depends on the number of
  // steps.
  for (uint32 dst_step_num : step_intersection.DstStepNumbers()) {
    combined_step_db->add_step_sequence()->set_step_num(dst_step_num);
  }
  // Record the number of steps that are dropped.
  combined_step_db->set_num_steps_dropped(step_intersection.StepsDropped());

  combined_step_db->set_empty_intersect(step_intersection.EmptyIntersect());

  // Initialize all the OpMetricsDbCombiners.
  OpMetricsDbCombiner host_op_metrics_db_combiner(
      combined_op_stats->mutable_host_op_metrics_db());
  OpMetricsDbCombiner device_op_metrics_db_combiner(
      combined_op_stats->mutable_device_op_metrics_db());
  OpMetricsDbCombiner hlo_metrics_db_complete_steps_only_combiner(
      combined_op_stats->mutable_hlo_metrics_db_complete_steps_only());
  std::vector<OpMetricsDbCombiner> hlo_metrics_db_per_step_combiners;
  hlo_metrics_db_per_step_combiners.reserve(
      combined_step_db->step_sequence_size());
  for (PerCoreStepInfo& step_info :
       *combined_step_db->mutable_step_sequence()) {
    hlo_metrics_db_per_step_combiners.emplace_back(
        step_info.mutable_hlo_metrics_db());
  }

  bool no_accelerator_in_system = NoAcceleratorInSystem(all_op_stats_info);

  for (const auto& op_stats_info : all_op_stats_info) {
    CombineOpStats(no_accelerator_in_system, op_stats_info.src_host_id,
                   op_stats_info.hardware_type, step_intersection,
                   *op_stats_info.op_stats, combined_op_stats,
                   &host_op_metrics_db_combiner, &device_op_metrics_db_combiner,
                   &hlo_metrics_db_complete_steps_only_combiner,
                   &hlo_metrics_db_per_step_combiners);
  }

  // Sorts all the kernel reports that have been merged by CombineTfOpStats and
  // keeps only the top kernel reports with long kernel duration.
  SortAndKeepTopKDurationKernelReportsInDb(
      combined_op_stats->mutable_kernel_stats_db());

  // Process performance counter results.
  combined_op_stats->mutable_performance_counter_result()
      ->set_matrix_unit_utilization_percent(
          combined_op_stats->performance_counter_result()
              .matrix_unit_utilization_percent() /
          all_op_stats_info.size());
}

}  // namespace profiler
}  // namespace tensorflow
