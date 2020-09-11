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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/profiler/convert/op_metrics_db_combiner.h"
#include "tensorflow/core/profiler/convert/xplane_to_tf_functions.h"
#include "tensorflow/core/profiler/protobuf/diagnostics.pb.h"
#include "tensorflow/core/profiler/protobuf/hardware_types.pb.h"
#include "tensorflow/core/profiler/protobuf/kernel_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/op_stats.pb.h"
#include "tensorflow/core/profiler/protobuf/steps_db.pb.h"
#include "tensorflow/core/profiler/utils/hardware_type_utils.h"
#include "tensorflow/core/profiler/utils/step_interval.h"

namespace tensorflow {
namespace profiler {

namespace {

// Combines the src PerCoreStepInfo into the dst PerCoreStepInfo.
void CombinePerCoreStepInfo(
    int src_host_id, bool use_incomplete_step, const PerCoreStepInfo& src,
    PerCoreStepInfo* dst,
    OpMetricsDbCombiner* hlo_metrics_db_complete_steps_only_combiner,
    OpMetricsDbCombiner* hlo_metrics_db_per_step_combiner) {
  DCHECK_EQ(dst->step_num(), src.step_num());
  CombineCoreIdMap(src_host_id, src.step_info_per_core(),
                   dst->mutable_step_info_per_core());
  if (!use_incomplete_step) {
    hlo_metrics_db_complete_steps_only_combiner->Combine(src.hlo_metrics_db());
  }
  hlo_metrics_db_per_step_combiner->Combine(src.hlo_metrics_db());
  CombineCoreIdMap(src_host_id, src.flow_db_per_core(),
                   dst->mutable_flow_db_per_core());
  CombineCoreIdMap(src_host_id, src.all_reduce_db_per_core(),
                   dst->mutable_all_reduce_db_per_core());
  CombineCoreIdMap(src_host_id, src.core_id_to_replica_id_map(),
                   dst->mutable_core_id_to_replica_id_map());
}

void CombineStepDatabase(
    int src_host_id, StepInterval step_intersection,
    const StepDatabaseResult& src, StepDatabaseResult* dst,
    OpMetricsDbCombiner* hlo_metrics_db_complete_steps_only_combiner,
    std::vector<OpMetricsDbCombiner>* hlo_metrics_db_per_step_combiners) {
  if (src.use_incomplete_step()) {
    dst->set_use_incomplete_step(true);
  }
  for (const PerCoreStepInfo& src_step_info : src.step_sequence()) {
    uint32 step_num = src_step_info.step_num();
    if (!step_intersection.Contains(step_num)) {
      continue;
    }
    uint32 dst_step_sequence_index = step_intersection.Index(step_num);
    CombinePerCoreStepInfo(
        src_host_id, src.use_incomplete_step(), src_step_info,
        dst->mutable_step_sequence(dst_step_sequence_index),
        hlo_metrics_db_complete_steps_only_combiner,
        &(*hlo_metrics_db_per_step_combiners)[dst_step_sequence_index]);
  }
}

void CombineRunEnvironment(const RunEnvironment& src, RunEnvironment* dst) {
  dst->mutable_hostnames()->insert(src.hostnames().begin(),
                                   src.hostnames().end());
  dst->set_host_count(dst->hostnames_size());
  if (src.device_type() != "CPU") {
    dst->set_device_type(src.device_type());
    // TODO(b/111402648): Batch size may differ per-core. Currently, we report
    // the max batch size. We need to come up with a better measure.
    dst->set_per_core_batch_size(
        std::max(src.per_core_batch_size(), dst->per_core_batch_size()));
    dst->set_device_core_count(src.device_core_count() +
                               dst->device_core_count());
    // Replica count and num cores per replica must be same for all copies.
    dst->set_replica_count(std::max(src.replica_count(), dst->replica_count()));
    dst->set_num_cores_per_replica(
        std::max(src.num_cores_per_replica(), dst->num_cores_per_replica()));
    *dst->mutable_topology() = src.topology();
  }
  dst->set_task_count(src.task_count() + dst->task_count());
  (*dst->mutable_host_independent_job_info()) = src.host_independent_job_info();
  for (const auto& job_info : src.host_dependent_job_info()) {
    *(dst->add_host_dependent_job_info()) = job_info;
  }
  dst->set_host_trace_level(src.host_trace_level());
}

// Combines the src PerfEnv into the dst PerfEnv.
void CombinePerfEnv(const PerfEnv& src, PerfEnv* dst) {
  dst->set_peak_tera_flops_per_second(src.peak_tera_flops_per_second());
  dst->set_peak_hbm_bw_giga_bytes_per_second(
      src.peak_hbm_bw_giga_bytes_per_second());
  dst->set_ridge_point(src.ridge_point());
}

// Combines the src Diagnostics into the dst Diagnostics.
void CombineDiagnostics(const Diagnostics& src, Diagnostics* dst) {
  dst->mutable_info()->MergeFrom(src.info());
  dst->mutable_warnings()->MergeFrom(src.warnings());
  dst->mutable_errors()->MergeFrom(src.errors());
}

}  // namespace

bool IsCoordinator(bool no_accelerator_in_system, HardwareType hardware_type) {
  // A host is a coordinator if:
  //   (1) The host doesn't have a device, and
  //   (2) The system does use accelerator (if not, it uses CPU only and so this
  //   host should be regarded as a worker as well).
  return !HasDevice(hardware_type) && !no_accelerator_in_system;
}

uint32 GlobalCoreId(int host_id, uint32 device_ordinal) {
  constexpr uint32 kMaxDevicesPerHost = 1000;  // power-of-10 for debuggability
  return host_id * kMaxDevicesPerHost + device_ordinal;
}

void CombineOpStats(
    bool no_accelerator_in_system, int src_host_id, HardwareType hardware_type,
    StepInterval step_intersection, const OpStats& src, OpStats* dst,
    OpMetricsDbCombiner* host_op_metrics_db_combiner,
    OpMetricsDbCombiner* device_op_metrics_db_combiner,
    OpMetricsDbCombiner* hlo_metrics_db_complete_steps_only_combiner,
    std::vector<OpMetricsDbCombiner>* hlo_metrics_db_per_step_combiners) {
  // Combine host_metrics_db.
  host_op_metrics_db_combiner->Combine(src.host_op_metrics_db());
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
}

}  // namespace profiler
}  // namespace tensorflow
