/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/analytical_latency_estimator.h"

#include <memory>
#include <utility>

#include "absl/log/log.h"
#include "absl/time/time.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/model/gpu_collective_performance_model.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status.h"

namespace xla {
namespace gpu {
LatencyEstimator::TimeCost AnalyticalLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  const HloOpcode from_op = from.GetInstr().opcode();
  if (!config_.schedule_send_recvs &&
      (from_op == HloOpcode::kSend || from_op == HloOpcode::kRecv)) {
    return kLowLatency;
  }

  if (IsAsyncPair(from, target)) {
    double coll_time = absl::ToDoubleMicroseconds(
        GpuPerformanceWithCollectiveModel::ComputeCollectiveTime(
            from.GetInstr(), &*cost_analysis_, gpu_info_));
    VLOG(10) << "Analytical estimator calculated latency between "
             << from.GetInstr().name() << " and " << target.GetInstr().name()
             << " to be: " << coll_time << " us.";
    return coll_time;
  }
  return latency_estimator_->GetLatencyBetween(from, target);
}

LatencyEstimator::TimeCost AnalyticalLatencyEstimator::NodeCost(
    const HloInstruction* instr) const {
  if (hlo_query::IsAsyncCollectiveStartOp(instr, /*include_send_recv=*/true) ||
      hlo_query::IsAsyncCollectiveDoneOp(instr, /*include_send_recv=*/true)) {
    return kLowCost;
  }

  absl::Duration total_estimated_time =
      GpuPerformanceModel::EstimateRunTimeForInstruction(
          instr, gpu_info_, &*cost_analysis_,
          GpuPerformanceModelOptions::Default())
          .exec_time;
  LatencyEstimator::TimeCost cost_in_us =
      absl::ToDoubleMicroseconds(total_estimated_time);
  VLOG(10) << "Analytical estimator calculated cost for: " << instr->name()
           << ". Cost: " << cost_in_us;
  return cost_in_us;
}

AnalyticalLatencyEstimator::AnalyticalLatencyEstimator(
    const SchedulerConfig& config,
    std::unique_ptr<LatencyEstimator> latency_estimator,
    const se::DeviceDescription& gpu_info,
    HloCostAnalysis::ShapeSizeFunction shape_size_function,
    HloComputation* computation)
    : config_(config),
      gpu_info_(gpu_info),
      latency_estimator_(std::move(latency_estimator)),
      shape_size_function_(shape_size_function) {
  cost_analysis_.emplace(
      GpuHloCostAnalysis::Options{shape_size_function_,
                                  /*per_second_rates=*/{},
                                  /*min_latencies_seconds=*/{},
                                  /*count_multiple_input_accesses=*/true},
      gpu_info_);
  TF_CHECK_OK(computation->Accept(&cost_analysis_.value()));
}

}  // namespace gpu
}  // namespace xla
