/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/model/sol_latency_estimator.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/time/time.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/sol_gpu_cost_model.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/util.h"
#include "tsl/platform/status.h"

namespace xla {
namespace gpu {

namespace {

int64_t ComputeMessageSize(const HloInstruction& instr,
                           HloCostAnalysis::ShapeSizeFunction fun) {
  int64_t msg_size = 0;
  ShapeUtil::ForEachSubshape(
      instr.shape(),
      [&msg_size, &fun](const Shape& subshape, const ShapeIndex&) {
        if (subshape.IsArray()) {
          msg_size += fun(subshape);
        }
      });
  return msg_size;
}

int GetNumGpus(const HloInstruction& instr) {
  const HloInstruction* i = &instr;
  if (instr.opcode() == HloOpcode::kAsyncStart) {
    i = instr.async_wrapped_instruction();
  }
  int size = 0;
  for (auto& rg : i->replica_groups()) {
    size += rg.replica_ids_size();
  }
  return size;
}

/*static*/ absl::Duration ComputeCollectiveTime(
    const HloInstruction& instr, const se::DeviceDescription& gpu_device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    const SolGPUCostModel::Config& sol_flags) {
  const int num_nodes = GetNumGpus(instr) / sol_flags.gpus_per_node;
  if (num_nodes == 1) {
    VLOG(8) << "Returning only kernel launch overhead for a single node.";
    return GpuPerformanceModelBase::kNcclKernelLaunchOverhead;
  }

  if (HloDataflowAnalysis::IsAsynchronousOperationDone(instr.opcode())) {
    VLOG(8) << "Returning 0 cost for async done op " << instr.name();
    return absl::ZeroDuration();
  }
  SolGPUCostModel sol_model(sol_flags);
  const int64_t msg_size = ComputeMessageSize(instr, shape_size_fn);

  switch (instr.opcode()) {
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart: {
      return sol_model.RingLatency(msg_size, num_nodes,
                                   SolGPUCostModel::CollectiveType::kAllGather);
    }
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart: {
      return sol_model.RingLatency(msg_size, num_nodes,
                                   SolGPUCostModel::CollectiveType::kAllReduce);
    }
    case HloOpcode::kReduceScatter: {
      return sol_model.RingLatency(
          msg_size, num_nodes, SolGPUCostModel::CollectiveType::kReduceScatter);
    }
    case HloOpcode::kAsyncStart: {
      if (instr.async_wrapped_opcode() == HloOpcode::kReduceScatter) {
        return sol_model.RingLatency(
            msg_size, num_nodes,
            SolGPUCostModel::CollectiveType::kReduceScatter);
      }
      break;
    }
    case HloOpcode::kRecv:
    case HloOpcode::kSend: {
      return sol_model.RingLatency(msg_size, num_nodes,
                                   SolGPUCostModel::CollectiveType::kSendRecv);
    }
    // note: AllToAll is not yet supported in XLA
    default: {
      LOG(WARNING)
          << "[SoL] Runtime estimate for " << instr.name()
          << " not implemented. Returning only the kernel launch time.";
      return GpuPerformanceModelBase::kNcclKernelLaunchOverhead;
    }
  }
  return GpuPerformanceModelBase::kNcclKernelLaunchOverhead;
}

}  // namespace

LatencyEstimator::TimeCost SolLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  const HloOpcode from_op = from.GetInstr().opcode();
  if (!config_.schedule_send_recvs &&
      (from_op == HloOpcode::kSend || from_op == HloOpcode::kRecv)) {
    return kLowLatency;
  }

  if (IsAsyncPair(from, target)) {
    double coll_time = absl::ToDoubleMicroseconds(ComputeCollectiveTime(
        from.GetInstr(), gpu_info_, shape_size_function_, sol_flags_));
    VLOG(10) << "[SoL] Analytical estimator calculated latency between "
             << from.GetInstr().name() << " and " << target.GetInstr().name()
             << " to be: " << coll_time << " us.";
    return coll_time;
  }
  return latency_estimator_->GetLatencyBetween(from, target);
}

LatencyEstimator::TimeCost SolLatencyEstimator::NodeCost(
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

SolLatencyEstimator::SolLatencyEstimator(
    const SchedulerConfig& config,
    std::unique_ptr<LatencyEstimator> latency_estimator,
    const se::DeviceDescription& gpu_info,
    HloCostAnalysis::ShapeSizeFunction shape_size_function,
    HloComputation* computation)
    : config_(config),
      gpu_info_(gpu_info),
      latency_estimator_(std::move(latency_estimator)),
      shape_size_function_(shape_size_function),
      sol_flags_(SolGPUCostModel::GetConfig(computation->parent())) {
  cost_analysis_.emplace(
      GpuHloCostAnalysis::Options{shape_size_function_,
                                  /*per_second_rates=*/{},
                                  /*min_latencies_seconds=*/{},
                                  /*count_multiple_input_accesses=*/true},
      gpu_info_);
  TF_CHECK_OK(computation->Accept(&cost_analysis_.value()));
  if (sol_flags_.nccl_op_launch_time == absl::ZeroDuration() ||
      sol_flags_.nic_speed_gbps == 0 ||
      sol_flags_.chunk_prep_time == absl::ZeroDuration() ||
      sol_flags_.rtt == absl::ZeroDuration() || sol_flags_.gpus_per_node == 0) {
    LOG(WARNING) << "[SoL] Failed to parse SoL system config options.";
  }
}

}  // namespace gpu
}  // namespace xla
