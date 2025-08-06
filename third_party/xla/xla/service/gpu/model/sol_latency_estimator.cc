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
#include <optional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/flag_utils.h"
#include "xla/service/gpu/model/collective_interpolator.h"
#include "xla/service/gpu/model/gpu_hlo_cost_analysis.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/service/gpu/model/hlo_op_profile.pb.h"
#include "xla/service/gpu/model/hlo_op_profiles.h"
#include "xla/service/gpu/model/matmul_interpolator.h"
#include "xla/service/gpu/model/sol_gpu_cost_model.h"
#include "xla/service/gpu/transforms/collectives/collective_ops_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/latency_hiding_scheduler.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace {

bool IsSupportedCollectiveOp(const HloInstruction& instr) {
  return HloPredicateIsOp<HloOpcode::kAllReduceStart, HloOpcode::kAllReduce,
                          HloOpcode::kReduceScatter, HloOpcode::kAllGatherStart,
                          HloOpcode::kAllGather>(&instr);
}

bool HasOnlySupportedCollectives(const HloModule& module) {
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (hlo_query::IsCollectiveCommunicationOp(instr->opcode()) &&
          !IsSupportedCollectiveOp(*instr)) {
        return false;
      }
    }
  }
  return true;
}

absl::StatusOr<HloInstructionProfileList> ReadProfiles(
    const std::string& perf_table_path,
    const se::DeviceDescription& device_info) {
  DeviceHloInstructionProfiles profile;

  TF_RETURN_IF_ERROR(tsl::Env::Default()->FileExists(perf_table_path));
  TF_RETURN_IF_ERROR(tsl::ReadTextOrBinaryProto(tsl::Env::Default(),
                                                perf_table_path, &profile));
  std::string key = HloOpProfiles::GetProfileName(device_info);

  if (!profile.entries().contains(key)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Key not present: ", key));
  }
  return profile.entries().at(key);
}

absl::StatusOr<absl::Duration> DCNCollectiveDuration(
    int num_participating_hosts, int num_communicators,
    const HloInstruction& instr, const se::DeviceDescription& gpu_device_info,
    const SolGPUCostModel::Config& sol_flags,
    const GpuHloCostAnalysis& analysis) {
  SolGPUCostModel sol_model(sol_flags);
  const int64_t msg_size = analysis.BytesTransferred(instr);

  // TODO(b/385111575): We should call just `.exec_time` but we need to better
  // (more granularly) model bytes accessed (input + output) for collectives.
  absl::Duration result = absl::Seconds(1.0f * analysis.bytes_accessed(instr) /
                                        gpu_device_info.memory_bandwidth());
  GpuPerformanceModelOwning gpu_performance_model{gpu_device_info};
  switch (instr.opcode()) {
    case HloOpcode::kAllGather:
    case HloOpcode::kAllGatherStart: {
      TF_ASSIGN_OR_RETURN(
          absl::Duration runtime,
          sol_model.RingLatency(msg_size, num_participating_hosts,
                                SolGPUCostModel::CollectiveType::kAllGather,
                                num_communicators));
      result += runtime;
      break;
    }
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart: {
      result +=
          gpu_performance_model.EstimateRunTimeForInstruction(&instr, &analysis)
              .compute_time;
      TF_ASSIGN_OR_RETURN(
          absl::Duration runtime,
          sol_model.RingLatency(msg_size, num_participating_hosts,
                                SolGPUCostModel::CollectiveType::kAllReduce,
                                num_communicators));
      result += runtime;
      break;
    }
    case HloOpcode::kReduceScatter: {
      result +=
          gpu_performance_model.EstimateRunTimeForInstruction(&instr, &analysis)
              .compute_time;
      TF_ASSIGN_OR_RETURN(
          absl::Duration runtime,
          sol_model.RingLatency(msg_size, num_participating_hosts,
                                SolGPUCostModel::CollectiveType::kReduceScatter,
                                num_communicators));
      result += runtime;
      break;
    }
    case HloOpcode::kAsyncStart: {
      if (instr.async_wrapped_opcode() == HloOpcode::kReduceScatter) {
        result += gpu_performance_model
                      .EstimateRunTimeForInstruction(
                          instr.async_wrapped_instruction(), &analysis)
                      .compute_time;
        TF_ASSIGN_OR_RETURN(absl::Duration runtime,
                            sol_model.RingLatency(
                                msg_size, num_participating_hosts,
                                SolGPUCostModel::CollectiveType::kReduceScatter,
                                num_communicators));
        result += runtime;
      }
      break;
    }
    case HloOpcode::kRecv:
    case HloOpcode::kSend: {
      TF_ASSIGN_OR_RETURN(
          absl::Duration runtime,
          sol_model.RingLatency(msg_size, num_participating_hosts,
                                SolGPUCostModel::CollectiveType::kSendRecv,
                                num_communicators));
      result += runtime;
      break;
    }
    // note: AllToAll is not yet supported in XLA
    default: {
      LOG(WARNING)
          << "[SoL] Runtime estimate for " << instr.name()
          << " not implemented. Returning only the kernel launch time.";
      result += GpuPerformanceModelBase::kNcclKernelLaunchOverhead;
    }
  }
  return result;
}

absl::StatusOr<absl::Duration> DispatchEstimation(
    const absl::StatusOr<GPUCommunicationType>& communication_type,
    const HloCollectiveInstruction& instr,
    const se::DeviceDescription& gpu_device_info,
    const SolGPUCostModel::Config& sol_flags,
    const GpuHloCostAnalysis& analysis,
    const CollectiveInterpolator* collective_interpolator) {
  TF_RETURN_IF_ERROR(communication_type.status());

  GPUCommunicationType comm = *communication_type;
  TF_ASSIGN_OR_RETURN(auto num_groups_and_devices,
                      GetReplicaGroupCountAndSize(&instr));

  switch (comm) {
    case GPUCommunicationType::RAIL_ALIGNED: {
      return DCNCollectiveDuration(
          num_groups_and_devices->second / sol_flags.gpus_per_node,
          /*num_communicators=*/num_groups_and_devices->first, instr,
          gpu_device_info, sol_flags, analysis);
    }
    case GPUCommunicationType::NON_RAIL_ALIGNED: {
      return DCNCollectiveDuration(
          num_groups_and_devices->second,
          /*num_communicators=*/num_groups_and_devices->first, instr,
          gpu_device_info, sol_flags, analysis);
    }
    case GPUCommunicationType::SINGLE_HOST: {
      if (collective_interpolator == nullptr) {
        return absl::InvalidArgumentError(
            "Collective interpolator is required for single host collectives");
      }
      return collective_interpolator->EstimatedRuntime(instr);
    }
    case xla::gpu::GPUCommunicationType::UNDEFINED:
      return absl::InvalidArgumentError("Cannot determine communication type");
  }
}

absl::StatusOr<std::unique_ptr<CollectiveInterpolator>>
CreateCollectiveInterpolator(int num_devices_per_host, const HloModule& module,
                             const se::DeviceDescription& device_info,
                             const GpuHloCostAnalysis& analysis) {
  absl::StatusOr<HloInstructionProfileList> collective_profiles =
      ReadProfiles(module.config()
                       .debug_options()
                       .xla_gpu_experimental_collective_perf_table_path(),
                   device_info);
  std::unique_ptr<CollectiveInterpolator> collective_interpolator;
  if (collective_profiles.ok()) {
    return CollectiveInterpolator::Create(
        num_devices_per_host, *collective_profiles, device_info, &analysis);
  }
  return CollectiveInterpolator::Create(num_devices_per_host, device_info,
                                        &analysis);
}

absl::StatusOr<std::unique_ptr<MatmulInterpolator>> CreateMatmulInterpolator(
    const HloModule& module, const se::DeviceDescription& device_info) {
  absl::StatusOr<HloInstructionProfileList> matmul_profiles =
      ReadProfiles(module.config()
                       .debug_options()
                       .xla_gpu_experimental_matmul_perf_table_path(),
                   device_info);
  std::unique_ptr<MatmulInterpolator> matmul_interpolator;
  if (matmul_profiles.ok()) {
    return MatmulInterpolator::Create(*matmul_profiles, device_info);
  }
  return MatmulInterpolator::Create(device_info);
}

}  // namespace

/*static*/ absl::StatusOr<absl::Duration>
SolLatencyEstimator::ComputeCollectiveTime(
    const HloInstruction& instr, const se::DeviceDescription& gpu_device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    const SolGPUCostModel::Config& sol_flags,
    const CollectiveInterpolator* collective_interpolator) {
  GpuHloCostAnalysis analysis(
      GpuHloCostAnalysis::Options{shape_size_fn,
                                  /*per_second_rates=*/{},
                                  /*min_latencies_seconds=*/{},
                                  /*count_multiple_input_accesses=*/true},
      gpu_device_info);

  CHECK_OK(instr.parent()->Accept(&analysis));

  if (instr.IsAsynchronous()) {
    CHECK_OK(instr.async_wrapped_instruction()->Accept(&analysis));
  }

  return SolLatencyEstimator::ComputeCollectiveTime(
      instr, gpu_device_info, shape_size_fn, sol_flags, analysis,
      collective_interpolator);
}

/*static*/ absl::StatusOr<absl::Duration>
SolLatencyEstimator::ComputeCollectiveTime(
    const HloInstruction& instr, const se::DeviceDescription& gpu_device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    const SolGPUCostModel::Config& sol_flags,
    const GpuHloCostAnalysis& analysis,
    const CollectiveInterpolator* collective_interpolator) {
  if (HloDataflowAnalysis::IsAsynchronousOperationDone(instr.opcode())) {
    VLOG(8) << "Returning 0 cost for async done op " << instr.name();
    return absl::ZeroDuration();
  }

  const HloCollectiveInstruction* collective_instr =
      DynCast<HloCollectiveInstruction>(
          instr.IsAsynchronous() ? instr.async_wrapped_instruction() : &instr);

  if (collective_instr == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported collective instruction: ", instr.ToString()));
  }

  TF_ASSIGN_OR_RETURN(
      GPUCommunicationType communication_type,
      CommunicationType(sol_flags.gpus_per_node, *collective_instr,
                        gpu_device_info.gpu_compute_capability()));
  TF_ASSIGN_OR_RETURN(
      absl::Duration result,
      DispatchEstimation(communication_type, *collective_instr, gpu_device_info,
                         sol_flags, analysis, collective_interpolator));
  return result;
}

/*static*/ absl::StatusOr<std::unique_ptr<SolLatencyEstimator>>
SolLatencyEstimator::Create(
    const SchedulerConfig& config,
    std::unique_ptr<LatencyEstimator> latency_estimator,
    const se::DeviceDescription& gpu_info,
    HloCostAnalysis::ShapeSizeFunction shape_size_function,
    const HloComputation* computation,
    std::unique_ptr<GpuHloCostAnalysis> cost_analysis) {
  if (cost_analysis == nullptr) {
    cost_analysis =
        std::make_unique<GpuHloCostAnalysis>(GpuHloCostAnalysis::Options{
            shape_size_function,
            /*per_second_rates=*/{},
            /*min_latencies_seconds=*/{},
            /*count_multiple_input_accesses=*/true,
        });
    TF_RETURN_IF_ERROR(computation->Accept(cost_analysis.get()));
  }
  SolGPUCostModel::Config sol_config =
      SolGPUCostModel::GetConfig(computation->parent(), gpu_info);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<CollectiveInterpolator> collective_interpolator,
      CreateCollectiveInterpolator(sol_config.gpus_per_node,
                                   *computation->parent(), gpu_info,
                                   *cost_analysis));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<MatmulInterpolator> matmul_interpolator,
      CreateMatmulInterpolator(*computation->parent(), gpu_info));
  return std::unique_ptr<SolLatencyEstimator>(new SolLatencyEstimator(
      config, std::move(latency_estimator), gpu_info, std::move(cost_analysis),
      shape_size_function, sol_config, std::move(collective_interpolator),
      std::move(matmul_interpolator)));
}

/*static*/ bool SolLatencyEstimator::IsSupportedForModule(
    const HloModule& module, const se::DeviceDescription& gpu_device_info) {
  if (IsPassEnabledAtOptimizationEffort<LatencyHidingScheduler>(module)) {
    // If the user enabled opt effort we turn the estimator on if we're
    // compiling for Hopper.
    return gpu_device_info.cuda_compute_capability().IsHopper();
  }
  // If this flag is on by default then we provide users an escape hatch in case
  // they find the new cost model less profitable than T-shirt sizes.
  if (!module.config()
           .debug_options()
           .xla_gpu_enable_analytical_sol_latency_estimator()) {
    return false;
  }
  // Otherwise we are more conservative and we turn it on only for Hopper and if
  // `module` contains only supported collectives.
  return gpu_device_info.cuda_compute_capability().IsHopper() &&
         HasOnlySupportedCollectives(module);
}

LatencyEstimator::TimeCost SolLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  const HloOpcode from_op = from.GetInstr().opcode();
  if (!config_.schedule_send_recvs &&
      (from_op == HloOpcode::kSend || from_op == HloOpcode::kRecv)) {
    return kLowLatency;
  }

  if (!IsAsyncPair(from, target) || !IsSupportedCollectiveOp(from.GetInstr())) {
    return latency_estimator_->GetLatencyBetween(from, target);
  }

  absl::StatusOr<absl::Duration> coll_time = ComputeCollectiveTime(
      from.GetInstr(), gpu_info_, shape_size_function_, sol_flags_,
      *cost_analysis_, collective_interpolator_.get());
  if (!coll_time.ok()) {
    VLOG(1) << "Failed to compute collective time: " << coll_time.status()
            << " for " << from.GetInstr().name();
    return latency_estimator_->GetLatencyBetween(from, target);
  }
  return absl::ToDoubleMicroseconds(*coll_time);
}

LatencyEstimator::TimeCost SolLatencyEstimator::NodeCost(
    const HloInstruction* instr) const {
  if (hlo_query::IsAsyncCollectiveStartOp(instr, /*include_send_recv=*/true) ||
      hlo_query::IsAsyncCollectiveDoneOp(instr, /*include_send_recv=*/true)) {
    return kLowCost;
  }

  if (std::optional<absl::Duration> matmul_duration =
          matmul_interpolator_->EstimatedRuntime(*instr);
      matmul_duration.has_value()) {
    return absl::ToDoubleMicroseconds(*matmul_duration);
  }

  LatencyEstimator::TimeCost cost_in_us;
  // Custom fusion ops are hard to estimate, so we only use the performance
  // model for loop and input fusions. Otherwise we return a small cost to make
  // sure we can achieve overlap (even at the cost of overextension).
  if (instr->IsLoopFusion() || instr->IsInputFusion()) {
    absl::Duration total_estimated_time =
        gpu_performance_model_
            .EstimateRunTimeForInstruction(instr, &*cost_analysis_)
            .exec_time;
    cost_in_us = absl::ToDoubleMicroseconds(total_estimated_time);
  } else {
    cost_in_us = 0.01 * latency_estimator_->NodeCost(instr);
  }
  VLOG(10) << "Analytical estimator calculated cost for: " << instr->name()
           << ". Cost: " << cost_in_us;
  return cost_in_us;
}

SolLatencyEstimator::SolLatencyEstimator(
    const SchedulerConfig& config,
    std::unique_ptr<LatencyEstimator> latency_estimator,
    const se::DeviceDescription& gpu_info,
    std::unique_ptr<const GpuHloCostAnalysis> cost_analysis,
    const HloCostAnalysis::ShapeSizeFunction shape_size_function,
    const SolGPUCostModel::Config sol_flags,
    std::unique_ptr<CollectiveInterpolator> collective_interpolator,
    std::unique_ptr<MatmulInterpolator> matmul_interpolator)
    : config_(config),
      gpu_info_(gpu_info),
      gpu_performance_model_(gpu_info),
      cost_analysis_(std::move(cost_analysis)),
      latency_estimator_(std::move(latency_estimator)),
      shape_size_function_(shape_size_function),
      sol_flags_(sol_flags),
      collective_interpolator_(std::move(collective_interpolator)),
      matmul_interpolator_(std::move(matmul_interpolator)) {}

}  // namespace gpu
}  // namespace xla
