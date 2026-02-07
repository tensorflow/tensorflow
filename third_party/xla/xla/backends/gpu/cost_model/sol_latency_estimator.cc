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

#include "xla/backends/gpu/cost_model/sol_latency_estimator.h"

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
#include "xla/backends/gpu/cost_model/collective_interpolator.h"
#include "xla/backends/gpu/cost_model/gpu_hlo_cost_analysis.h"
#include "xla/backends/gpu/cost_model/gpu_performance_model.h"
#include "xla/backends/gpu/cost_model/gpu_performance_model_base.h"
#include "xla/backends/gpu/cost_model/hlo_op_profile.pb.h"
#include "xla/backends/gpu/cost_model/hlo_op_profiles.h"
#include "xla/backends/gpu/cost_model/matmul_interpolator.h"
#include "xla/backends/gpu/cost_model/sol_gpu_cost_model.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/flag_utils.h"
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

using ::mlir::MLIRContext;

bool IsSupportedCollectiveOp(const HloInstruction& instr) {
  return HloPredicateIsOp<HloOpcode::kAllReduceStart, HloOpcode::kAllReduce,
                          HloOpcode::kReduceScatter, HloOpcode::kAllGatherStart,
                          HloOpcode::kAllToAll,
                          HloOpcode::kCollectivePermuteStart,
                          HloOpcode::kCollectivePermute, HloOpcode::kAllGather>(
      &instr);
}

bool IsHostOffloaded(const HloInstruction& instr) {
  auto backend_config = instr.backend_config<GpuBackendConfig>();
  return backend_config.ok() &&
         backend_config->device_type() == DEVICE_TYPE_HOST;
}

bool HasOnlySupportedCollectives(const HloModule& module) {
  for (const HloComputation* comp : module.computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      if (IsHostOffloaded(*instr)) {
        return false;
      }
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
    const GpuHloCostAnalysis& analysis, MLIRContext* mlir_context) {
  SolGPUCostModel sol_model(sol_flags);
  const int64_t msg_size = analysis.BytesTransferred(instr);

  // TODO(b/385111575): We should call just `.exec_time` but we need to better
  // (more granularly) model bytes accessed (input + output) for collectives.
  absl::Duration result = absl::Seconds(1.0f * analysis.bytes_accessed(instr) /
                                        gpu_device_info.memory_bandwidth());
  GpuPerformanceModelOwning gpu_performance_model{gpu_device_info,
                                                  mlir_context};
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
    case HloOpcode::kAllToAll: {
      TF_ASSIGN_OR_RETURN(
          absl::Duration runtime,
          sol_model.AllToAllLatency(msg_size, num_participating_hosts,
                                    num_communicators));
      result += runtime;
      break;
    }
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart: {
      result += gpu_performance_model.Get()
                    .EstimateRunTimeForInstruction(&instr, &analysis)
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
      result += gpu_performance_model.Get()
                    .EstimateRunTimeForInstruction(&instr, &analysis)
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
        result += gpu_performance_model.Get()
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
      if (instr.async_wrapped_opcode() == HloOpcode::kAllToAll) {
        TF_ASSIGN_OR_RETURN(
            absl::Duration runtime,
            sol_model.AllToAllLatency(msg_size, num_participating_hosts,
                                      num_communicators));
        result += runtime;
      }
      break;
    }
    case HloOpcode::kRecv:
    case HloOpcode::kSend:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCollectivePermuteStart: {
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

int64_t GetPartitionSize(const HloInstruction& instr,
                         const SolGPUCostModel::Config& sol_flags) {
  if (sol_flags.partition_size > 0) {
    return sol_flags.partition_size;
  }
  if (instr.GetModule()->config().partition_size() > 0) {
    return instr.GetModule()->config().partition_size();
  }
  return sol_flags.gpus_per_node;
}

absl::StatusOr<absl::Duration> DispatchEstimation(
    const absl::StatusOr<GPUCommunicationType>& communication_type,
    const HloCollectiveInstruction& instr,
    const se::DeviceDescription& gpu_device_info,
    const SolGPUCostModel::Config& sol_flags,
    const GpuHloCostAnalysis& analysis,
    const CollectiveInterpolator* collective_interpolator,
    MLIRContext* mlir_context) {
  TF_RETURN_IF_ERROR(communication_type.status());

  GPUCommunicationType comm = *communication_type;
  TF_ASSIGN_OR_RETURN(auto num_groups_and_devices,
                      GetReplicaGroupCountAndSize(&instr));
  int64_t partition_size = GetPartitionSize(instr, sol_flags);

  switch (comm) {
    case GPUCommunicationType::MULTI_HOST_WORLD_LEVEL: {
      return DCNCollectiveDuration(
          num_groups_and_devices->second / partition_size,
          /*num_communicators=*/num_groups_and_devices->first, instr,
          gpu_device_info, sol_flags, analysis, mlir_context);
    }
    case GPUCommunicationType::MULTI_HOST_NON_WORLD_LEVEL: {
      return DCNCollectiveDuration(
          num_groups_and_devices->second,
          /*num_communicators=*/num_groups_and_devices->first, instr,
          gpu_device_info, sol_flags, analysis, mlir_context);
    }
    case GPUCommunicationType::SINGLE_PARTITION: {
      if (collective_interpolator == nullptr) {
        return absl::InvalidArgumentError(
            "Collective interpolator is required for single partition "
            "collectives");
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
    const SolGPUCostModel::Config& sol_flags, MLIRContext* mlir_context,
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
      instr, gpu_device_info, shape_size_fn, sol_flags, analysis, mlir_context,
      collective_interpolator);
}

/*static*/ absl::StatusOr<absl::Duration>
SolLatencyEstimator::ComputeCollectiveTime(
    const HloInstruction& instr, const se::DeviceDescription& gpu_device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    const SolGPUCostModel::Config& sol_flags,
    const GpuHloCostAnalysis& analysis, MLIRContext* mlir_context,
    const CollectiveInterpolator* collective_interpolator) {
  if (HloDataflowAnalysis::IsAsynchronousOperationDone(instr.opcode())) {
    VLOG(8) << "Returning 0 cost for async done op " << instr.name();
    return absl::ZeroDuration();
  }

  const HloInstruction* collective =
      instr.IsAsynchronous() ? instr.async_wrapped_instruction() : &instr;
  if (const auto* cp = DynCast<HloCollectivePermuteInstruction>(collective)) {
    // Handles the collective-permute ops.
    int64_t partition_size = GetPartitionSize(*cp, sol_flags);
    CollectivePermuteCostModelType cost_model_type =
        GetCollectivePermuteCostModelType(*cp, partition_size);

    switch (cost_model_type) {
      case CollectivePermuteCostModelType::kIntraPartitionOneWay:
      case CollectivePermuteCostModelType::kIntraPartitionTwoWayAllMutual:
      case CollectivePermuteCostModelType::kIntraPartitionTwoWayHasNonMutual:
        return collective_interpolator->EstimatedRuntime(*cp);
      case CollectivePermuteCostModelType::kInterPartitionOneWay:
      case CollectivePermuteCostModelType::kInterPartitionTwoWayAllMutual:
      case CollectivePermuteCostModelType::kInterPartitionTwoWayHasNonMutual: {
        // TODO(wfelix): Distinguish different types of inter-partition
        // collectives.
        TF_ASSIGN_OR_RETURN(
            absl::Duration duration,
            DCNCollectiveDuration(/*num_participating_hosts=*/2,
                                  /*num_communicators=*/1, *cp, gpu_device_info,
                                  sol_flags, analysis, mlir_context));
        return duration;
      }
      case CollectivePermuteCostModelType::kUnknown:
        return absl::InvalidArgumentError(
            "Unknown collective permute cost model type.");
    }
  } else if (const auto* collective_instr =
                 DynCast<HloCollectiveInstruction>(collective)) {
    // Handles the collective ops.
    int64_t partition_size = GetPartitionSize(*collective_instr, sol_flags);
    TF_ASSIGN_OR_RETURN(
        GPUCommunicationType communication_type,
        CommunicationType(partition_size, *collective_instr,
                          gpu_device_info.gpu_compute_capability()));
    TF_ASSIGN_OR_RETURN(
        absl::Duration result,
        DispatchEstimation(communication_type, *collective_instr,
                           gpu_device_info, sol_flags, analysis,
                           collective_interpolator, mlir_context));
    return result;
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported collective instruction: ", instr.ToString()));
}

/*static*/ absl::StatusOr<std::unique_ptr<SolLatencyEstimator>>
SolLatencyEstimator::Create(
    const SchedulerConfig& config,
    std::unique_ptr<LatencyEstimator> latency_estimator,
    const se::DeviceDescription& gpu_info,
    HloCostAnalysis::ShapeSizeFunction shape_size_function,
    const HloComputation* computation, MLIRContext* mlir_context,
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
      std::move(matmul_interpolator), mlir_context));
}

/*static*/ bool SolLatencyEstimator::IsSupportedForModule(
    const HloModule& module, const se::DeviceDescription& gpu_device_info) {
  bool is_supported_device =
      gpu_device_info.cuda_compute_capability().IsHopper() ||
      gpu_device_info.cuda_compute_capability().IsBlackwell();
  if (IsPassEnabledAtOptimizationEffort<LatencyHidingScheduler>(module)) {
    // If the user enabled opt effort we turn the estimator on if we're
    // compiling for Hopper/Blackwell.
    return is_supported_device;
  }
  // If this flag is on by default then we provide users an escape hatch in case
  // they find the new cost model less profitable than T-shirt sizes.
  if (!module.config()
           .debug_options()
           .xla_gpu_enable_analytical_sol_latency_estimator()) {
    return false;
  }
  // Otherwise we are more conservative and we turn it on only for
  // Hopper/Blackwell and if `module` contains only supported collectives.
  return is_supported_device && HasOnlySupportedCollectives(module);
}

LatencyEstimator::TimeCost SolLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  const HloOpcode from_op = from.GetInstr().opcode();
  if (!config_.schedule_send_recvs &&
      (from_op == HloOpcode::kSend || from_op == HloOpcode::kRecv)) {
    VLOG(10) << "GetLatencyBetween: Returning kLowLatency for Send/Recv op "
             << from.GetInstr().name();
    return kLowLatency;
  }

  if (!IsAsyncPair(from, target) && !IsSupportedCollectiveOp(from.GetInstr())) {
    TimeCost latency = latency_estimator_->GetLatencyBetween(from, target);
    VLOG(10)
        << "GetLatencyBetween: Not an async pair or unsupported collective "
        << from.GetInstr().name() << ", returning latency from wrapped "
        << "estimator: " << latency;
    return latency;
  }

  absl::StatusOr<absl::Duration> coll_time = ComputeCollectiveTime(
      from.GetInstr(), gpu_info_, shape_size_function_, sol_flags_,
      *cost_analysis_, mlir_context_, collective_interpolator_.get());
  if (!coll_time.ok()) {
    VLOG(1) << "Failed to compute collective time: " << coll_time.status()
            << " for " << from.GetInstr().name();
    TimeCost latency = latency_estimator_->GetLatencyBetween(from, target);
    VLOG(10) << "GetLatencyBetween: Fallback to wrapped estimator due to "
                "ComputeCollectiveTime failure for "
             << from.GetInstr().name() << ", returning latency: " << latency;
    return latency;
  }
  TimeCost latency = absl::ToDoubleMicroseconds(*coll_time);
  VLOG(10) << "GetLatencyBetween: Computed collective time for "
           << from.GetInstr().name() << ": " << latency << " us";
  return latency;
}

LatencyEstimator::TimeCost SolLatencyEstimator::NodeCost(
    const HloInstruction* instr) const {
  if (std::optional<double> latency = GetCustomCallLatencyMetadata(instr)) {
    VLOG(10) << "NodeCost: Returning latency from custom call for "
             << instr->name() << ": " << *latency << " us";
    return *latency;
  }
  if (hlo_query::IsAsyncCollectiveStartOp(instr, /*include_send_recv=*/true) ||
      hlo_query::IsAsyncCollectiveDoneOp(instr, /*include_send_recv=*/true)) {
    VLOG(10) << "NodeCost: Returning kLowCost for async start/done op "
             << instr->name();
    return kLowCost;
  }

  if (std::optional<absl::Duration> matmul_duration =
          matmul_interpolator_->EstimatedRuntime(*instr);
      matmul_duration.has_value()) {
    TimeCost cost = absl::ToDoubleMicroseconds(*matmul_duration);
    VLOG(10) << "NodeCost: Matmul cost from matmul_interpolator for "
             << instr->name() << ": " << cost << " us";
    return cost;
  }

  LatencyEstimator::TimeCost cost_in_us;
  // Custom fusion ops are hard to estimate, so we only use the performance
  // model for loop and input fusions. Otherwise we return a small cost to make
  // sure we can achieve overlap (even at the cost of overextension).
  if (instr->IsLoopFusion() || instr->IsInputFusion()) {
    absl::Duration total_estimated_time =
        gpu_performance_model_.Get()
            .EstimateRunTimeForInstruction(instr, &*cost_analysis_)
            .exec_time;
    cost_in_us = absl::ToDoubleMicroseconds(total_estimated_time);
    VLOG(10) << "NodeCost: Fusion cost from gpu_performance_model for "
             << instr->name() << ": " << cost_in_us << " us";
  } else {
    cost_in_us = 0.01 * latency_estimator_->NodeCost(instr);
    VLOG(10) << "NodeCost: Fallback cost for " << instr->name() << ": "
             << cost_in_us << " us";
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
    std::unique_ptr<MatmulInterpolator> matmul_interpolator,
    MLIRContext* mlir_context)
    : config_(config),
      gpu_info_(gpu_info),
      gpu_performance_model_(gpu_info, mlir_context),
      cost_analysis_(std::move(cost_analysis)),
      latency_estimator_(std::move(latency_estimator)),
      shape_size_function_(shape_size_function),
      sol_flags_(sol_flags),
      collective_interpolator_(std::move(collective_interpolator)),
      matmul_interpolator_(std::move(matmul_interpolator)),
      mlir_context_(mlir_context) {}

}  // namespace gpu
}  // namespace xla
