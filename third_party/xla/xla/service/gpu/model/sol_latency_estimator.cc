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
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/service/gpu/backend_configs.pb.h"
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

std::optional<absl::Duration> DCNCollectiveDuration(
    int num_participating_hosts, absl::string_view mask,
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
      result += sol_model.RingLatency(
          msg_size, num_participating_hosts,
          SolGPUCostModel::CollectiveType::kAllGather, mask);
      break;
    }
    case HloOpcode::kAllReduce:
    case HloOpcode::kAllReduceStart: {
      result +=
          gpu_performance_model.EstimateRunTimeForInstruction(&instr, &analysis)
              .compute_time;
      result += sol_model.RingLatency(
          msg_size, num_participating_hosts,
          SolGPUCostModel::CollectiveType::kAllReduce, mask);
      break;
    }
    case HloOpcode::kReduceScatter: {
      result +=
          gpu_performance_model.EstimateRunTimeForInstruction(&instr, &analysis)
              .compute_time;
      result += sol_model.RingLatency(
          msg_size, num_participating_hosts,
          SolGPUCostModel::CollectiveType::kReduceScatter, mask);
      break;
    }
    case HloOpcode::kAsyncStart: {
      if (instr.async_wrapped_opcode() == HloOpcode::kReduceScatter) {
        result += gpu_performance_model
                      .EstimateRunTimeForInstruction(
                          instr.async_wrapped_instruction(), &analysis)
                      .compute_time;
        result += sol_model.RingLatency(
            msg_size, num_participating_hosts,
            SolGPUCostModel::CollectiveType::kReduceScatter, mask);
      }
      break;
    }
    case HloOpcode::kRecv:
    case HloOpcode::kSend: {
      result += sol_model.RingLatency(
          msg_size, num_participating_hosts,
          SolGPUCostModel::CollectiveType::kSendRecv, mask);
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

std::optional<absl::Duration> DispatchEstimation(
    const absl::StatusOr<GPUCommunicationType>& communication_type,
    const HloCollectiveInstruction& instr,
    const se::DeviceDescription& gpu_device_info,
    const SolGPUCostModel::Config& sol_flags,
    const GpuHloCostAnalysis& analysis,
    const CollectiveInterpolator* collective_interpolator) {
  if (!communication_type.ok()) {
    VLOG(1) << "Failed to determine communication type: "
            << communication_type.status();
    return std::nullopt;
  }

  GPUCommunicationType comm = *communication_type;
  auto num_groups_and_devices = GetReplicaGroupCountAndSize(&instr);
  if (!num_groups_and_devices.ok()) {
    VLOG(1) << "Failed to determine a number of devices participating in "
               "the collective: "
            << instr.ToString();
    return std::nullopt;
  }

  switch (comm) {
    case GPUCommunicationType::RAIL_ALIGNED: {
      return DCNCollectiveDuration(
          (*num_groups_and_devices)->second / sol_flags.gpus_per_node,
          SolGPUCostModel::kSplitMaskWorldLevel, instr, gpu_device_info,
          sol_flags, analysis);
    }
    case GPUCommunicationType::NON_RAIL_ALIGNED: {
      return DCNCollectiveDuration((*num_groups_and_devices)->second,
                                   SolGPUCostModel::kSplitMaskNonRailAligned,
                                   instr, gpu_device_info, sol_flags, analysis);
    }
    case GPUCommunicationType::SINGLE_HOST: {
      if (collective_interpolator == nullptr) {
        return GpuPerformanceModelBase::kNcclKernelLaunchOverhead;
      }
      return collective_interpolator->EstimatedRuntime(instr);
    }
    case xla::gpu::GPUCommunicationType::UNDEFINED:
      LOG(WARNING) << "Cannot determine communication type: "
                   << instr.ToString();
      return std::nullopt;
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

/*static*/ absl::Duration SolLatencyEstimator::ComputeCollectiveTime(
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

/*static*/ absl::Duration SolLatencyEstimator::ComputeCollectiveTime(
    const HloInstruction& instr, const se::DeviceDescription& gpu_device_info,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn,
    const SolGPUCostModel::Config& sol_flags,
    const GpuHloCostAnalysis& analysis,
    const CollectiveInterpolator* collective_interpolator) {
  if (HloDataflowAnalysis::IsAsynchronousOperationDone(instr.opcode())) {
    VLOG(8) << "Returning 0 cost for async done op " << instr.name();
    return absl::ZeroDuration();
  }

  if (auto* collective_instr = DynCast<HloCollectiveInstruction>(
          instr.IsAsynchronous() ? instr.async_wrapped_instruction() : &instr);
      collective_instr != nullptr) {
    absl::StatusOr<GPUCommunicationType> communication_type =
        CommunicationType(sol_flags.gpus_per_node, *collective_instr,
                          gpu_device_info.gpu_compute_capability());
    return DispatchEstimation(communication_type, *collective_instr,
                              gpu_device_info, sol_flags, analysis,
                              collective_interpolator)
        .value_or(absl::ZeroDuration());
  }

  return absl::ZeroDuration();
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

LatencyEstimator::TimeCost SolLatencyEstimator::GetLatencyBetween(
    const HloGraphNode& from, const HloGraphNode& target) const {
  const HloOpcode from_op = from.GetInstr().opcode();
  if (!config_.schedule_send_recvs &&
      (from_op == HloOpcode::kSend || from_op == HloOpcode::kRecv)) {
    return kLowLatency;
  }

  if (IsAsyncPair(from, target)) {
    double coll_time = absl::ToDoubleMicroseconds(ComputeCollectiveTime(
        from.GetInstr(), gpu_info_, shape_size_function_, sol_flags_,
        *cost_analysis_, collective_interpolator_.get()));
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

  if (std::optional<absl::Duration> matmul_duration =
          matmul_interpolator_->EstimatedRuntime(*instr);
      matmul_duration.has_value()) {
    return absl::ToDoubleMicroseconds(*matmul_duration);
  }

  LatencyEstimator::TimeCost cost_in_us;
  if (instr->opcode() == HloOpcode::kFusion &&
      (instr->IsLoopFusion() || instr->IsInputFusion())) {
    absl::Duration total_estimated_time =
        gpu_performance_model_
            .EstimateRunTimeForInstruction(instr, &*cost_analysis_)
            .exec_time;
    cost_in_us = absl::ToDoubleMicroseconds(total_estimated_time);
  } else {
    cost_in_us = 0.01 * kLowCost;
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
