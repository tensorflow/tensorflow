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

#include "xla/service/gpu/model/gpu_cost_model_stats_collection.h"

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/service/gpu/model/gpu_dot_fusion_cost_model.h"
#include "xla/service/gpu/model/gpu_indexing_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/service/gpu/model/gpu_performance_model_base.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

absl::StatusOr<EstimateRunTimeData> MaybeGetGemmCostModelForGemmTritonFusion(
    const se::DeviceDescription& device_info,
    const HloInstruction& instruction) {
  const HloFusionInstruction* fusion =
      DynCast<HloFusionInstruction>(&instruction);
  if (fusion == nullptr ||
      fusion->fusion_kind() != HloInstruction::FusionKind::kCustom) {
    return absl::FailedPreconditionError("Not a custom fusion.");
  }

  TF_ASSIGN_OR_RETURN(GpuBackendConfig config,
                      fusion->backend_config<GpuBackendConfig>());
  if (config.fusion_backend_config().kind() != kTritonNestedGemmFusionKind) {
    return absl::FailedPreconditionError("Not a Triton GeMM fusion.");
  }

  const HloInstruction* dot_instruction =
      hlo_query::GetFirstInstructionWithOpcode(
          *fusion->fused_instructions_computation(), HloOpcode::kDot);
  if (dot_instruction == nullptr) {
    return absl::FailedPreconditionError(
        "No kDot instruction found in Triton fusion.");
  }

  const HloDotInstruction* dot =
      DynCast<const HloDotInstruction>(dot_instruction);
  if (dot == nullptr) {
    return absl::FailedPreconditionError(
        "No kDot instruction found in Triton fusion.");
  }

  if (!config.fusion_backend_config().has_block_level_fusion_config()) {
    return absl::FailedPreconditionError(
        "Fusion backend config does not have block level fusion config.");
  }
  BlockLevelParameters block_params =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          config.fusion_backend_config().block_level_fusion_config());

  return gpu_dot_fusion_cost_model::EstimateRunTimeForDotOpWithBlockParameters(
      dot, block_params, device_info);
}

// If `instruction` is a Triton-fused GEMM, computes its runtime estimation
// using an analytical cost model and adds this as a reification cost.
// This cost model focuses on the dot operation within the fusion. Fusions
// with non-trivial operations on dot operands might not be fully accounted for.
absl::Status RecordGemmCostModelEstimateIfApplicable(
    const se::DeviceDescription& device_info, HloInstruction& instruction) {
  TF_ASSIGN_OR_RETURN(
      EstimateRunTimeData runtime,
      MaybeGetGemmCostModelForGemmTritonFusion(device_info, instruction));

  ReificationCost cost =
      GpuPerformanceModelBase::MakeReificationCostFromRuntime(
          runtime, device_info, "experimental-gemm-cost-model");

  VLOG(1) << "Adding GeMM fusion cost model estimate: " << cost.DebugString();

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instruction.backend_config<GpuBackendConfig>());
  *gpu_config.add_reification_cost() = cost;
  return instruction.set_backend_config(gpu_config);
}

absl::StatusOr<EstimateRunTimeData> MaybeGetIndexingCostModelForFusion(
    GpuPerformanceModelWithIndexingAnalysis& perf_model,
    const se::DeviceDescription& device_info, HloInstruction& instruction) {
  const HloFusionInstruction* fusion =
      DynCast<HloFusionInstruction>(&instruction);
  if (fusion == nullptr ||
      fusion->fusion_kind() != HloInstruction::FusionKind::kCustom) {
    return absl::FailedPreconditionError("Not a custom fusion.");
  }

  TF_ASSIGN_OR_RETURN(EstimateRunTimeData runtime,
                      perf_model.EstimateRunTimeForTriton(&instruction));

  return runtime;
}

absl::Status RecordIndexingPerformanceModelEstimateIfApplicable(
    GpuPerformanceModelWithIndexingAnalysis& perf_model,
    const se::DeviceDescription& device_info, HloInstruction& instruction) {
  TF_ASSIGN_OR_RETURN(
      EstimateRunTimeData runtime,
      MaybeGetIndexingCostModelForFusion(perf_model, device_info, instruction));

  ReificationCost cost =
      GpuPerformanceModelBase::MakeReificationCostFromRuntime(
          runtime, device_info, "indexing-cost-model");

  VLOG(1) << "Adding indexing performance model estimate: "
          << cost.DebugString();

  TF_ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                      instruction.backend_config<GpuBackendConfig>());
  *gpu_config.add_reification_cost() = cost;
  return instruction.set_backend_config(gpu_config);
}

}  // namespace

absl::StatusOr<bool> GpuCostModelStatsCollection::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Scan all computations for fusion instructions.

  GpuPerformanceModelOwning gpu_performance_model{device_info_, mlir_context_};
  for (auto* computation : module->MakeComputationPostOrder()) {
    CHECK_OK(computation->Accept(&cost_analysis_));

    for (auto* fusion_instr : computation->instructions()) {
      if (fusion_instr->opcode() != HloOpcode::kFusion) {
        continue;
      }

      VLOG(1) << "Collecting cost model stats on "
              << fusion_instr->ToShortString();

      gpu_performance_model.Get().RecordEstimatedRunTime(fusion_instr,
                                                         &cost_analysis_);

      if (absl::Status status = RecordGemmCostModelEstimateIfApplicable(
              device_info_, *fusion_instr);
          !status.ok()) {
        VLOG(1) << "Skipping GeMM fusion cost model estimate: "
                << status.ToString(
                       (VLOG_IS_ON(2))
                           ? absl::StatusToStringMode::kWithEverything
                           : absl::StatusToStringMode::kWithNoExtraData);
      }
      if (absl::Status status =
              RecordIndexingPerformanceModelEstimateIfApplicable(
                  indexing_cost_analysis_, device_info_, *fusion_instr);
          !status.ok()) {
        VLOG(1) << "Skipping indexing cost model estimate: "
                << status.ToString(
                       (VLOG_IS_ON(2))
                           ? absl::StatusToStringMode::kWithEverything
                           : absl::StatusToStringMode::kWithNoExtraData);
      }
    }
  }
  return false;
}

}  // namespace gpu
}  // namespace xla
