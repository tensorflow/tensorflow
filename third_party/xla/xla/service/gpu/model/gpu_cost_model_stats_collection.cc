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
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
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
#include "xla/service/gpu/model/gpu_performance_model.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

absl::StatusOr<absl::Duration> MaybeGetGemmCostModelForGemmTritonFusion(
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

  return GpuDotFusionCostModel::EstimateRunTimeForDotOpWithBlockParameters(
      dot, block_params, device_info);
}

// If `instruction` is a Triton-fused GEMM, computes its runtime estimation
// using an analytical cost model and adds this as a reification cost.
// This cost model focuses on the dot operation within the fusion. Fusions
// with non-trivial operations on dot operands might not be fully accounted for.
void RecordGemmCostModelEstimateIfApplicable(
    const se::DeviceDescription& device_info, HloInstruction& instruction) {
  absl::StatusOr<absl::Duration> duration =
      MaybeGetGemmCostModelForGemmTritonFusion(device_info, instruction);
  if (!duration.ok()) {
    VLOG(3) << "Skipping the GeMM fusion cost model: "
            << duration.status().ToString(
                   absl::StatusToStringMode::kWithNoExtraData)
            << "\nInstruction: " << instruction.ToShortString();
    return;
  }

  absl::StatusOr<GpuBackendConfig> gpu_config =
      instruction.backend_config<GpuBackendConfig>();

  ReificationCost* gemm_reification_cost = gpu_config->add_reification_cost();
  gemm_reification_cost->set_name("experimental-gemm-cost-model");
  gemm_reification_cost->set_end_to_end_cycles(
      absl::ToDoubleNanoseconds(*duration) * device_info.clock_rate_ghz());
  gemm_reification_cost->set_exec_time_us(
      absl::ToDoubleMicroseconds(*duration));

  VLOG(1) << "Adding GeMM fusion cost model estimate: "
          << gemm_reification_cost->DebugString()
          << "\nInstruction: " << instruction.ToString();

  CHECK_OK(instruction.set_backend_config(*gpu_config));
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

      gpu_performance_model.Get().RecordEstimatedRunTime(fusion_instr,
                                                         &cost_analysis_);

      RecordGemmCostModelEstimateIfApplicable(device_info_, *fusion_instr);
    }
  }
  return false;
}

}  // namespace gpu
}  // namespace xla
