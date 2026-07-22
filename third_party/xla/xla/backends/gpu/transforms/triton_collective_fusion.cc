/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/triton_collective_fusion.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/triton/collective_emitter.h"
#include "xla/backends/gpu/codegen/triton/support.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<bool> TritonCollectiveFusion::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  const se::GpuComputeCapability& gpu_version =
      device_description_.gpu_compute_capability();

  for (HloComputation* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    std::vector<HloInstruction*> ar_starts;
    for (HloInstruction* instr : computation->instructions()) {
      if (instr->opcode() == HloOpcode::kAllReduceStart) {
        ar_starts.push_back(instr);
      }
    }

    for (HloInstruction* ar_start : ar_starts) {
      if (ar_start->operand_count() != 1) {
        continue;
      }
      HloInstruction* operand = ar_start->mutable_operand(0);
      if (operand->opcode() != HloOpcode::kFusion) {
        continue;
      }
      HloInstruction* gemm_fusion = operand;
      if (!IsTritonGemm(*gemm_fusion)) {
        continue;
      }

      if (!IsTritonSupportedInstruction(*ar_start, gpu_version).IsAllowed()) {
        VLOG(3) << "All-reduce not supported by Triton: "
                << ar_start->ToString();
        continue;
      }

      VLOG(2) << "Fusing " << ar_start->name() << " into "
              << gemm_fusion->name();

      HloComputation* fused_comp =
          gemm_fusion->fused_instructions_computation();
      HloInstruction* old_root = fused_comp->root_instruction();

      HloInstruction* cloned_ar_start = fused_comp->AddInstruction(
          ar_start->CloneWithNewOperands(ar_start->shape(), {old_root}));

      fused_comp->set_root_instruction(cloned_ar_start);

      Shape old_shape = gemm_fusion->shape();
      *gemm_fusion->mutable_shape() = ar_start->shape();

      auto* fusion_instr = Cast<HloFusionInstruction>(gemm_fusion);
      auto status_or_supported = TrySetGpuBackendConfigForCollective(
          device_description_, fusion_instr);
      bool supported = false;
      if (status_or_supported.ok()) {
        supported = *status_or_supported;
      } else {
        VLOG(2) << "Triton collective fusion failed to set config: "
                << status_or_supported.status().message();
      }

      if (supported) {
        TF_RETURN_IF_ERROR(ar_start->ReplaceAllUsesWith(gemm_fusion));
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar_start));
        changed = true;
      } else {
        fused_comp->set_root_instruction(old_root);
        TF_RETURN_IF_ERROR(fused_comp->RemoveInstruction(cloned_ar_start));
        *gemm_fusion->mutable_shape() = old_shape;
        VLOG(2) << "Rollback fusion of " << ar_start->name() << " into "
                << gemm_fusion->name()
                << " because collective config could not be set.";
      }
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
