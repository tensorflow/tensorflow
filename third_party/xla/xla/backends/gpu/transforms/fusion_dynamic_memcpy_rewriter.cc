/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/fusion_dynamic_memcpy_rewriter.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"

namespace xla::gpu {
namespace {

HloInstruction* SkipOptionalBitcast(HloInstruction* instr) {
  while (instr->opcode() == HloOpcode::kBitcast) {
    instr = instr->mutable_operand(0);
  }
  return instr;
}

bool HasDynamicSliceConfig(const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  return config.ok() && config->has_dynamic_slice_config();
}

}  // namespace

absl::StatusOr<bool> FusionDynamicMemcpyRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool has_changed = false;

  for (HloComputation* computation : module->computations(execution_threads)) {
    if (!computation->IsFusionComputation()) {
      continue;
    }

    HloFusionInstruction* fusion =
        Cast<HloFusionInstruction>(computation->FusionInstruction());

    // Skip kCustom fusions: they already have a hero and were created by the
    // dynamic slice fusion rewriter. Their DUS roots also carry
    // DynamicSliceConfig but must not be re-processed.
    if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
      continue;
    }

    HloInstruction* fused_root = fusion->fused_expression_root();
    HloInstruction* ds_or_dus = SkipOptionalBitcast(fused_root);

    if (ds_or_dus->opcode() != HloOpcode::kDynamicSlice &&
        ds_or_dus->opcode() != HloOpcode::kDynamicUpdateSlice) {
      continue;
    }

    if (!HasDynamicSliceConfig(ds_or_dus)) {
      continue;
    }

    if (ds_or_dus->opcode() == HloOpcode::kDynamicSlice) {
      // DS case: insert copy after the root (which is DS or bitcast of DS).
      HloInstruction* copy =
          computation->AddInstruction(HloInstruction::CreateUnary(
              fused_root->shape(), HloOpcode::kCopy, fused_root));
      computation->set_root_instruction(copy);
    } else {
      // DUS case: insert copy before the DUS update operand (operand 1).
      HloInstruction* update = ds_or_dus->mutable_operand(1);
      HloInstruction* copy =
          computation->AddInstruction(HloInstruction::CreateUnary(
              update->shape(), HloOpcode::kCopy, update));
      RETURN_IF_ERROR(ds_or_dus->ReplaceOperandWith(1, copy));
    }

    // Set backend config to target DynamicSliceFusionV2.
    ASSIGN_OR_RETURN(auto backend_config,
                     fusion->backend_config<GpuBackendConfig>());
    auto* fusion_config = backend_config.mutable_fusion_backend_config();
    fusion_config->set_kind(kCustomFusionKind);
    fusion_config->mutable_custom_fusion_config()->set_name(
        kDynamicSliceFusionConfigName);
    RETURN_IF_ERROR(fusion->set_backend_config(backend_config));

    fusion->set_fusion_kind(HloInstruction::FusionKind::kCustom);
    has_changed = true;
  }

  return has_changed;
}

}  // namespace xla::gpu
