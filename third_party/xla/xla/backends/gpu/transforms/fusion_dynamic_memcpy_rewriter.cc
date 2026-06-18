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

#include <memory>
#include <optional>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

bool HasDynamicSliceConfig(const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  return config.ok() && config->has_dynamic_slice_config();
}

bool IsSlicingInstruction(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kSlice ||
         instr->opcode() == HloOpcode::kDynamicSlice ||
         instr->opcode() == HloOpcode::kDynamicUpdateSlice;
}

bool IsSlicingInstructionCompatible(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kSlice) {
    return IsContiguousSlice(*instr) &&
           ShapeUtil::ByteStrides(instr->operand(0)->shape()).has_value();
  }

  return HasDynamicSliceConfig(instr);
}

bool AllSlicingInstructionsCompatible(const HloComputation* computation) {
  for (const HloInstruction* instr : computation->instructions()) {
    if (IsSlicingInstruction(instr) && !IsSlicingInstructionCompatible(instr)) {
      return false;
    }
  }
  return true;
}

bool IsBitcastOrReshape(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBitcast ||
         instr->opcode() == HloOpcode::kReshape;
}

HloInstruction* WalkThroughBitcastsAndReshapes(HloInstruction* instr) {
  while (IsBitcastOrReshape(instr)) {
    instr = instr->mutable_operand(0);
  }
  return instr;
}

struct MemcpyFusionCandidate {
  HloInstruction* slicing;
  HloInstruction* copy_operand;
};

// Detects DS/DUS-root memcpy fusions before they have a copy hero. `slicing` is
// the DS/DUS instruction carrying DynamicSliceConfig, and `copy_operand` is the
// value that would become the operand of the inserted copy hero.
std::optional<MemcpyFusionCandidate> FindMemcpyFusionCandidate(
    HloComputation* computation) {
  HloInstruction* root = computation->root_instruction();
  HloInstruction* ds_or_dus = WalkThroughBitcastsAndReshapes(root);

  if (!HasDynamicSliceConfig(ds_or_dus)) {
    return std::nullopt;
  }

  if (ds_or_dus->opcode() == HloOpcode::kDynamicSlice) {
    return MemcpyFusionCandidate{ds_or_dus, root};
  }

  if (ds_or_dus->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return MemcpyFusionCandidate{ds_or_dus, ds_or_dus->mutable_operand(1)};
  }

  return std::nullopt;
}

bool CanRewriteAsDynamicSliceFusion(const MemcpyFusionCandidate& candidate) {
  auto resolve_copy_hero_parameters = [](HloInstruction* operand) {
    std::unique_ptr<HloInstruction> copy = HloInstruction::CreateUnary(
        operand->shape(), HloOpcode::kCopy, operand);
    return DynamicSliceFusion::ResolveParameters(copy.get());
  };

  auto parameters = resolve_copy_hero_parameters(candidate.copy_operand);
  if (!parameters.ok()) {
    return false;
  }

  for (const DynamicSliceFusion::Parameter& parameter : *parameters) {
    if (parameter.slice_config.has_value()) {
      continue;
    }

    // Without DynamicSliceConfig, DynamicSliceFusion will pass the original
    // parameter buffer base address to the embedded copy thunk. This is only
    // correct for unsliced pass-through operands.
    if (ShapeUtil::ByteSizeOf(parameter.slice_shape) !=
        ShapeUtil::ByteSizeOf(parameter.parameter_shape)) {
      return false;
    }
  }

  if (candidate.slicing->opcode() == HloOpcode::kDynamicSlice) {
    return true;
  }

  return DynamicSliceFusion::ResolveResults(candidate.copy_operand).ok();
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

    std::optional<MemcpyFusionCandidate> candidate =
        FindMemcpyFusionCandidate(computation);
    if (!candidate.has_value() ||
        !AllSlicingInstructionsCompatible(computation) ||
        !CanRewriteAsDynamicSliceFusion(*candidate)) {
      continue;
    }

    ASSIGN_OR_RETURN(auto backend_config,
                     fusion->backend_config<GpuBackendConfig>());
    auto* fusion_config = backend_config.mutable_fusion_backend_config();
    fusion_config->set_kind(kCustomFusionKind);
    fusion_config->mutable_custom_fusion_config()->set_name(
        kDynamicSliceFusionConfigName);

    if (candidate->slicing->opcode() == HloOpcode::kDynamicSlice) {
      // DS case: insert copy after the root (which is DS or bitcast of DS).
      HloInstruction* copy =
          computation->AddInstruction(HloInstruction::CreateUnary(
              candidate->copy_operand->shape(), HloOpcode::kCopy,
              candidate->copy_operand));
      computation->set_root_instruction(copy);
    } else {
      // DUS case: insert copy before the DUS update operand (operand 1).
      HloInstruction* copy =
          computation->AddInstruction(HloInstruction::CreateUnary(
              candidate->copy_operand->shape(), HloOpcode::kCopy,
              candidate->copy_operand));
      RETURN_IF_ERROR(candidate->slicing->ReplaceOperandWith(1, copy));
    }

    // Set backend config to target dynamic slice custom fusion.
    RETURN_IF_ERROR(fusion->set_backend_config(backend_config));

    fusion->set_fusion_kind(HloInstruction::FusionKind::kCustom);
    has_changed = true;
  }

  return has_changed;
}

}  // namespace xla::gpu
