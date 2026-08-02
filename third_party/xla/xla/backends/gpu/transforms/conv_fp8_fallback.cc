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

#include "xla/backends/gpu/transforms/conv_fp8_fallback.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/transforms/cudnn_fusion_compiler.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

using DevicelessFusionSupport = CuDnnFusionCompiler::DevicelessFusionSupport;

bool ShapeContainsF8(const Shape& shape) {
  bool contains_f8 = false;
  ShapeUtil::ForEachSubshape(
      shape, [&](const Shape& subshape, const ShapeIndex&) {
        contains_f8 |= subshape.IsArray() &&
                       primitive_util::IsF8Type(subshape.element_type());
      });
  return contains_f8;
}

Shape WithBf16ForF8(const Shape& shape) {
  Shape result = shape;
  ShapeUtil::ForEachMutableSubshape(
      &result, [](Shape* subshape, const ShapeIndex&) {
        if (subshape->IsArray() &&
            primitive_util::IsF8Type(subshape->element_type())) {
          subshape->set_element_type(BF16);
        }
      });
  return result;
}

// A candidate for the fallback: a __cudnn$fusion whose fused computation
// contains a convolution and at least one F8-typed shape.
bool IsFp8ConvFusion(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  auto gpu_config = instr.backend_config<GpuBackendConfig>();
  if (!gpu_config.ok() ||
      gpu_config->fusion_backend_config().kind() != kCuDnnFusionKind) {
    return false;
  }
  bool has_conv = false;
  bool has_f8 = false;
  for (const HloInstruction* fused : Cast<HloFusionInstruction>(&instr)
                                         ->fused_instructions_computation()
                                         ->instructions()) {
    has_conv |= fused->opcode() == HloOpcode::kConvolution;
    has_f8 |= ShapeContainsF8(fused->shape());
  }
  return has_conv && has_f8;
}

// Builds a clone of `fusion` in the same computation whose F8 shapes are
// BF16 (in the fused computation as well as on the fusion boundary: F8
// operands are wrapped in converts to BF16). The clone has no users; the
// caller either replaces `fusion` with it (ReplaceWithBf16Fusion) or removes
// it again (RemoveBf16Fusion).
absl::StatusOr<HloFusionInstruction*> BuildBf16Fusion(
    HloFusionInstruction* fusion) {
  HloComputation* fused_computation = fusion->fused_instructions_computation();
  HloComputation::Builder builder(
      absl::StrCat(fused_computation->name(), ".bf16"));
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> clone_map;
  for (const HloInstruction* instr :
       fused_computation->MakeInstructionPostOrder()) {
    HloInstruction* clone;
    if (instr->opcode() == HloOpcode::kConstant &&
        ShapeContainsF8(instr->shape())) {
      ASSIGN_OR_RETURN(
          Literal bf16_literal,
          Cast<HloConstantInstruction>(instr)->literal().Convert(BF16));
      clone = builder.AddInstruction(
          HloInstruction::CreateConstant(std::move(bf16_literal)));
    } else {
      std::vector<HloInstruction*> operands;
      operands.reserve(instr->operand_count());
      for (const HloInstruction* operand : instr->operands()) {
        operands.push_back(clone_map.at(operand));
      }
      clone = builder.AddInstruction(
          instr->CloneWithNewOperands(WithBf16ForF8(instr->shape()), operands));
    }
    clone_map[instr] = clone;
  }
  HloComputation* bf16_computation =
      fusion->GetModule()->AddEmbeddedComputation(
          builder.Build(clone_map.at(fused_computation->root_instruction())));

  HloComputation* parent = fusion->parent();
  std::vector<HloInstruction*> new_operands;
  new_operands.reserve(fusion->operand_count());
  for (HloInstruction* operand : fusion->operands()) {
    if (ShapeContainsF8(operand->shape())) {
      operand = parent->AddInstruction(HloInstruction::CreateConvert(
          WithBf16ForF8(operand->shape()), operand));
    }
    new_operands.push_back(operand);
  }
  HloInstruction* bf16_fusion =
      parent->AddInstruction(HloInstruction::CreateFusion(
          bf16_computation->root_instruction()->shape(), fusion->fusion_kind(),
          new_operands, bf16_computation));
  ASSIGN_OR_RETURN(GpuBackendConfig gpu_config,
                   fusion->backend_config<GpuBackendConfig>());
  RETURN_IF_ERROR(bf16_fusion->set_backend_config(gpu_config));
  return Cast<HloFusionInstruction>(bf16_fusion);
}

// Removes an unused BF16 fusion built by BuildBf16Fusion again, including
// its boundary converts and its fused computation.
absl::Status RemoveBf16Fusion(HloFusionInstruction* bf16_fusion) {
  HloComputation* bf16_computation =
      bf16_fusion->fused_instructions_computation();
  HloModule* module = bf16_fusion->GetModule();
  RETURN_IF_ERROR(
      bf16_fusion->parent()->RemoveInstructionAndUnusedOperands(bf16_fusion));
  return module->RemoveEmbeddedComputation(bf16_computation);
}

// Replaces `fusion` with `bf16_fusion`, converting BF16 (tuple-)outputs back
// to the original F8 types where they differ.
absl::Status ReplaceWithBf16Fusion(HloFusionInstruction* fusion,
                                   HloFusionInstruction* bf16_fusion) {
  HloComputation* parent = fusion->parent();
  const Shape& old_shape = fusion->shape();
  HloInstruction* replacement = bf16_fusion;
  if (old_shape.IsTuple()) {
    // cuDNN conv fusions produce flat tuples of arrays.
    std::vector<HloInstruction*> elements;
    elements.reserve(old_shape.tuple_shapes().size());
    for (int64_t i = 0; i < old_shape.tuple_shapes().size(); ++i) {
      HloInstruction* element =
          parent->AddInstruction(HloInstruction::CreateGetTupleElement(
              bf16_fusion->shape().tuple_shapes(i), bf16_fusion, i));
      if (!ShapeUtil::Equal(element->shape(), old_shape.tuple_shapes(i))) {
        element = parent->AddInstruction(
            HloInstruction::CreateConvert(old_shape.tuple_shapes(i), element));
      }
      elements.push_back(element);
    }
    replacement = parent->AddInstruction(HloInstruction::CreateTuple(elements));
  } else if (!ShapeUtil::Equal(bf16_fusion->shape(), old_shape)) {
    replacement = parent->AddInstruction(
        HloInstruction::CreateConvert(old_shape, bf16_fusion));
  }
  return parent->ReplaceInstruction(fusion, replacement);
}

}  // namespace

absl::StatusOr<HloFusionInstruction*> RewriteFp8FusionToBf16(
    HloFusionInstruction* fusion) {
  ASSIGN_OR_RETURN(HloFusionInstruction * bf16_fusion, BuildBf16Fusion(fusion));
  RETURN_IF_ERROR(ReplaceWithBf16Fusion(fusion, bf16_fusion));
  return bf16_fusion;
}

absl::StatusOr<bool> ConvFp8Fallback::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    // Collect candidates first; the rewrite mutates the computation.
    std::vector<HloFusionInstruction*> candidates;
    for (HloInstruction* instr : comp->instructions()) {
      if (IsFp8ConvFusion(*instr)) {
        candidates.push_back(Cast<HloFusionInstruction>(instr));
      }
    }
    for (HloFusionInstruction* fusion : candidates) {
      const DevicelessFusionSupport fp8_support =
          CuDnnFusionCompiler::SupportsFusionDeviceless(device_description_,
                                                        *fusion);
      if (fp8_support != DevicelessFusionSupport::kUnsupported) {
        // kSupported, or kUnknown (a cuDNN runtime too old for the target, a
        // graph the deviceless probe cannot model, or a cuDNN frontend
        // failure). Leave the fusion alone rather than rewrite on a guess.
        VLOG(1) << "Keeping FP8 conv fusion " << fusion->name()
                << " (deviceless probe verdict: "
                << (fp8_support == DevicelessFusionSupport::kSupported
                        ? "supported"
                        : "unknown")
                << ").";
        continue;
      }

      // No FP8 plans — check that the BF16 replacement has plans before
      // rewriting.
      ASSIGN_OR_RETURN(HloFusionInstruction * bf16_fusion,
                       BuildBf16Fusion(fusion));
      if (CuDnnFusionCompiler::SupportsFusionDeviceless(device_description_,
                                                        *bf16_fusion) !=
          DevicelessFusionSupport::kSupported) {
        LOG(WARNING) << "FP8 conv fusion " << fusion->name()
                     << " has no cuDNN plans for either FP8 or BF16.";
        RETURN_IF_ERROR(RemoveBf16Fusion(bf16_fusion));
        continue;
      }

      LOG(WARNING) << "FP8 conv fusion " << fusion->name()
                   << " has no cuDNN FP8 plans; rewriting to BF16. "
                   << "Try different convolution dimensions/group counts "
                   << "to regain FP8.";
      RETURN_IF_ERROR(ReplaceWithBf16Fusion(fusion, bf16_fusion));
      changed = true;
    }
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
