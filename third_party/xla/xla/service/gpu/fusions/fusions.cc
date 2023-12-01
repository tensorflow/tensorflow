/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/service/gpu/fusions/fusions.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/fusions/copy.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/in_place_dynamic_update_slice.h"
#include "xla/service/gpu/fusions/input_slices.h"
#include "xla/service/gpu/fusions/loop.h"
#include "xla/service/gpu/fusions/reduction.h"
#include "xla/service/gpu/fusions/transpose.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {
namespace {

bool IsParameterOrGteOfParameter(const HloInstruction* instr) {
  if (instr->opcode() == HloOpcode::kParameter) {
    return true;
  }
  if (instr->opcode() == HloOpcode::kGetTupleElement) {
    return IsParameterOrGteOfParameter(instr->operand(0));
  }
  return false;
}

bool IsSingleInstructionFusion(const HloFusionAnalysis& analysis) {
  return analysis.fusion_roots().size() == 1 &&
         absl::c_all_of(analysis.fusion_roots()[0]->operands(),
                        IsParameterOrGteOfParameter);
}

bool IsDynamicUpdateSliceFusion(const HloFusionAnalysis& analysis) {
  return absl::c_all_of(
      analysis.fusion_roots(), [](const HloInstruction* root) {
        return root->opcode() == HloOpcode::kDynamicUpdateSlice ||
               (root->opcode() == HloOpcode::kBitcast &&
                root->operand(0)->opcode() == HloOpcode::kDynamicUpdateSlice);
      });
}

}  // namespace

std::optional<std::unique_ptr<FusionInterface>> GetFusionEmitter(
    HloFusionAnalysis& analysis,
    absl::Span<const BufferAllocation* const> allocations,
    mlir::lmhlo::FusionOp fusion_op) {
  switch (analysis.GetEmitterFusionKind()) {
    case HloFusionAnalysis::EmitterFusionKind::kInputSlices:
      return std::make_unique<InputSlicesFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kLoop: {
      bool is_single = IsSingleInstructionFusion(analysis);
      if (!is_single && IsDynamicUpdateSliceFusion(analysis)) {
        if (allocations.empty() || fusion_op == nullptr) {
          return std::nullopt;
        }
        if (CanEmitFusedDynamicUpdateSliceInPlaceForGpu(fusion_op,
                                                        allocations)) {
          return std::make_unique<InPlaceDynamicUpdateSliceEmitter>(analysis);
        }
      }
      if (is_single && analysis.fusion_roots().size() == 1 &&
          analysis.fusion_roots().front()->opcode() == HloOpcode::kCopy) {
        if (!fusion_op) {
          return std::nullopt;
        }
        mlir::Value operand = GetHloOperands(fusion_op).front();
        mlir::Value output = GetHloOutputs(fusion_op).front();
        Shape operand_shape = GetShape(operand);
        Shape output_shape = GetShape(output);
        if (LayoutUtil::Equal(operand_shape.layout(), output_shape.layout()) &&
            GetAllocationSlice(operand, allocations).ok()) {
          return std::make_unique<MemcpyFusion>(operand, output);
        }
      }
      return std::make_unique<LoopFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
      return std::make_unique<ReductionFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kTranspose:
      return std::make_unique<TransposeFusion>(analysis);
    default:
      break;
  }
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
