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
#include "tensorflow/compiler/xla/service/gpu/fusions/fusions.h"

#include <memory>
#include <optional>

#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/gpu/fusions/copy.h"
#include "tensorflow/compiler/xla/service/gpu/fusions/in_place_dynamic_update_slice.h"
#include "tensorflow/compiler/xla/service/gpu/fusions/input_slices.h"
#include "tensorflow/compiler/xla/service/gpu/fusions/loop.h"
#include "tensorflow/compiler/xla/service/gpu/fusions/reduction.h"
#include "tensorflow/compiler/xla/service/gpu/fusions/transpose.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_fusion_analysis.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

namespace xla {
namespace gpu {
namespace {

bool IsSingleInstructionFusion(mlir::lmhlo::FusionOp fusion) {
  bool seen_instruction = false;
  for (mlir::Operation& instr : fusion.getRegion().front()) {
    if (mlir::isa<mlir::lmhlo::TerminatorOp, mlir::mhlo::ReturnOp,
                  mlir::bufferization::ToTensorOp, mlir::memref::TensorStoreOp>(
            &instr)) {
      continue;
    }
    if (seen_instruction) return false;
    seen_instruction = true;
  }
  return seen_instruction;
}

}  // namespace

std::optional<std::unique_ptr<FusionInterface>> GetFusionEmitter(
    HloFusionAnalysis& analysis, IrEmitterContext& ir_emitter_context,
    ElementalIrEmitter& elemental_emitter, mlir::lmhlo::FusionOp fusion_op,
    const HloFusionInstruction& fusion) {
  switch (analysis.GetEmitterFusionKind()) {
    case HloFusionAnalysis::EmitterFusionKind::kInputSlices:
      return std::make_unique<InputSlicesFusion>(
          ir_emitter_context, elemental_emitter, fusion_op, fusion, analysis);
    case HloFusionAnalysis::EmitterFusionKind::kLoop: {
      bool is_single = IsSingleInstructionFusion(fusion_op);
      if (!is_single && CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
                            fusion_op, ir_emitter_context.allocations())) {
        return std::make_unique<InPlaceDynamicUpdateSliceEmitter>(
            ir_emitter_context, elemental_emitter, fusion_op, fusion);
      }
      if (is_single &&
          fusion.fused_expression_root()->opcode() == HloOpcode::kCopy) {
        mlir::Value operand = GetHloOperands(fusion_op).front();
        mlir::Value output = GetHloOutputs(fusion_op).front();
        Shape operand_shape = GetShape(operand);
        Shape output_shape = GetShape(output);
        if (LayoutUtil::Equal(operand_shape.layout(), output_shape.layout()) &&
            GetAllocationSlice(operand, ir_emitter_context.allocations())
                .ok()) {
          return std::make_unique<MemcpyFusion>(ir_emitter_context, fusion_op,
                                                operand, output);
        }
      }
      return std::make_unique<LoopFusion>(ir_emitter_context, elemental_emitter,
                                          fusion_op, fusion, analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
      return std::make_unique<ReductionFusion>(
          ir_emitter_context, elemental_emitter, fusion_op, fusion, analysis);
    case HloFusionAnalysis::EmitterFusionKind::kTranspose:
      return std::make_unique<TransposeFusion>(
          ir_emitter_context, elemental_emitter, fusion_op, fusion, analysis);
    default:
      break;
  }
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
