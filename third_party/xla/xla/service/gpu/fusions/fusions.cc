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
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
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
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {
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

bool IsDynamicUpdateSliceFusion(const HloFusionAnalysis& analysis) {
  return absl::c_all_of(
      analysis.fusion_roots(), [](const HloInstruction* root) {
        return root->opcode() == HloOpcode::kDynamicUpdateSlice ||
               (root->opcode() == HloOpcode::kBitcast &&
                root->operand(0)->opcode() == HloOpcode::kDynamicUpdateSlice);
      });
}

}  // namespace

StatusOr<std::optional<std::unique_ptr<FusionInterface>>> GetCopyFusionImpl(
    HloFusionAnalysis& analysis, LmhloFusionInfo fusion_info) {
  mlir::lmhlo::FusionOp fusion_op = fusion_info.fusion_op;
  absl::Span<const BufferAllocation* const> allocations =
      fusion_info.allocations;

  auto params = GetHloOperands(fusion_op);
  auto outputs = GetHloOutputs(fusion_op);
  std::vector<mlir::Value> srcs;
  srcs.reserve(outputs.size());

  for (auto* root : analysis.fusion_roots()) {
    if (root->opcode() != HloOpcode::kCopy ||
        root->operand(0)->opcode() != HloOpcode::kParameter ||
        !LayoutUtil::Equal(root->operand(0)->shape().layout(),
                           root->shape().layout())) {
      return std::nullopt;
    }

    mlir::Value src = params[root->operand(0)->parameter_number()];
    if (!GetAllocationSlice(src, allocations).ok()) return std::nullopt;

    srcs.emplace_back(src);
  }

  auto dsts = std::vector<mlir::Value>(outputs.begin(), outputs.end());
  DCHECK(srcs.size() == dsts.size());
  std::vector<BufferAllocation::Slice> src_buffers;
  std::vector<BufferAllocation::Slice> dst_buffers;
  for (int i = 0; i < srcs.size(); ++i) {
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                        GetAllocationSlice(srcs[i], allocations));
    src_buffers.push_back(src_buffer);
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                        GetAllocationSlice(dsts[i], allocations));
    dst_buffers.push_back(dst_buffer);
  }

  return std::make_unique<MemcpyFusion>(std::move(src_buffers),
                                        std::move(dst_buffers), std::move(srcs),
                                        std::move(dsts));
}

StatusOr<std::optional<std::unique_ptr<FusionInterface>>> GetCopyFusionImpl(
    HloFusionAnalysis& analysis, HloFusionInfo fusion_info) {
  const HloFusionInstruction* fusion = fusion_info.instr;
  const BufferAssignment* buffer_assignment = fusion_info.buffer_assignment;

  std::vector<BufferAllocation::Slice> src_buffers;
  for (auto* root : analysis.fusion_roots()) {
    if (root->opcode() != HloOpcode::kCopy ||
        root->operand(0)->opcode() != HloOpcode::kParameter ||
        !LayoutUtil::Equal(root->operand(0)->shape().layout(),
                           root->shape().layout())) {
      return std::nullopt;
    }

    const HloInstruction* src_instr =
        fusion->operands()[root->operand(0)->parameter_number()];
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        buffer_assignment->GetUniqueSlice(src_instr, {}));
    src_buffers.push_back(slice);
  }

  std::vector<BufferAllocation::Slice> dst_buffers;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      fusion->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return OkStatus();
        }
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            buffer_assignment->GetUniqueSlice(fusion, index));
        dst_buffers.push_back(slice);
        return OkStatus();
      }));

  DCHECK(src_buffers.size() == dst_buffers.size());
  std::vector<mlir::Value> srcs;
  std::vector<mlir::Value> dsts;
  return std::make_unique<MemcpyFusion>(std::move(src_buffers),
                                        std::move(dst_buffers),
                                        /*srcs=*/std::vector<mlir::Value>(),
                                        /*dsts=*/std::vector<mlir::Value>());
}

StatusOr<std::optional<std::unique_ptr<FusionInterface>>> GetCopyFusion(
    HloFusionAnalysis& analysis,
    std::variant<HloFusionInfo, LmhloFusionInfo> fusion_info) {
  if (std::holds_alternative<HloFusionInfo>(fusion_info)) {
    return GetCopyFusionImpl(analysis, std::get<HloFusionInfo>(fusion_info));
  } else {
    return GetCopyFusionImpl(analysis, std::get<LmhloFusionInfo>(fusion_info));
  }
}

}  // namespace

StatusOr<std::optional<std::unique_ptr<FusionInterface>>> GetFusionEmitter(
    HloFusionAnalysis& analysis,
    std::variant<HloFusionInfo, LmhloFusionInfo> fusion_info) {
  switch (analysis.GetEmitterFusionKind()) {
    case HloFusionAnalysis::EmitterFusionKind::kInputSlices:
      return std::make_unique<InputSlicesFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kLoop: {
      if (IsDynamicUpdateSliceFusion(analysis)) {
        if (!std::holds_alternative<LmhloFusionInfo>(fusion_info)) {
          return std::nullopt;
        }
        auto lmhlo_fusion_info = std::get<LmhloFusionInfo>(fusion_info);
        absl::Span<const BufferAllocation* const> allocations =
            lmhlo_fusion_info.allocations;
        mlir::lmhlo::FusionOp fusion_op = lmhlo_fusion_info.fusion_op;
        if (CanEmitFusedDynamicUpdateSliceInPlaceForGpu(fusion_op,
                                                        allocations)) {
          return std::make_unique<InPlaceDynamicUpdateSliceEmitter>(analysis);
        }
      }
      TF_ASSIGN_OR_RETURN(
          std::optional<std::unique_ptr<FusionInterface>> copy_fusion,
          GetCopyFusion(analysis, fusion_info));
      if (copy_fusion.has_value()) {
        return copy_fusion;
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
