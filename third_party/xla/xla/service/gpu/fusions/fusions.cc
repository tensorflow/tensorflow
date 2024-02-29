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
#include "xla/service/gpu/fusions/fusions.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/fusions/concatenate.h"
#include "xla/service/gpu/fusions/concatenate_mlir.h"
#include "xla/service/gpu/fusions/copy.h"
#include "xla/service/gpu/fusions/cudnn.h"
#include "xla/service/gpu/fusions/custom.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/in_place_dynamic_update_slice.h"
#include "xla/service/gpu/fusions/input_slices.h"
#include "xla/service/gpu/fusions/input_slices_mlir.h"
#include "xla/service/gpu/fusions/loop.h"
#include "xla/service/gpu/fusions/loop_mlir.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/reduction.h"
#include "xla/service/gpu/fusions/scatter.h"
#include "xla/service/gpu/fusions/transpose.h"
#include "xla/service/gpu/fusions/transpose_mlir.h"
#include "xla/service/gpu/fusions/triton.h"
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

std::optional<absl::StatusOr<std::unique_ptr<FusionInterface>>>
HloFusionInfo::GetCopyFusion() const {
  std::vector<BufferAllocation::Slice> src_buffers;
  for (auto* root : analysis().fusion_roots()) {
    if (root->opcode() != HloOpcode::kCopy ||
        root->operand(0)->opcode() != HloOpcode::kParameter ||
        !LayoutUtil::Equal(root->operand(0)->shape().layout(),
                           root->shape().layout())) {
      return std::nullopt;
    }

    const HloInstruction* src_instr =
        instr_->operands()[root->operand(0)->parameter_number()];
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        buffer_assignment_->GetUniqueSlice(src_instr, {}));
    src_buffers.push_back(slice);
  }

  std::vector<BufferAllocation::Slice> dst_buffers;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      instr_->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            buffer_assignment_->GetUniqueSlice(instr_, index));
        dst_buffers.push_back(slice);
        return absl::OkStatus();
      }));

  DCHECK(src_buffers.size() == dst_buffers.size());
  std::vector<mlir::Value> srcs;
  std::vector<mlir::Value> dsts;
  return std::make_unique<MemcpyFusion>(std::move(src_buffers),
                                        std::move(dst_buffers),
                                        /*srcs=*/std::vector<mlir::Value>(),
                                        /*dsts=*/std::vector<mlir::Value>());
}

bool HloFusionInfo::CanEmitDynamicUpdateSliceInPlace() const {
  auto ret = CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
      instr_, buffer_assignment_, analysis().fusion_roots());
  return ret.ok() && *ret;
}

absl::StatusOr<std::unique_ptr<FusionInterface>> GetFusionEmitter(
    const FusionInfo& fusion_info) {
  const auto& analysis = fusion_info.analysis();
  const FusionBackendConfig& backend_config = analysis.fusion_backend_config();

  bool enable_mlir_emitters = analysis.fusion_roots()
                                  .front()
                                  ->GetModule()
                                  ->config()
                                  .debug_options()
                                  .xla_gpu_enable_mlir_emitters();

  switch (analysis.GetEmitterFusionKind()) {
    case HloFusionAnalysis::EmitterFusionKind::kCustomFusion: {
      const auto& config = backend_config.custom_fusion_config();
      if (config.name() == "address_computation") {
        return std::make_unique<AddressComputationFusion>(analysis);
      }
      return std::make_unique<CustomFusion>();
    }
    case HloFusionAnalysis::EmitterFusionKind::kInputSlices:
      if (enable_mlir_emitters &&
          mlir_converter::IsHloConversionSupported(
              analysis.fusion(),
              fusion_info.analysis().device_info().gpu_compute_capability())) {
        return std::make_unique<MlirInputSlicesFusion>(analysis);
      }
      return std::make_unique<InputSlicesFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kLoop: {
      if (IsDynamicUpdateSliceFusion(analysis) &&
          fusion_info.CanEmitDynamicUpdateSliceInPlace()) {
        return std::make_unique<InPlaceDynamicUpdateSliceFusion>(analysis);
      }

      if (auto copy_fusion = fusion_info.GetCopyFusion()) {
        return *std::move(copy_fusion);
      }

      if (enable_mlir_emitters &&
          mlir_converter::IsHloConversionSupported(
              analysis.fusion(),
              fusion_info.analysis().device_info().gpu_compute_capability())) {
        return std::make_unique<MlirLoopFusion>(analysis);
      }
      return std::make_unique<LoopFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
      return std::make_unique<ReductionFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kScatter:
      return std::make_unique<ScatterFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kTranspose: {
      if (enable_mlir_emitters && MlirTransposeFusion::IsSupported(analysis)) {
        return std::make_unique<MlirTransposeFusion>(analysis);
      }
      return std::make_unique<TransposeFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate: {
      if (enable_mlir_emitters &&
          MlirConcatenateFusion::IsSupported(analysis)) {
        return std::make_unique<MlirConcatenateFusion>(analysis);
      }
      return std::make_unique<ConcatenateFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kTriton:
      return std::make_unique<TritonFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kCuDnn:
      return std::make_unique<CuDnnFusion>(analysis);
  }
}

}  // namespace gpu
}  // namespace xla
