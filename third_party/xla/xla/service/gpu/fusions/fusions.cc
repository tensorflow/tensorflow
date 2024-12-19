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

#include "absl/algorithm/container.h"
#include "absl/strings/match.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/fusions/concatenate_mlir.h"
#include "xla/service/gpu/fusions/copy.h"
#include "xla/service/gpu/fusions/cudnn.h"
#include "xla/service/gpu/fusions/custom.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/fusions/in_place_dynamic_update_slice_mlir.h"
#include "xla/service/gpu/fusions/input_slices_mlir.h"
#include "xla/service/gpu/fusions/loop_mlir.h"
#include "xla/service/gpu/fusions/reduction_mlir.h"
#include "xla/service/gpu/fusions/scatter_mlir.h"
#include "xla/service/gpu/fusions/transpose_mlir.h"
#include "xla/service/gpu/fusions/triton.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
namespace {

bool IsDynamicUpdateSliceFusion(const HloFusionAnalysis& analysis) {
  return absl::c_all_of(
      analysis.fusion_roots(), [](const HloInstructionAdaptor& root) {
        return root.opcode() == HloOpcode::kDynamicUpdateSlice ||
               (root.opcode() == HloOpcode::kBitcast &&
                root.GetOperand(0).opcode() == HloOpcode::kDynamicUpdateSlice);
      });
}

}  // namespace

std::optional<std::unique_ptr<FusionInterface>> HloFusionInfo::GetCopyFusion()
    const {
  for (const HloInstructionAdaptor& root_adaptor : analysis().fusion_roots()) {
    const HloInstruction* root = &root_adaptor.instruction();
    if (root->opcode() != HloOpcode::kCopy ||
        root->operand(0)->opcode() != HloOpcode::kParameter ||
        !LayoutUtil::Equal(root->operand(0)->shape().layout(),
                           root->shape().layout())) {
      return std::nullopt;
    }
  }

  return std::make_unique<MemcpyFusion>(analysis(), buffer_assignment_);
}

bool HloFusionInfo::CanEmitDynamicUpdateSliceInPlace() const {
  auto ret = CanEmitFusedDynamicUpdateSliceInPlaceForGpu(
      analysis().fusion(),
      [this](const HloInstruction* instruction, const ShapeIndex& index) {
        return GetAllocationSlice(*buffer_assignment_, instruction, index);
      },
      instr_);
  return ret.ok() && *ret;
}

std::unique_ptr<FusionInterface> GetFusionEmitter(
    const FusionInfo& fusion_info) {
  const auto& analysis = fusion_info.analysis();
  const FusionBackendConfig& backend_config = analysis.fusion_backend_config();

  switch (analysis.GetEmitterFusionKind()) {
    case HloFusionAnalysis::EmitterFusionKind::kCustomFusion: {
      const auto& config = backend_config.custom_fusion_config();
      if (absl::StrContains(config.name(), "address_computation")) {
        return std::make_unique<DynamicSliceFusion>(analysis);
      }
      return std::make_unique<CustomFusion>();
    }
    case HloFusionAnalysis::EmitterFusionKind::kInputSlices:
      return std::make_unique<MlirInputSlicesFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kLoop: {
      if (IsDynamicUpdateSliceFusion(analysis) &&
          fusion_info.CanEmitDynamicUpdateSliceInPlace()) {
        return std::make_unique<MlirInPlaceDynamicUpdateSliceFusion>(analysis);
      }
      if (auto copy_fusion = fusion_info.GetCopyFusion()) {
        return *std::move(copy_fusion);
      }
      return std::make_unique<MlirLoopFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kReduction:
      return CreateMlirReductionFusion(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kScatter: {
      return CreateMlirScatterFusion(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kTranspose: {
      return std::make_unique<MlirTransposeFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate: {
      return std::make_unique<MlirConcatenateFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kTriton:
      return std::make_unique<TritonFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kCuDnn:
      return std::make_unique<CuDnnFusion>(analysis);
  }
}

}  // namespace gpu
}  // namespace xla
