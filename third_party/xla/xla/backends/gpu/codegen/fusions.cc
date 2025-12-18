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
#include "xla/backends/gpu/codegen/fusions.h"

#include <memory>
#include <optional>
#include <utility>

#include "absl/strings/string_view.h"
#include "xla/backends/gpu/codegen/copy.h"
#include "xla/backends/gpu/codegen/cudnn.h"
#include "xla/backends/gpu/codegen/custom.h"
#include "xla/backends/gpu/codegen/emitters/concatenate.h"
#include "xla/backends/gpu/codegen/emitters/in_place_dynamic_update_slice.h"
#include "xla/backends/gpu/codegen/emitters/loop.h"
#include "xla/backends/gpu/codegen/emitters/reduction.h"
#include "xla/backends/gpu/codegen/emitters/scatter.h"
#include "xla/backends/gpu/codegen/emitters/transpose.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/sort.h"
#include "xla/backends/gpu/codegen/triton/fusion.h"
#include "xla/codegen/ir_emission_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

std::optional<std::unique_ptr<FusionInterface>> HloFusionInfo::GetCopyFusion()
    const {
  if (analysis().emitter_fusion_kind() ==
      HloFusionAnalysis::EmitterFusionKind::kDynamicMemcpy) {
    if (IsDynamicUpdateSliceFusion(analysis().fusion_spec()) &&
        !CanEmitDynamicUpdateSliceInPlace()) {
      // We currently only implement in-place DUSes as memcpys.
      return std::nullopt;
    }

    return std::make_unique<DynamicMemcpyFusion>(analysis());
  }

  for (const HloInstructionAdaptor& root_adaptor : analysis().fusion_roots()) {
    const HloInstruction* root = &root_adaptor.instruction();
    if (root->opcode() != HloOpcode::kCopy ||
        root->operand(0)->opcode() != HloOpcode::kParameter ||
        !LayoutUtil::Equal(root->operand(0)->shape().layout(),
                           root->shape().layout())) {
      return std::nullopt;
    }
  }

  return std::make_unique<MemcpyFusion>(analysis());
}

bool HloFusionInfo::CanEmitDynamicUpdateSliceInPlace() const {
  auto ret = CanEmitFusedDynamicUpdateSliceInPlace(analysis().fusion(),
                                                   buffer_assignment_, instr_);
  return ret.ok() && *ret;
}

std::unique_ptr<FusionInterface> GetFusionEmitter(
    const FusionInfo& fusion_info, mlir::MLIRContext* mlir_context) {
  const auto& analysis = fusion_info.analysis();
  const FusionBackendConfig& backend_config = analysis.fusion_backend_config();

  switch (analysis.emitter_fusion_kind()) {
    case HloFusionAnalysis::EmitterFusionKind::kCustomFusion: {
      const absl::string_view& config_name =
          backend_config.custom_fusion_config().name();
      if (config_name ==
              kDynamicSliceFusionWithStaticAddressComputationConfigName ||
          config_name ==
              kDynamicSliceFusionWithDynamicAddressComputationConfigName) {
        const HloFusionInfo* hlo_fusion_info =
            dynamic_cast<const HloFusionInfo*>(&fusion_info);
        return std::make_unique<DynamicSliceFusion>(
            analysis, hlo_fusion_info->GetCallGraph());
      }
      return std::make_unique<CustomFusion>();
    }
    case HloFusionAnalysis::EmitterFusionKind::kDynamicMemcpy:
    case HloFusionAnalysis::EmitterFusionKind::kLoop: {
      // Check for a memcpy fusion before checking if a DUS can be emitted in
      // place. DUS cmemcpy fusions can be emitted in place, but lowering them
      // to a memcpy is still better.
      if (auto copy_fusion = fusion_info.GetCopyFusion()) {
        return *std::move(copy_fusion);
      }
      if (IsDynamicUpdateSliceFusion(analysis.fusion_spec()) &&
          fusion_info.CanEmitDynamicUpdateSliceInPlace()) {
        return std::make_unique<InPlaceDynamicUpdateSliceFusion>(analysis);
      }
      return std::make_unique<LoopFusion>(analysis, mlir_context);
    }
    case HloFusionAnalysis::EmitterFusionKind::kReduction: {
      return CreateReductionFusion(analysis, mlir_context);
    }
    case HloFusionAnalysis::EmitterFusionKind::kScatter: {
      return CreateScatterFusion(analysis, mlir_context);
    }
    case HloFusionAnalysis::EmitterFusionKind::kTranspose: {
      return CreateTransposeFusion(analysis, mlir_context);
    }
    case HloFusionAnalysis::EmitterFusionKind::kConcatenate: {
      return std::make_unique<ConcatenateFusion>(analysis);
    }
    case HloFusionAnalysis::EmitterFusionKind::kSort: {
      return std::make_unique<SortFusion>();
    }
    case HloFusionAnalysis::EmitterFusionKind::kTriton:
      return std::make_unique<TritonFusion>(analysis);
    case HloFusionAnalysis::EmitterFusionKind::kCuDnn:
      return std::make_unique<CuDnnFusion>(analysis);
  }
}

}  // namespace gpu
}  // namespace xla
