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

#include "xla/backends/gpu/transforms/dynamic_slice_copy.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

bool IsBitcastOrReshape(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kBitcast ||
         instr->opcode() == HloOpcode::kReshape;
}

const HloInstruction* WalkThroughBitcastsAndReshapes(
    const HloInstruction* instr) {
  while (IsBitcastOrReshape(instr)) {
    instr = instr->operand(0);
  }
  return instr;
}

std::optional<DynamicSliceConfig> ExtractDynamicSliceConfig(
    const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  if (!config.ok() || !config->has_dynamic_slice_config()) {
    return std::nullopt;
  }
  return config->dynamic_slice_config();
}

bool HasDynamicSliceConfig(const HloInstruction* instr) {
  return ExtractDynamicSliceConfig(instr).has_value();
}

bool IsSlicingInstruction(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kSlice:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
      return true;
    default:
      return false;
  }
}

bool IsContiguousStaticSlice(const HloSliceInstruction* slice) {
  const Shape& orig = slice->operand(0)->shape();
  const Shape& sliced = slice->shape();
  std::optional<int64_t> sliced_dim;

  for (int64_t dim : orig.layout().minor_to_major()) {
    if (sliced_dim.has_value() && sliced.dimensions(dim) != 1) {
      return false;
    }

    if (sliced.dimensions(dim) < orig.dimensions(dim)) {
      if (slice->slice_strides(dim) != 1 && sliced.dimensions(dim) > 1) {
        return false;
      }
      sliced_dim = dim;
    }
  }
  return true;
}

bool IsSlicingInstructionCompatible(const HloInstruction* instr) {
  if (auto* slice = DynCast<HloSliceInstruction>(instr)) {
    return IsContiguousStaticSlice(slice) &&
           ShapeUtil::ByteStrides(instr->operand(0)->shape()).has_value();
  }

  return HasDynamicSliceConfig(instr);
}

bool AllSlicingInstructionsCompatible(const HloComputation* body) {
  for (const HloInstruction* instr : body->instructions()) {
    if (IsSlicingInstruction(instr) && !IsSlicingInstructionCompatible(instr)) {
      return false;
    }
  }
  return true;
}

std::optional<int64_t> StaticSliceByteOffset(const HloSliceInstruction* slice) {
  auto byte_strides = ShapeUtil::ByteStrides(slice->operand(0)->shape());
  if (!byte_strides.has_value()) {
    return std::nullopt;
  }

  int64_t byte_offset = 0;
  for (int64_t dim = 0; dim < slice->shape().dimensions().size(); ++dim) {
    byte_offset += slice->slice_starts(dim) * (*byte_strides)[dim];
  }
  return byte_offset;
}

struct DynamicSliceCopyCandidate {
  const HloInstruction* slicing;
  const HloInstruction* copy_operand;
};

std::optional<DynamicSliceCopyCandidate> FindDynamicSliceCopyCandidate(
    const HloInstruction* instr) {
  if (instr == nullptr || instr->opcode() != HloOpcode::kFusion) {
    return std::nullopt;
  }

  const auto* fusion = Cast<HloFusionInstruction>(instr);
  if (fusion->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    return std::nullopt;
  }

  const HloComputation* body = instr->fused_instructions_computation();
  const HloInstruction* root = body->root_instruction();
  const HloInstruction* ds_or_dus = WalkThroughBitcastsAndReshapes(root);

  if (!HasDynamicSliceConfig(ds_or_dus)) {
    return std::nullopt;
  }

  if (ds_or_dus->opcode() == HloOpcode::kDynamicSlice) {
    return DynamicSliceCopyCandidate{ds_or_dus, root};
  }

  if (ds_or_dus->opcode() == HloOpcode::kDynamicUpdateSlice) {
    return DynamicSliceCopyCandidate{ds_or_dus, ds_or_dus->operand(1)};
  }

  return std::nullopt;
}

bool CanUseAsUnslicedParameter(const DynamicSliceFusion::Parameter& parameter) {
  if (parameter.slice_config.has_value()) {
    return true;
  }

  // Without DynamicSliceConfig, DynamicSliceFusion will pass the original
  // parameter buffer base address to the embedded copy thunk. This is only
  // correct for unsliced pass-through operands.
  return ShapeUtil::ByteSizeOf(parameter.slice_shape) ==
         ShapeUtil::ByteSizeOf(parameter.parameter_shape);
}

bool HasCompatibleCopyShapes(const Shape& source,
                             const DynamicSliceFusion::Result& result) {
  return LayoutUtil::LayoutsInShapesEqual(source, result.update_shape,
                                          Layout::Equal().MinorToMajorOnly()) &&
         ShapeUtil::ByteSizeOf(source) ==
             ShapeUtil::ByteSizeOf(result.update_shape);
}

}  // namespace

absl::StatusOr<std::optional<DynamicSliceCopyFusion>>
AnalyzeDynamicSliceCopyFusion(const HloInstruction* instr) {
  std::optional<DynamicSliceCopyCandidate> candidate =
      FindDynamicSliceCopyCandidate(instr);
  if (!candidate.has_value()) {
    return std::nullopt;
  }

  if (!candidate->copy_operand->shape().IsArray()) {
    return std::nullopt;
  }

  const HloComputation* body = instr->fused_instructions_computation();
  if (!AllSlicingInstructionsCompatible(body)) {
    return std::nullopt;
  }

  absl::StatusOr<DynamicSliceFusion::Parameter> parameter =
      DynamicSliceFusion::ResolveParameter(candidate->copy_operand);
  if (!parameter.ok() || !CanUseAsUnslicedParameter(*parameter)) {
    return std::nullopt;
  }

  std::vector<DynamicSliceFusion::Result> results;
  if (candidate->slicing->opcode() == HloOpcode::kDynamicSlice) {
    const Shape& shape = candidate->copy_operand->shape();
    results.push_back(DynamicSliceFusion::Result{std::nullopt, 0, shape, shape,
                                                 std::nullopt, std::nullopt});
  } else {
    absl::StatusOr<std::vector<DynamicSliceFusion::Result>> resolved_results =
        DynamicSliceFusion::ResolveResults(candidate->copy_operand);
    if (!resolved_results.ok()) {
      return std::nullopt;
    }
    results = std::move(resolved_results).value();
  }

  if (results.size() != 1 ||
      !HasCompatibleCopyShapes(candidate->copy_operand->shape(), results[0])) {
    return std::nullopt;
  }

  std::vector<DynamicSliceFusion::Parameter> parameters;
  parameters.push_back(std::move(parameter).value());
  return DynamicSliceCopyFusion{candidate->copy_operand, std::move(parameters),
                                std::move(results)};
}

absl::StatusOr<std::optional<StaticSliceCopyFusion>>
AnalyzeStaticSliceCopyFusion(const HloFusionInstruction* instr) {
  if (instr->fusion_kind() == HloInstruction::FusionKind::kCustom) {
    return std::nullopt;
  }

  auto* slice = DynCast<HloSliceInstruction>(instr->fused_expression_root());
  if (slice == nullptr) {
    return std::nullopt;
  }

  auto* source = DynCast<HloParameterInstruction>(slice->operand(0));
  if (source == nullptr) {
    return std::nullopt;
  }

  const Shape& src_shape = source->shape();
  const Shape& dst_shape = instr->shape();
  if (!Layout::Equal().MinorToMajorOnly()(src_shape.layout(),
                                          dst_shape.layout()) ||
      !IsContiguousStaticSlice(slice)) {
    return std::nullopt;
  }

  std::optional<int64_t> source_byte_offset = StaticSliceByteOffset(slice);
  if (!source_byte_offset.has_value()) {
    return std::nullopt;
  }

  return StaticSliceCopyFusion{source->parameter_number(), slice->shape(),
                               *source_byte_offset};
}

bool IsCopyHeroDynamicSliceFusion(const HloInstruction* instr) {
  static constexpr char kDynamicSliceFusionV2ConfigName[] =
      "dynamic_slice_fusion";

  if (instr == nullptr || instr->opcode() != HloOpcode::kFusion ||
      instr->fusion_kind() != HloInstruction::FusionKind::kCustom) {
    return false;
  }

  absl::StatusOr<GpuBackendConfig> backend_config =
      instr->backend_config<GpuBackendConfig>();
  if (!backend_config.ok() || !backend_config->has_fusion_backend_config()) {
    return false;
  }

  const FusionBackendConfig& fusion_config =
      backend_config->fusion_backend_config();
  if (!fusion_config.has_custom_fusion_config() ||
      fusion_config.custom_fusion_config().name() !=
          kDynamicSliceFusionV2ConfigName) {
    return false;
  }

  const HloInstruction* hero =
      DynamicSliceFusion::FindHero(instr->fused_instructions_computation());
  return hero != nullptr && hero->opcode() == HloOpcode::kCopy;
}

bool IsDynamicSliceCopyFusion(const HloInstruction* instr) {
  absl::StatusOr<std::optional<DynamicSliceCopyFusion>> analysis =
      AnalyzeDynamicSliceCopyFusion(instr);
  return (analysis.ok() && analysis->has_value()) ||
         IsCopyHeroDynamicSliceFusion(instr);
}

}  // namespace xla::gpu
