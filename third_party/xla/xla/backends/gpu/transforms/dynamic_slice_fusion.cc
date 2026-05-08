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

#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

const HloInstruction* DynamicSliceFusion::FindHero(const HloComputation* body) {
  for (const HloInstruction* instr : body->instructions()) {
    switch (instr->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kConstant:
      case HloOpcode::kSlice:
      case HloOpcode::kDynamicSlice:
      case HloOpcode::kDynamicUpdateSlice:
      case HloOpcode::kBitcast:
      case HloOpcode::kTuple:
      case HloOpcode::kGetTupleElement:
        continue;
      default:
        return instr;
    }
  }
  return nullptr;
}

namespace {

const HloInstruction* WalkThroughBitcasts(const HloInstruction* instr) {
  while (instr->opcode() == HloOpcode::kBitcast) {
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

std::optional<DynamicSliceConfig> ComputeStaticSliceConfig(
    const HloSliceInstruction* slice) {
  auto byte_strides = ShapeUtil::ByteStrides(slice->operand(0)->shape());
  if (!byte_strides.has_value()) {
    return std::nullopt;
  }
  int64_t byte_offset = 0;
  for (int64_t dim = 0; dim < slice->shape().dimensions_size(); ++dim) {
    byte_offset += slice->slice_starts(dim) * (*byte_strides)[dim];
  }
  DynamicSliceConfig config;
  config.set_byte_offset(byte_offset);
  config.set_byte_stride(0);
  return config;
}

// Resolves per-dimension offset info for a DS/DUS. Returns one entry per
// dimension: RuntimeOffset for fusion parameters, ConstantOffset for
// constants sunk into the fusion body.
std::vector<DynamicSliceFusion::Offset> ResolveOffsets(
    const HloInstruction* instr, int32_t first_offset_index) {
  std::vector<DynamicSliceFusion::Offset> offsets;
  offsets.reserve(instr->operand_count() - first_offset_index);
  for (int64_t i = first_offset_index; i < instr->operand_count(); ++i) {
    int64_t dim = i - first_offset_index;
    const HloInstruction* idx = WalkThroughBitcasts(instr->operand(i));
    auto* idx_param = DynCast<HloParameterInstruction>(idx);
    if (idx_param != nullptr) {
      offsets.push_back(DynamicSliceFusion::RuntimeOffset{
          idx_param->parameter_number(), dim});
    } else {
      offsets.push_back(DynamicSliceFusion::ConstantOffset{0, dim});
    }
  }
  return offsets;
}

// Resolves one result chain: walks from a hero output through bitcasts to a
// DUS, extracts the config, and finds the target parameter.
static absl::StatusOr<std::optional<DynamicSliceFusion::Result>>
ResolveOneResultChain(const HloInstruction* start, const Shape& hero_shape,
                      int64_t result_number) {
  const HloInstruction* walk = start;
  while (walk->opcode() == HloOpcode::kBitcast) {
    if (walk->user_count() != 1) {
      break;
    }
    walk = walk->users().front();
  }

  auto* dus = DynCast<HloDynamicUpdateSliceInstruction>(walk);
  if (dus == nullptr) {
    return std::nullopt;
  }

  std::optional<DynamicSliceConfig> config = ExtractDynamicSliceConfig(dus);

  const HloInstruction* target = WalkThroughBitcasts(dus->operand(0));
  auto* target_param = DynCast<HloParameterInstruction>(target);
  if (target_param == nullptr) {
    return Internal(
        "DynamicSliceFusionV2: DUS target must be a fusion parameter, got %s",
        target->ToString());
  }

  return DynamicSliceFusion::Result{
      std::optional<int64_t>(target_param->parameter_number()),
      result_number,
      dus->operand(0)->shape(),
      dus->operand(1)->shape(),
      config,
      std::optional(ResolveOffsets(dus, 2)),
  };
}

// Walks a GTE chain from `instr` following the given ShapeIndex path. For
// example, ShapeIndex{0,1} means: find the GTE with index 0 among instr's
// users, then find the GTE with index 1 among that GTE's users. Returns
// nullptr if the chain does not exist.
const HloInstruction* WalkGteChain(const HloInstruction* instr,
                                   const ShapeIndex& index) {
  const HloInstruction* current = instr;
  for (int64_t i = 0; i < index.size(); ++i) {
    const HloInstruction* found = nullptr;
    for (const HloInstruction* user : current->users()) {
      auto* gte = DynCast<HloGetTupleElementInstruction>(user);
      if (gte != nullptr && gte->tuple_index() == index[i]) {
        found = gte;
        break;
      }
    }
    if (found == nullptr) {
      return nullptr;
    }
    current = found;
  }
  return current;
}

}  // namespace

absl::StatusOr<std::vector<DynamicSliceFusion::Parameter>>
DynamicSliceFusion::ResolveParameters(const HloInstruction* hero) {
  std::vector<DynamicSliceFusion::Parameter> result;
  result.reserve(hero->operand_count());

  for (const HloInstruction* operand : hero->operands()) {
    const HloInstruction* walk = WalkThroughBitcasts(operand);

    std::optional<DynamicSliceConfig> config;
    std::optional<std::vector<DynamicSliceFusion::Offset>> offsets;
    const HloInstruction* source = walk;
    Shape slice_shape = operand->shape();

    if (auto* ds = DynCast<HloDynamicSliceInstruction>(walk)) {
      config = ExtractDynamicSliceConfig(ds);
      offsets = ResolveOffsets(ds, 1);
      slice_shape = ds->shape();
      source = ds->operand(0);
    } else if (auto* slice = DynCast<HloSliceInstruction>(walk)) {
      config = ComputeStaticSliceConfig(slice);
      slice_shape = slice->shape();
      source = slice->operand(0);
    }

    source = WalkThroughBitcasts(source);
    auto* parameter = DynCast<HloParameterInstruction>(source);
    if (parameter == nullptr) {
      return Internal(
          "DynamicSliceFusionV2: expected fusion parameter backing hero "
          "operand, got %s",
          source->ToString());
    }

    result.push_back(DynamicSliceFusion::Parameter{
        parameter->parameter_number(),
        source->shape(),
        slice_shape,
        config,
        std::move(offsets),
    });
  }

  return result;
}

absl::StatusOr<std::vector<DynamicSliceFusion::Result>>
DynamicSliceFusion::ResolveResults(const HloInstruction* hero) {
  if (hero->shape().IsTuple()) {
    auto leaves = ShapeUtil::GetLeafShapes(hero->shape());
    int64_t n = leaves.size();
    std::vector<DynamicSliceFusion::Result> results(n);

    for (int64_t i = 0; i < n; ++i) {
      const Shape& leaf = leaves[i].shape;
      results[i] = Result{std::nullopt, i, leaf, leaf};
    }

    for (int64_t i = 0; i < n; ++i) {
      const ShapeIndex& index = leaves[i].index;
      const HloInstruction* leaf_gte = WalkGteChain(hero, index);
      if (leaf_gte == nullptr) {
        continue;
      }

      for (const HloInstruction* user : leaf_gte->users()) {
        TF_ASSIGN_OR_RETURN(auto rs,
                            ResolveOneResultChain(user, leaves[i].shape, i));
        if (rs.has_value()) {
          results[i] = *std::move(rs);
        }
      }
    }
    return results;
  }

  // Non-tuple hero: single result.
  for (const HloInstruction* user : hero->users()) {
    TF_ASSIGN_OR_RETURN(auto rs, ResolveOneResultChain(user, hero->shape(), 0));
    if (rs.has_value()) {
      return std::vector{*std::move(rs)};
    }
  }

  return std::vector{Result{std::nullopt, 0, hero->shape(), hero->shape()}};
}

}  // namespace xla::gpu
