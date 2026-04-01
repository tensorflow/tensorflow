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

#include "xla/service/gpu/model/tiling_from_block_parameters.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/tsl/platform/statusor.h"

namespace xla::gpu {

namespace {
absl::StatusOr<FlatTiling> DotTilingParameters(
    const HloInstruction* hlo,
    const SymbolicTileAnalysis& symbolic_tile_analysis) {
  if (absl::c_all_of(hlo->operands(), [](const HloInstruction* operand) {
        return operand->opcode() != HloOpcode::kFusion;
      })) {
    TF_ASSIGN_OR_RETURN(Tile tile_config, hlo->backend_config<Tile>());
    return FlatTiling(tile_config.sizes().begin(), tile_config.sizes().end());
  }
  const HloInstruction* lhs = hlo->operand(0);
  // When encountering a `dot`, we always expect its operands to be nests.
  auto backend_config = lhs->backend_config<GpuBackendConfig>();
  if (!backend_config.ok() || !backend_config->fusion_backend_config()
                                   .has_block_level_fusion_config()) {
    return absl::FailedPreconditionError(
        absl::StrCat("No block_level_fusion_config in ", lhs->ToString()));
  }
  std::vector<int64_t> lhs_output_tile_sizes =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          backend_config->fusion_backend_config().block_level_fusion_config())
          .output_tile_sizes.front();

  FlatTiling dot_tiling_parameters;
  dot_tiling_parameters.reserve(
      hlo->dot_dimension_numbers().lhs_contracting_dimensions().size());
  for (int64_t contracting_dim_id :
       hlo->dot_dimension_numbers().lhs_contracting_dimensions()) {
    if (contracting_dim_id >= lhs_output_tile_sizes.size()) {
      return absl::FailedPreconditionError(
          absl::StrCat("Output tile sizes index ", contracting_dim_id,
                       " is out of bounds for ", lhs->ToString()));
    }
    dot_tiling_parameters.push_back(lhs_output_tile_sizes[contracting_dim_id]);
  }
  return dot_tiling_parameters;
}
}  // namespace

absl::StatusOr<Tiling> TilingFromAnnotatedFusion(
    const SymbolicTileAnalysis& symbolic_tile_analysis,
    const BlockLevelParameters& block_level_parameters) {
  Tiling::TileMapping tile_mapping;
  int64_t real_root_index = symbolic_tile_analysis.real_root_index();
  const HloInstruction* real_root =
      symbolic_tile_analysis.GetRoots()[real_root_index];

  for (const auto& [hlo, num_tiling_parameters] :
       symbolic_tile_analysis.GetTilingSpecification().parameter_mapping()) {
    // TODO(b/419026602): handle reductions.
    if (hlo->opcode() == HloOpcode::kDot ||
        hlo->opcode() == HloOpcode::kScaledDot) {
      TF_ASSIGN_OR_RETURN(tile_mapping[hlo],
                          DotTilingParameters(hlo, symbolic_tile_analysis));
    }

    // TODO(b/390559452): this should change for generalized multi-output
    // fusions.
    if (hlo == real_root) {
      if (real_root_index >= block_level_parameters.output_tile_sizes.size()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Output tile sizes index ", real_root_index,
            " is out of bounds for block level fusion config: ",
            block_level_parameters.ToBlockLevelFusionConfig().DebugString()));
      }
      absl::Span<const int64_t> output_tile_sizes =
          block_level_parameters.output_tile_sizes[real_root_index];
      tile_mapping[hlo].insert(tile_mapping[hlo].end(),
                               output_tile_sizes.begin(),
                               output_tile_sizes.end());
    }
  }

  return Tiling(std::move(tile_mapping));
}

}  // namespace xla::gpu
