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
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/model/block_level_parameters.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla::gpu {

namespace {

using DimensionSemantics =
    ::xla::gpu::experimental::TilingSpace::DimensionSemantics;

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
      TF_ASSIGN_OR_RETURN(Tile tile_config, hlo->backend_config<Tile>());
      tile_mapping[hlo] =
          FlatTiling(tile_config.sizes().begin(), tile_config.sizes().end());
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

absl::StatusOr<llvm::SmallVector<int64_t>> GetTilingSpaceConcreteSizes(
    const xla::gpu::experimental::TilingSpace& tiling_space,
    const BlockLevelParameters& block_level_parameters) {
  if (block_level_parameters.output_tile_sizes.size() != 1) {
    return Internal(
        "Only single-result fusions are supported for now. Received %d "
        "roots.",
        block_level_parameters.output_tile_sizes.size());
  }
  const auto& parallel_tile_sizes = block_level_parameters.output_tile_sizes[0];
  if (int64_t num_parallel_dims = tiling_space.num_parallel_dimsensions();
      num_parallel_dims != parallel_tile_sizes.size()) {
    return Internal(
        "Number of parallel dimensions in the tiling space (%d) does not match "
        "the number of output tile sizes in the block level fusion config "
        "(%d).",
        num_parallel_dims, parallel_tile_sizes.size());
  }
  llvm::SmallVector<int64_t> tile_sizes;
  tile_sizes.reserve(tiling_space.dimensions().size());
  int parallel_dim_count = 0;
  for (const xla::gpu::experimental::TilingSpace::DimensionInfo& dim :
       tiling_space.dimensions()) {
    switch (dim.type) {
      case DimensionSemantics::kParallel:
        tile_sizes.push_back(parallel_tile_sizes[parallel_dim_count]);
        parallel_dim_count++;
        break;
      case DimensionSemantics::kSequential: {
        if (dim.hlo->has_backend_config()) {
          TF_ASSIGN_OR_RETURN(Tile config, dim.hlo->backend_config<Tile>());
          if (config.sizes_size() != 1) {
            return Internal(
                "Only single-reduction operations are supported "
                "dimension. Got %d tile sizes in backend config.",
                config.sizes_size());
          }
          tile_sizes.push_back(config.sizes(0));
        } else {
          VLOG(1) << "No backend_config set for HLO instruction of dimension "
                  << dim.ToString() << ". Using dimension size as tile size.";
          tile_sizes.push_back(dim.dimension_size);
        }
        break;
      }
      default:
        return Internal("Unsupported dimension type: %d", dim.type);
    }
  }
  return std::move(tile_sizes);
}

}  // namespace xla::gpu
