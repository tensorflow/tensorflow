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

#include "xla/codegen/tiling/experimental/scheduling.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/codegen/tiling/experimental/tiled_hlo.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/util.h"

namespace xla::gpu::experimental {

absl::StatusOr<IndexingMap> Schedule(
    const TiledHloComputation& tiled_computation) {
  // Compute the block counts for each parallel dimension.
  llvm::SmallVector<int64_t, 4> block_counts;
  for (const auto& dimension : tiled_computation.tiling_space().dimensions()) {
    if (dimension.type != TilingSpace::DimensionSemantics::kParallel) {
      continue;
    }
    block_counts.push_back(
        CeilOfRatio(dimension.dimension_size, dimension.tile_size));
  }
  auto ctx = tiled_computation.GetMLIRContext();
  auto pid = CreateDimExpr(0, ctx);
  auto symbolic_map =
      SymbolicMap::Get(ctx, 1, 0, DelinearizeIndex(block_counts, pid, ctx));
  return IndexingMap(
      symbolic_map,
      /*dimensions=*/
      {IndexingMap::Variable{0, Product(block_counts) - 1, "pid"}},
      /*range_vars=*/{}, /*rt_vars=*/{});
}

}  // namespace xla::gpu::experimental
