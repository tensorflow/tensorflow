/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/codegen/tiling/tiled_hlo_schedule.h"

#include <cstdint>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<IndexingMap> MajorToMinorTiledHloSchedule::Schedule(
    const IndexingMap& tile_offsets_indexing, IterationSpace iteration_space,
    mlir::MLIRContext* ctx) const {
  if (iteration_space.size() != tile_offsets_indexing.GetDimVarsCount()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected iteration space to have exactly as many dimensions as there "
        "are parameters in the tile offsets indexing map, but iteration space "
        "has %d dimensions, and tile offsets indexing map has %d dimensions.",
        iteration_space.size(), tile_offsets_indexing.GetDimVarsCount()));
  }

  mlir::AffineExpr program_id = mlir::getAffineDimExpr(0, ctx);

  std::vector<int64_t> iteration_space_sizes;
  iteration_space_sizes.reserve(iteration_space.size());
  for (const auto& dim_info : iteration_space) {
    iteration_space_sizes.push_back(dim_info.dimension_size);
  }

  std::vector<mlir::AffineExpr> tile_exprs(
      tile_offsets_indexing.GetDimVarsCount(),
      mlir::getAffineConstantExpr(0, ctx));

  for (auto [dim_info, tile_expr] :
       llvm::zip(iteration_space,
                 DelinearizeIndex(iteration_space_sizes, program_id, ctx))) {
    if (dim_info.dimension_id >= tile_exprs.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Dimension id %d is out of bounds for tile offsets indexing map with "
          "%d dimensions. This can happen if ",
          dim_info.dimension_id, tile_exprs.size()));
    }
    tile_exprs[dim_info.dimension_id] = tile_expr;
  }
  std::vector<IndexingMap::Variable> dim_vars{
      {0, Product(iteration_space_sizes) - 1, "pid_0"}};
  IndexingMap program_id_to_output_dims{
      mlir::AffineMap::get(
          /*dimCount=*/1, /*symbolCount=*/0, tile_exprs, ctx),
      dim_vars, /*range_vars=*/{}, /*rt_vars=*/{}};
  auto scheduled_indexing =
      ComposeIndexingMaps(program_id_to_output_dims, tile_offsets_indexing);
  scheduled_indexing.Simplify();
  scheduled_indexing.RescaleSymbols();
  scheduled_indexing.RemoveUnusedSymbols();
  return scheduled_indexing;
}

}  // namespace xla
