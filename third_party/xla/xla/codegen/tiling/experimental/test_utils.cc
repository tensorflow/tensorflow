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

#include "xla/codegen/tiling/experimental/test_utils.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/index_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu::experimental {

using ::llvm::SmallVector;
using ::mlir::MLIRContext;

Tile GetTestTile(const TilingSpace& tiling_space,
                 absl::Span<const int64_t> shape) {
  MLIRContext* mlir_context = tiling_space.mlir_context();
  CHECK(mlir_context != nullptr);
  int64_t rank = shape.size();
  SmallVector<DimTile> dim_tiles;
  dim_tiles.reserve(rank);
  for (auto [index, dim] : llvm::enumerate(shape)) {
    auto tid = CreateDimExpr(index, mlir_context);
    auto ts =
        CreateSymbolExpr(index, tiling_space.num_dimensions(), mlir_context);
    dim_tiles.push_back(DimTile{tid * ts, ts,
                                CreateSymbolicConstant(index + 1, mlir_context),
                                CreateSymbolicConstant(dim, mlir_context)});
  }
  return Tile{tiling_space, std::move(dim_tiles)};
}

bool BumpCoordinates(absl::Span<const int64_t> limits,
                     absl::Span<int64_t> coordinates) {
  for (int64_t i = coordinates.size() - 1; i >= 0; --i) {
    if (coordinates[i] + 1 < limits[i]) {
      coordinates[i]++;
      std::fill(coordinates.begin() + i + 1, coordinates.end(), 0);
      return true;
    }
  }
  return false;
}

absl::StatusOr<std::vector<int64_t>> EvaluateAccessedIndexesForTile(
    DimensionVector dim_values, const Tile& tile, const Shape& shape) {
  VLOG(2) << "EvaluateAccessedIndexesForTile, dim_values = "
          << absl::StrJoin(dim_values, ",");
  Tile simplified_tile = tile;
  simplified_tile.Simplify();

  size_t tile_rank = simplified_tile.dim_tiles().size();
  SmallVector<int64_t, 4> offsets, sizes, strides, upper_bounds;
  offsets.reserve(tile_rank);
  sizes.reserve(tile_rank);
  strides.reserve(tile_rank);
  upper_bounds.reserve(tile_rank);

  for (const DimTile& d : simplified_tile.dim_tiles()) {
    int64_t offset = d.offset.Evaluate(dim_values);
    int64_t size = d.size.Evaluate(dim_values);
    int64_t stride = d.stride.Evaluate(dim_values);
    int64_t upper_bound = d.upper_bound.Evaluate(dim_values);
    VLOG(3) << "offset: " << offset << ", size: " << size
            << ", stride: " << stride << ", upper_bound: " << upper_bound;
    offsets.push_back(offset);
    sizes.push_back(size);
    strides.push_back(stride);
    upper_bounds.push_back(upper_bound);
  }
  std::vector<int64_t> accessed_indexes;
  int64_t total_elements = Product(sizes);
  VLOG(2) << "tile area " << total_elements;
  accessed_indexes.reserve(total_elements);

  SmallVector<int64_t, 4> local_coords(tile_rank, 0);
  SmallVector<int64_t, 4> accessed_index(tile_rank, 0);
  if (total_elements > 0) {
    do {
      bool outside = false;
      for (int j = 0; j < tile_rank; ++j) {
        accessed_index[j] = offsets[j] + local_coords[j] * strides[j];
        outside = outside || (accessed_index[j] >= upper_bounds[j]);
      }
      if (!outside) {
        for (int j = 0; j < tile_rank; ++j) {
          if (accessed_index[j] < 0 ||
              accessed_index[j] >= shape.dimensions(j)) {
            return absl::InternalError(absl::StrCat(
                "Tile evaluated to index out of bounds, tried to access index ",
                accessed_index[j], " of dimension ", j, " in shape ",
                shape.ToString(), " tile.upper_bound=", upper_bounds[j]));
          }
        }
        int64_t x = IndexUtil::MultidimensionalIndexToLinearIndex(
            shape, accessed_index);
        accessed_indexes.push_back(x);
      }
    } while (BumpCoordinates(sizes, absl::MakeSpan(local_coords)));
  }
  return accessed_indexes;
}

absl::Status VerifyTileEquivalence(const Tile& tile_a, const Shape& shape_a,
                                   const Tile& tile_b, const Shape& shape_b,
                                   TilingSpace* tiling_space) {
  // TODO: b/532069302 - remove after updating tests.
  if (tiling_space->IsSymbolic()) {
    return absl::OkStatus();
  }
  CHECK(tiling_space->num_parallel_dimensions() ==
        tiling_space->num_dimensions());
  VLOG(1) << tiling_space->ToString() << "\n"
          << shape_a.ToString() << " -> " << shape_b.ToString();
  VLOG(1) << "From tile: " << tile_a.ToString();
  VLOG(1) << "To tile:   " << tile_b.ToString();
  // Define grid size based on shape.
  SmallVector<int64_t, 4> grid_size;
  grid_size.reserve(tiling_space->num_dimensions());
  for (const TilingSpace::DimensionInfo& dim : tiling_space->dimensions()) {
    int64_t b = CeilOfRatio(dim.dimension_size, *dim.tile_size);
    grid_size.emplace_back(b);
  }
  Shape grid_shape = ShapeUtil::MakeShapeWithDescendingLayout(F32, grid_size);
  int64_t total_blocks = Product(grid_size);
  VLOG(1) << "grid size: " << absl::StrJoin(grid_size, ",") << " = "
          << total_blocks;
  for (int64_t pid = 0; pid < total_blocks; ++pid) {
    DimensionVector dim_values =
        IndexUtil::LinearIndexToMultidimensionalIndex(grid_shape, pid);
    CHECK(dim_values.size() == tiling_space->num_dimensions());
    VLOG(2) << "evaluating indexes for pid: " << pid << " -> dims "
            << absl::StrJoin(dim_values, ",");
    auto a_indexes_or =
        EvaluateAccessedIndexesForTile(dim_values, tile_a, shape_a);
    if (!a_indexes_or.ok()) {
      return absl::InternalError(
          absl::StrCat("Tile access error for pid=", pid,
                       " in Tile A: ", a_indexes_or.status().message()));
    }
    auto a_indexes = a_indexes_or.value();
    VLOG(3) << "a_indexes: " << absl::StrJoin(a_indexes, ",");

    auto b_indexes_or =
        EvaluateAccessedIndexesForTile(dim_values, tile_b, shape_b);
    if (!b_indexes_or.ok()) {
      return absl::InternalError(
          absl::StrCat("Tile access error for pid=", pid,
                       " in Tile B: ", b_indexes_or.status().message()));
    }
    auto b_indexes = b_indexes_or.value();
    VLOG(3) << "b_indexes: " << absl::StrJoin(b_indexes, ",");

    if (a_indexes != b_indexes) {
      return absl::InternalError(
          absl::StrCat("Tile access mismatch for pid=", pid, " A accessed [",
                       absl::StrJoin(a_indexes, ","), "] B accessed [",
                       absl::StrJoin(b_indexes, ","), "]"));
    }
  }
  return absl::OkStatus();
}

}  // namespace xla::gpu::experimental
