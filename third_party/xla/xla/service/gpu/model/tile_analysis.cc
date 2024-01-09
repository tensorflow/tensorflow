/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/tile_analysis.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/service/gpu/model/indexing_map_simplifier.h"

namespace xla {
namespace gpu {
namespace {

using mlir::AffineBinaryOpExpr;
using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::AffineMap;
using mlir::AffineSymbolExpr;
using mlir::getAffineBinaryOpExpr;
using mlir::getAffineDimExpr;
using mlir::MLIRContext;

// Helper for `AffineMapDescribesTile`. The `trivial_symbol_ids` set contains
// the indices of symbols whose exclusive upper bound is 1.
bool AffineExprReducesToScalar(
    const AffineExpr& expr,
    const absl::flat_hash_set<int64_t>& trivial_symbol_ids) {
  switch (expr.getKind()) {
    case AffineExprKind::Add:
    case AffineExprKind::Mul:
    case AffineExprKind::Mod:
    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv: {
      auto binop_expr = llvm::cast<AffineBinaryOpExpr>(expr);
      return AffineExprReducesToScalar(binop_expr.getLHS(),
                                       trivial_symbol_ids) &&
             AffineExprReducesToScalar(binop_expr.getRHS(), trivial_symbol_ids);
    }
    case AffineExprKind::Constant:
      return true;
    case AffineExprKind::DimId:
      return true;
    case AffineExprKind::SymbolId:
      return trivial_symbol_ids.contains(
          llvm::cast<AffineSymbolExpr>(expr).getPosition());
  }
}

// Helper for `AffineMapDescribesTile`. The `trivial_symbol_ids` set contains
// the indices of symbols whose exclusive upper bound is 1.
bool AffineExprIsStridedRangeExpression(
    const AffineExpr& expr,
    const absl::flat_hash_set<int64_t>& trivial_symbol_ids) {
  switch (expr.getKind()) {
    case AffineExprKind::Add:
    case AffineExprKind::Mul: {
      auto binop_expr = llvm::cast<AffineBinaryOpExpr>(expr);
      return AffineExprReducesToScalar(binop_expr.getLHS(),
                                       trivial_symbol_ids) ||
             AffineExprReducesToScalar(binop_expr.getRHS(), trivial_symbol_ids);
    }
    case AffineExprKind::Mod:
    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv:
      return AffineExprReducesToScalar(expr, trivial_symbol_ids);
    case AffineExprKind::Constant:
    case AffineExprKind::DimId:
    case AffineExprKind::SymbolId:
      return true;
  }
}

// Checks whether an `AffineMap` describes a tile as defined in the docstring of
// `SymbolicTile`. The `trivial_symbol_ids` set contains the indices of symbols
// whose exclusive upper bound is 1.
bool AffineMapDescribesTile(
    const AffineMap& affine_map,
    const absl::flat_hash_set<int64_t>& trivial_symbol_ids) {
  return absl::c_all_of(
      affine_map.getResults(), [&trivial_symbol_ids](const AffineExpr& expr) {
        return AffineExprIsStridedRangeExpression(expr, trivial_symbol_ids);
      });
}

}  // namespace

SymbolicTile::SymbolicTile(absl::Span<int64_t const> target_shape,
                           MLIRContext* mlir_context) {
  std::vector<mlir::AffineExpr> exprs;
  IndexingMapSimplifier simplifier(mlir_context);
  int64_t num_target_dims = target_shape.size();

  std::vector<int64_t> max_strides_and_offsets;
  max_strides_and_offsets.reserve(2 * num_target_dims);

  for (int dim = 0; dim < num_target_dims; ++dim) {
    AffineExpr tile_size = getAffineSymbolExpr(dim, mlir_context);
    AffineExpr stride = getAffineDimExpr(2 * dim, mlir_context);
    AffineExpr offset = getAffineDimExpr(2 * dim + 1, mlir_context);

    exprs.push_back(getAffineBinaryOpExpr(
        AffineExprKind::Add,
        getAffineBinaryOpExpr(AffineExprKind::Mul, stride, tile_size), offset));

    max_strides_and_offsets.push_back(target_shape[dim]);
    max_strides_and_offsets.push_back(target_shape[dim]);
  }

  AffineMap affine_map = mlir::AffineMap::get(
      2 * num_target_dims, num_target_dims, exprs, mlir_context);

  affine_map_ = affine_map;
  sizes_ = std::vector<std::optional<int64_t>>(num_target_dims, std::nullopt);
  max_sizes_ = std::vector<int64_t>(target_shape.begin(), target_shape.end());
  max_strides_and_offsets_ = max_strides_and_offsets;
}

std::optional<SymbolicTile> SymbolicTile::TryPropagateTileThroughIndexingMap(
    const IndexingMap& indexing_map) const {
  AffineMap composed_map = indexing_map.affine_map.compose(affine_map_);
  MLIRContext* ctx = composed_map.getContext();

  IndexingMapSimplifier simplifier(ctx);

  std::vector<std::optional<int64_t>> new_sizes;
  std::vector<int64_t> new_max_sizes;
  new_max_sizes.reserve(max_sizes_.size());
  new_sizes.reserve(sizes_.size());

  // The symbols in 'composed_map' should be ordered such that the symbols
  // declared in 'indexing_map.affine_map' precede those defined in affine_map_.
  int64_t symbol_index = 0;
  for (const Range& symbol_range : indexing_map.domain.symbol_ranges) {
    int64_t symbol_bound = symbol_range.upper_bound;
    simplifier.SetInclusiveBounds(getAffineSymbolExpr(symbol_index++, ctx),
                                  /*lower=*/0, /*upper=*/symbol_bound - 1);
    // Newly inserted symbols capture the whole target dimension.
    new_sizes.push_back(symbol_bound);
    new_max_sizes.push_back(symbol_bound);
  }

  for (const auto [msize, max_size] : llvm::zip(sizes_, max_sizes_)) {
    int64_t symbol_bound = msize.has_value() ? msize.value() : max_size;
    simplifier.SetInclusiveBounds(getAffineSymbolExpr(symbol_index++, ctx),
                                  /*lower=*/0, /*upper=*/symbol_bound - 1);
  }
  absl::c_copy(sizes_, std::back_inserter(new_sizes));
  absl::c_copy(max_sizes_, std::back_inserter(new_max_sizes));

  absl::flat_hash_set<int64_t> trivial_symbol_indices;
  for (symbol_index = 0; symbol_index < new_sizes.size(); ++symbol_index) {
    if ((new_sizes[symbol_index].has_value() &&
         new_sizes[symbol_index].value() == 1) ||
        new_max_sizes[symbol_index] == 1) {
      trivial_symbol_indices.insert(symbol_index);
    }
  }

  for (const auto [dim_id, max_stride_or_offset] :
       llvm::enumerate(max_strides_and_offsets_)) {
    simplifier.SetInclusiveBounds(getAffineDimExpr(dim_id, ctx),
                                  /*lower=*/0,
                                  /*upper=*/max_stride_or_offset - 1);
  }

  AffineMap potential_tile_map = simplifier.Simplify(composed_map);

  if (AffineMapDescribesTile(potential_tile_map, trivial_symbol_indices)) {
    return SymbolicTile(potential_tile_map, new_sizes, new_max_sizes,
                        max_strides_and_offsets_);
  }

  return std::nullopt;
}

std::ostream& operator<<(std::ostream& out, const SymbolicTile& symbolic_tile) {
  out << ToString(symbolic_tile.affine_map()) << " with\n";
  for (const auto [dim_id, max_stride_or_offset] :
       llvm::enumerate(symbolic_tile.max_strides_and_offsets())) {
    out << absl::StrCat("\td", dim_id, " in [0, ", max_stride_or_offset, "]\n");
  }

  int symbol_id = 0;
  for (const auto [size, max_size] :
       llvm::zip(symbolic_tile.sizes(), symbolic_tile.max_sizes())) {
    const std::string bound_string =
        size.has_value() ? absl::StrCat(size.value()) : "?";
    out << absl::StrCat("\ts", symbol_id++, " in [0, ", bound_string, "] ",
                        "(upper bound: ", max_size, ")\n");
  }
  return out;
}

std::string ToString(const SymbolicTile& symbolic_tile) {
  return ToStringImpl(symbolic_tile);
}

}  // namespace gpu
}  // namespace xla
