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

#include "xla/codegen/tiling/symbolic_tile.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/codegen/tiling/constraint_expression.h"
#include "xla/codegen/tiling/size_and_stride_expression.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"

namespace xla {
namespace {

using ::llvm::SmallVector;
using ::mlir::MLIRContext;
using Constraint = ConstraintExpression::Constraint;
using ConjointConstraints = llvm::SmallVector<Constraint, 2>;

// Helper to perform function application to using the same parameter for every
// dimension and symbol parameter.
SymbolicMap SubstituteAllIndicesAndRangeVarSymbolsWithSameValue(
    SymbolicMap symbolic_map, SymbolicExpr value, int num_range_vars) {
  CHECK_LE(num_range_vars, symbolic_map.GetNumSymbols());
  MLIRContext* mlir_context = symbolic_map.GetContext();
  int64_t num_dims = symbolic_map.GetNumDims();
  llvm::DenseMap<SymbolicExpr, SymbolicExpr> indices;

  for (int64_t i = 0; i < num_dims; ++i) {
    indices[CreateDimExpr(i, mlir_context)] = value;
  }

  // Do not substitute RTVars.
  for (int64_t i = 0; i < num_range_vars; ++i) {
    indices[CreateSymbolExpr(i, num_dims, mlir_context)] = value;
  }

  return symbolic_map.Replace(indices);
}

// Simplifies the given affine expression using the constraints / bounds of
// the reference indexing map.
//
// The dimensions and symbols of the expression should correspond to the
// dimensions and symbols of the reference indexing map.
SymbolicExpr SimplifySymbolicExpr(const SymbolicExpr& expr,
                                  const IndexingMap& reference) {
  SymbolicMap tmp_symbolic_map =
      SymbolicMap::Get(/*ctx=*/reference.GetMLIRContext(),
                       /*num_dimensions=*/reference.GetDimVars().size(),
                       /*num_symbols=*/reference.GetSymbolCount(),
                       /*exprs=*/{expr});
  IndexingMap tmp_indexing_map(
      /*symbolic_map=*/tmp_symbolic_map,
      /*dimensions=*/reference.GetDimVars(),
      /*range_vars=*/reference.GetRangeVars(),
      /*rt_vars=*/reference.GetRTVars(),
      /*constraints=*/reference.GetSymbolicConstraints());
  tmp_indexing_map.Simplify(IndexingMap::SimplifyPointDimensions::kPreserve);

  CHECK_EQ(tmp_indexing_map.GetSymbolicMap().GetNumResults(), 1);
  return tmp_indexing_map.GetSymbolicMap().GetResult(0);
}

// Returns a boolean indicating whether the constraints of the parameter
// indexing maps are known to be irrelevant with regards to symbolic tile
// derivation.
bool IndexingMapConstraintsCanBeIgnored(const IndexingMap& indexing_map) {
  for (const auto& [expr, range] : indexing_map.GetSymbolicConstraints()) {
    bool range_has_no_offset = range.lower == 0;
    bool constrains_result =
        absl::c_linear_search(indexing_map.GetSymbolicMap().GetResults(), expr);
    // In this case, we know that the constraint we found here is
    //   1. directly restricting the range of a result of the input indexing
    //      map, and
    //   2. the restricted range may only invalidate "high values" (i.e., the
    //      range has a lower bound of 0)
    //
    // Since indexing map constraints only allow expressing continuous
    // intervals, 1. tells us that we are restricting the continuous range of
    // an output (i.e. we're not constraining as a way to express some kind of
    // interior padding), and 2. tells us that, if we are inserting padding
    // (i.e. the constraint is not redundant), then that padding applies to
    // high values.
    //
    // This essentially falls into the use case of "constructing a tile size
    // that spans indices outside of the input space", which is a use case we
    // intend for `SymbolicTile` to support. Therefore, we should be able to
    // safely ignore this constraint.
    //
    // Note that the same should hold for low padding, but we leave that for
    // future work in order to not overcomplicate things.
    if (range_has_no_offset && constrains_result) {
      continue;
    }

    return false;
  }

  return true;
}

}  // anonymous namespace

/*static*/ std::optional<SymbolicTile> SymbolicTile::FromIndexingMap(
    IndexingMap indexing_map) {
  VLOG(1) << "SymbolicTile::FromIndexingMap: " << indexing_map;
  if (indexing_map.IsUndefined()) {
    return std::nullopt;
  }

  // We do not handle indexing maps with pre-existing constraints for now.
  // Let's try to simplify the indexing map, because the constraints my be
  // redundant.
  // TODO(bchetioui): Consider doing the simplification in the caller, not here.
  bool did_simplify =
      indexing_map.Simplify(IndexingMap::SimplifyPointDimensions::kPreserve);
  VLOG(1) << "did_simplify: " << did_simplify;
  if (!IndexingMapConstraintsCanBeIgnored(indexing_map)) {
    VLOG(1) << "Deriving symbolic tile from indexing map with pre-existing "
            << "constraints might produce spurious constraints. Bailing out. "
            << indexing_map;
    return std::nullopt;
  }

  SymbolicMap input_symbolic_map = indexing_map.GetSymbolicMap();
  MLIRContext* mlir_context = input_symbolic_map.GetContext();

  // If indexing_map describes a tileable space, then input_affine_map can be
  // expressed as
  //   f(dim0, ..., dim{M-1})[sym0, ..., sym{P-1}] = (expr0, ..., expr{N-1})
  // where the result expressions expr0, ..., expr{N-1} are strided expressions
  // of the form
  //     offset_expr{i} + stride_expr{i} * index_expr{i}
  // with 0 <= i < N.
  //
  // We are interested in extracting expressions for offset_expr{i},
  // stride_expr{i}, and size_expr{i} (the count of different values that
  // expr{i} can represent).
  //
  // We have that the following equations hold:
  //
  // (1) f(0, ..., 0)[0, ..., 0]{i}
  //   = offset_expr{i} + stride_expr{i} * 0
  //   = offset_expr{i}
  //
  // (2) f(x0, ..., x{M-1})[x{M}, ..., x{M+P-1}]{i} - f(0, ..., 0)[0, ..., 0]{i}
  //   = offset_expr{i} + stride_expr{i} * index_expr{i} - offset_expr{i}
  //   = stride_expr{i} * index_expr{i}
  //
  // offset_expressions = f(0, ..., 0)[0, ..., 0].
  auto offset_expressions =
      SubstituteAllIndicesAndRangeVarSymbolsWithSameValue(
          input_symbolic_map, CreateSymbolicConstant(0, mlir_context),
          indexing_map.GetRangeVarsCount())
          .GetResults();
  for (SymbolicExpr& expr : offset_expressions) {
    expr = SimplifySymbolicExpr(expr, indexing_map);
  }

  ConstraintExpression constraints = ConstraintExpression::GetAlwaysSatisfied();
  std::vector<SymbolicExpr> size_expressions;
  std::vector<SymbolicExpr> stride_expressions;
  size_expressions.reserve(offset_expressions.size());
  stride_expressions.reserve(offset_expressions.size());

  // strided_indexing_expressions =
  //     f(x0, ..., x{M-1})[x{M}, ..., x{M+P-1}] - offset_expressions
  for (auto [composite_indexing, offset] :
       llvm::zip(input_symbolic_map.GetResults(), offset_expressions)) {
    std::optional<SizeAndStrideExpression> maybe_size_and_stride =
        ExtractSizeAndStride(SimplifySymbolicExpr(composite_indexing - offset,
                                                  /*reference=*/indexing_map),
                             indexing_map.GetDimensionBounds(),
                             indexing_map.GetSymbolBounds());
    if (!maybe_size_and_stride.has_value()) {
      VLOG(1) << "No size and stride extracted";
      return std::nullopt;
    }
    size_expressions.push_back(maybe_size_and_stride->size);
    stride_expressions.push_back(maybe_size_and_stride->stride);

    constraints = constraints && maybe_size_and_stride->constraints;
  }

  // DimVars in `indexing_map` represent indices, but in `tile_map` they will
  // represent the size of the tile. So we need to add 1 to the bounds.
  // For example: indices: [0, 9] -> sizes: [1, 10].
  std::vector<IndexingMap::Variable> tile_sizes = indexing_map.GetDimVars();
  for (IndexingMap::Variable& tile_size : tile_sizes) {
    tile_size.bounds.lower += 1;
    tile_size.bounds.upper += 1;
  }

  llvm::SmallVector<SymbolicExpr> results;
  absl::c_move(std::move(offset_expressions), std::back_inserter(results));
  absl::c_move(std::move(size_expressions), std::back_inserter(results));
  absl::c_move(std::move(stride_expressions), std::back_inserter(results));

  SymbolicMap tile_symbolic_map =
      SymbolicMap::Get(/*ctx=*/indexing_map.GetMLIRContext(),
                       /*num_dimensions=*/tile_sizes.size(),
                       /*num_symbols=*/indexing_map.GetSymbolCount(),
                       /*exprs=*/results);

  // TODO(b/349507828): Can we derive any constraint from the constraints of
  // the original indexing map?
  IndexingMap tile_map(
      /*symbolic_map=*/std::move(tile_symbolic_map),
      /*dimensions=*/std::move(tile_sizes),
      /*range_vars=*/indexing_map.GetRangeVars(),
      /*rt_vars=*/indexing_map.GetRTVars());
  tile_map.RemoveUnusedSymbols();
  CHECK_EQ(tile_map.GetRangeVarsCount(), 0);
  VLOG(1) << "tile_map: " << tile_map;

  constraints.Simplify();
  return SymbolicTile(std::move(tile_map), std::move(constraints));
}

std::string SymbolicTile::ToString(absl::string_view separator) const {
  std::stringstream ss;
  Print(ss, separator);
  return ss.str();
}

void SymbolicTile::Print(std::ostream& out, absl::string_view separator) const {
  out << "Symbolic tile with" << separator;
  out << "offset_map: " << offset_map() << separator;
  out << "size_map: " << size_map() << separator;
  out << "stride_map: " << stride_map();
  const std::vector<IndexingMap::Variable>& rt_vars = tile_map_.GetRTVars();
  if (!rt_vars.empty()) {
    out << separator << "rt_vars: ";
    for (const auto& [index, rt_var] : llvm::enumerate(rt_vars)) {
      out << 's' << index << " in " << rt_var.bounds << ", ";
    }
  }
  if (!constraints_.IsAlwaysSatisfied()) {
    out << separator << "constraints: ";
    constraints_.Print(out, tile_map_.GetDimensionCount());
  }
}

namespace {
// The results of `SymbolicTile::tile_map_` can be split into 3 groups: offsets,
// sizes, and strides.
constexpr int kNumComponentsPerTiledDimension = 3;
}  // namespace

SymbolicMap SymbolicTile::offset_map() const {
  int64_t num_results = tile_map_.GetSymbolicMap().GetNumResults();
  CHECK_EQ(num_results % kNumComponentsPerTiledDimension, 0);
  int64_t component_size = num_results / kNumComponentsPerTiledDimension;
  // RTVars are included in the symbols.
  return tile_map_.GetSymbolicMap().GetSliceMap(0, component_size);
}

SymbolicMap SymbolicTile::size_map() const {
  SymbolicMap symbolic_map = tile_map_.GetSymbolicMap();
  int64_t num_results = symbolic_map.GetNumResults();
  CHECK_EQ(num_results % kNumComponentsPerTiledDimension, 0);
  int64_t component_size = num_results / kNumComponentsPerTiledDimension;

  // RTVars are *not* included in the symbols.
  return SymbolicMap::Get(
      /*ctx=*/symbolic_map.GetContext(),
      /*num_dimensions=*/symbolic_map.GetNumDims(),
      /*num_symbols=*/symbolic_map.GetNumSymbols() - tile_map_.GetRTVarsCount(),
      /*exprs=*/
      SmallVector<SymbolicExpr>(
          symbolic_map.GetResults().slice(component_size, component_size)));
}

SymbolicMap SymbolicTile::stride_map() const {
  SymbolicMap symbolic_map = tile_map_.GetSymbolicMap();
  int64_t num_results = symbolic_map.GetNumResults();
  CHECK_EQ(num_results % kNumComponentsPerTiledDimension, 0);
  int64_t component_size = num_results / kNumComponentsPerTiledDimension;

  // RTVars are *not* included in the symbols.
  return SymbolicMap::Get(
      /*ctx=*/symbolic_map.GetContext(),
      /*num_dimensions=*/symbolic_map.GetNumDims(),
      /*num_symbols=*/symbolic_map.GetNumSymbols() - tile_map_.GetRTVarsCount(),
      /*exprs=*/
      SmallVector<SymbolicExpr>(
          symbolic_map.GetResults().slice(2 * component_size, component_size)));
}

llvm::SmallVector<int64_t> EvaluateTileOffsets(
    const SymbolicTile& symbolic_tile, absl::Span<int64_t const> parameters) {
  return symbolic_tile.offset_map().Evaluate(/*dim_values=*/parameters);
}

llvm::SmallVector<int64_t> EvaluateTileSizes(
    const SymbolicTile& symbolic_tile, absl::Span<int64_t const> parameters) {
  return symbolic_tile.size_map().Evaluate(/*dim_values=*/parameters);
}

llvm::SmallVector<int64_t> EvaluateTileStrides(
    const SymbolicTile& symbolic_tile, absl::Span<int64_t const> parameters) {
  llvm::SmallVector<int64_t> clamped_parameters;
  clamped_parameters.reserve(parameters.size());
  // We need to clamp the parameters to the dimension bounds, otherwise the
  // stride expressions would potentially return wrong results. The underlying
  // implementation detail is that the IfNeqOne affine expression that we use
  // for expanding reshapes assumes that the tile parameter is not bigger than
  // the dimension bound. To make the assumption hold, we clamp the parameters
  // accordingly.
  for (auto [parameter, dim_bounds] :
       llvm::zip(parameters, symbolic_tile.tile_map().GetDimensionBounds())) {
    clamped_parameters.push_back(std::min(parameter, dim_bounds.upper));
  }
  return symbolic_tile.stride_map().Evaluate(/*dim_values=*/clamped_parameters);
}

}  // namespace xla
