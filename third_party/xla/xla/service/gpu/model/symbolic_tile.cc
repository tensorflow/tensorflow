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

#include "xla/service/gpu/model/symbolic_tile.h"

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
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"
#include "xla/service/gpu/model/constraint_expression.h"
#include "xla/service/gpu/model/size_and_stride_expression.h"

namespace xla {
namespace gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::AffineExpr;
using ::mlir::AffineMap;
using ::mlir::getAffineConstantExpr;
using ::mlir::getAffineDimExpr;
using ::mlir::MLIRContext;
using Constraint = ConstraintExpression::Constraint;
using ConjointConstraints = llvm::SmallVector<Constraint, 2>;

// Helper to perform function application to using the same parameter for every
// dimension and symbol parameter.
AffineMap SubstituteAllIndicesAndRangeVarSymbolsWithSameValue(
    AffineMap affine_map, AffineExpr value, int num_range_vars) {
  CHECK_LE(num_range_vars, affine_map.getNumSymbols());
  MLIRContext* mlir_context = affine_map.getContext();
  int64_t num_dims = affine_map.getNumDims();
  int64_t num_symbols = affine_map.getNumSymbols();
  llvm::DenseMap<AffineExpr, AffineExpr> indices;

  for (int64_t i = 0; i < num_dims; ++i) {
    indices[getAffineDimExpr(i, mlir_context)] = value;
  }

  // Do not substitute RTVars.
  for (int64_t i = 0; i < num_range_vars; ++i) {
    indices[getAffineSymbolExpr(i, mlir_context)] = value;
  }

  return simplifyAffineMap(affine_map.replace(indices, num_dims, num_symbols));
}

// Simplifies the given affine expression using the constraints / bounds of
// the reference indexing map.
//
// The dimensions and symbols of the expression should correspond to the
// dimensions and symbols of the reference indexing map.
AffineExpr SimplifyAffineExpr(const AffineExpr& expr,
                              const IndexingMap& reference) {
  AffineMap tmp_affine_map =
      AffineMap::get(/*dimCount=*/reference.GetDimVars().size(),
                     /*symbolCount=*/reference.GetSymbolCount(),
                     /*results=*/{expr},
                     /*context=*/reference.GetMLIRContext());
  IndexingMap tmp_indexing_map(
      /*affine_map=*/std::move(tmp_affine_map),
      /*dimensions=*/reference.GetDimVars(),
      /*range_vars=*/reference.GetRangeVars(),
      /*rt_vars=*/reference.GetRTVars(),
      /*constraints=*/reference.GetConstraints());
  tmp_indexing_map.Simplify(IndexingMap::SimplifyPointDimensions::kPreserve);

  CHECK_EQ(tmp_indexing_map.GetAffineMap().getResults().size(), 1);
  return tmp_indexing_map.GetAffineMap().getResults().back();
}

}  // anonymous namespace

/*static*/ std::optional<SymbolicTile> SymbolicTile::FromIndexingMap(
    IndexingMap indexing_map) {
  VLOG(1) << "SymbolicTile::FromIndexingMap: " << indexing_map;

  // We do not handle indexing maps with pre-existing constraints for now.
  // Let's try to simplify the indexing map, because the constraints my be
  // redundant.
  // TODO(bchetioui): Consider doing the simplification in the caller, not here.
  bool did_simplify =
      indexing_map.Simplify(IndexingMap::SimplifyPointDimensions::kPreserve);
  VLOG(1) << "did_simplify: " << did_simplify;
  if (indexing_map.GetConstraintsCount() != 0) {
    VLOG(1) << "Deriving symbolic tile from indexing map with pre-existing "
            << "constraints might produce spurious constraints. Bailing out. "
            << indexing_map;
    return std::nullopt;
  }

  AffineMap input_affine_map = indexing_map.GetAffineMap();
  MLIRContext* mlir_context = input_affine_map.getContext();

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
  std::vector<AffineExpr> offset_expressions =
      SubstituteAllIndicesAndRangeVarSymbolsWithSameValue(
          input_affine_map, getAffineConstantExpr(0, mlir_context),
          indexing_map.GetRangeVarsCount())
          .getResults();
  for (AffineExpr& expr : offset_expressions) {
    expr = SimplifyAffineExpr(expr, indexing_map);
  }

  ConstraintExpression constraints = ConstraintExpression::GetAlwaysSatisfied();
  std::vector<AffineExpr> size_expressions;
  std::vector<AffineExpr> stride_expressions;
  size_expressions.reserve(offset_expressions.size());
  stride_expressions.reserve(offset_expressions.size());

  // strided_indexing_expressions =
  //     f(x0, ..., x{M-1})[x{M}, ..., x{M+P-1}] - offset_expressions
  for (auto [composite_indexing, offset] :
       llvm::zip(input_affine_map.getResults(), offset_expressions)) {
    std::optional<SizeAndStrideExpression> maybe_size_and_stride =
        ExtractSizeAndStride(SimplifyAffineExpr(composite_indexing - offset,
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

  std::vector<AffineExpr> results;
  absl::c_move(std::move(offset_expressions), std::back_inserter(results));
  absl::c_move(std::move(size_expressions), std::back_inserter(results));
  absl::c_move(std::move(stride_expressions), std::back_inserter(results));

  AffineMap tile_affine_map =
      AffineMap::get(/*dimCount=*/tile_sizes.size(),
                     /*symbolCount=*/indexing_map.GetSymbolCount(),
                     /*results=*/results,
                     /*context=*/indexing_map.GetMLIRContext());

  // TODO(b/349507828): Can we derive any constraint from the constraints of
  // the original indexing map?
  IndexingMap tile_map(
      /*affine_map=*/std::move(tile_affine_map),
      /*dimensions=*/std::move(tile_sizes),
      /*range_vars=*/indexing_map.GetRangeVars(),
      /*rt_vars=*/indexing_map.GetRTVars());
  tile_map.RemoveUnusedSymbols();
  CHECK_EQ(tile_map.GetRangeVarsCount(), 0);
  VLOG(1) << "tile_map: " << tile_map;

  constraints.Simplify();
  return SymbolicTile(std::move(tile_map), std::move(constraints));
}

std::string SymbolicTile::ToString() const {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}

void SymbolicTile::Print(std::ostream& out) const {
  out << "Symbolic tile with \n";
  out << "\toffset_map: " << offset_map();
  out << "\n\tsize_map: " << size_map();
  out << "\n\tstride_map: " << stride_map();
  const std::vector<IndexingMap::Variable>& rt_vars = tile_map_.GetRTVars();
  if (!rt_vars.empty()) {
    out << "\n\trt_vars: ";
    for (const auto& [index, rt_var] : llvm::enumerate(rt_vars)) {
      out << 's' << index << " in " << rt_var.bounds << ", ";
    }
  }
  if (!constraints_.IsAlwaysSatisfied()) {
    out << "\n\tconstraints: ";
    constraints_.Print(out);
  }
}

namespace {
// The results of `SymbolicTile::tile_map_` can be split into 3 groups: offsets,
// sizes, and strides.
constexpr int kNumComponentsPerTiledDimension = 3;
}  // namespace

AffineMap SymbolicTile::offset_map() const {
  int64_t num_results = tile_map_.GetAffineMap().getResults().size();
  CHECK_EQ(num_results % kNumComponentsPerTiledDimension, 0);
  int64_t component_size = num_results / kNumComponentsPerTiledDimension;
  // RTVars are included in the symbols.
  return tile_map_.GetAffineMap().getSliceMap(0, component_size);
}

AffineMap SymbolicTile::size_map() const {
  AffineMap affine_map = tile_map_.GetAffineMap();
  int64_t num_results = affine_map.getResults().size();
  CHECK_EQ(num_results % kNumComponentsPerTiledDimension, 0);
  int64_t component_size = num_results / kNumComponentsPerTiledDimension;

  // RTVars are *not* included in the symbols.
  return AffineMap::get(
      /*dimCount=*/affine_map.getNumDims(),
      /*symbolCount=*/affine_map.getNumSymbols() - tile_map_.GetRTVarsCount(),
      /*results=*/affine_map.getResults().slice(component_size, component_size),
      /*context=*/affine_map.getContext());
}

AffineMap SymbolicTile::stride_map() const {
  AffineMap affine_map = tile_map_.GetAffineMap();
  int64_t num_results = affine_map.getResults().size();
  CHECK_EQ(num_results % kNumComponentsPerTiledDimension, 0);
  int64_t component_size = num_results / kNumComponentsPerTiledDimension;

  // RTVars are *not* included in the symbols.
  return AffineMap::get(
      /*dimCount=*/affine_map.getNumDims(),
      /*symbolCount=*/affine_map.getNumSymbols() - tile_map_.GetRTVarsCount(),
      /*results=*/
      affine_map.getResults().slice(2 * component_size, component_size),
      /*context=*/affine_map.getContext());
}

llvm::SmallVector<int64_t> EvaluateTileOffsets(
    const SymbolicTile& symbolic_tile, absl::Span<int64_t const> parameters) {
  return EvaluateAffineMap(symbolic_tile.offset_map(),
                           /*dim_values=*/parameters);
}

llvm::SmallVector<int64_t> EvaluateTileSizes(
    const SymbolicTile& symbolic_tile, absl::Span<int64_t const> parameters) {
  return EvaluateAffineMap(symbolic_tile.size_map(),
                           /*dim_values=*/parameters);
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
  return EvaluateAffineMap(symbolic_tile.stride_map(),
                           /*dim_values=*/clamped_parameters);
}

}  // namespace gpu
}  // namespace xla
