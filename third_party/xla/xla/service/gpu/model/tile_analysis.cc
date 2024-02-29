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

#include "xla/service/gpu/model/tile_analysis.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {
namespace {

using absl::StrCat;
using mlir::AffineDimExpr;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::AffineSymbolExpr;
using mlir::getAffineConstantExpr;
using mlir::getAffineDimExpr;
using mlir::MLIRContext;
using mlir::simplifyAffineExpr;

// The number of tile parameters that are inserted for each input dimension when
// constructing a symbolic tile from an indexing map.
constexpr int kNumTileParametersPerInputDim = 3;

// Internal helper that checks whether an affine map of the form
//     (index0, ..., index{M-1})
//       [sym0, ..., sym{P-1}, offset0, size0, stride0, ...,
//        offset{M-1}, size{M-1}, stride{M-1}]
//  -> (expr0, ..., expr{N-1})
//
// describes a symbolic tile. The documentation for
// `RawSymbolicTileFromIndexingMap` explains what this means in details.
bool AffineMapDescribesTile(AffineMap affine_map) {
  int64_t num_known_symbols =
      affine_map.getNumSymbols() -
      kNumTileParametersPerInputDim * affine_map.getNumDims();
  for (AffineExpr result_expr : affine_map.getResults()) {
    int64_t num_hits = 0;
    result_expr.walk([&num_hits, &num_known_symbols](AffineExpr expr) {
      if (auto symbol_expr = llvm::dyn_cast<AffineSymbolExpr>(expr)) {
        num_hits += (symbol_expr.getPosition() < num_known_symbols);
      } else if (auto dim_expr = llvm::dyn_cast<AffineDimExpr>(expr)) {
        ++num_hits;
      }
    });

    if (num_hits > 1) {
      return false;
    }
  }
  return true;
}

// Internal helper to construct symbolic tiles. The only difference with an
// actual symbolic tile is that this structure does not enforce the relevant
// important invariants by construction.
struct RawSymbolicTile {
  AffineMap offset_map;
  AffineMap size_map;
  AffineMap stride_map;
};

// Helper to perform function applications as described in the documentation of
// `RawSymbolicTileFromIndexingMap`.
AffineMap SubstituteAllIndicesAndKnownSymbolsWithSameValue(
    AffineMap affine_map, AffineExpr value, int64_t num_known_symbols) {
  MLIRContext* mlir_context = affine_map.getContext();
  int64_t num_input_dims = affine_map.getNumDims();
  llvm::DenseMap<AffineExpr, AffineExpr> indices;

  for (int64_t i = 0; i < num_input_dims; ++i) {
    indices[getAffineDimExpr(i, mlir_context)] = value;
  }

  for (int64_t i = 0; i < num_known_symbols; ++i) {
    indices[getAffineSymbolExpr(i, mlir_context)] = value;
  }

  return simplifyAffineMap(affine_map.replace(indices, affine_map.getNumDims(),
                                              affine_map.getNumSymbols()));
}

// Extracts non-negative offset, size, and stride expression for each result
// expression in the parameter affine map if the affine map describes a tile.
// Returns `std::nullopt` if the parameter affine map does not describe a tile.
//
// The parameter affine map f must follow the pattern
//     (index0, ..., index{M-1})
//       [sym0, ..., sym{P-1}, offset0, size0, stride0, ...,
//        offset{M-1}, size{M-1}, stride{M-1}]
//  -> (expr0, ..., expr{N-1})
// where the result expressions expr0, ..., expr{N-1} are strided expressions of
// the form
//     offset_expr{i} + stride_expr{i} * index_expr{i}
// with 0 <= i < N, and index_expr{i} is either a symbol with known bound
// (sym0, ..., sym{P-1}) or an index.
//
// Let f'(x0, ..., x{M-1})[x{M}, ..., x{M+P-1}]
//   = f(x0, ..., x{M-1})[x{M}, ..., x{M+P-1}, offset0, size0, stride0, ...,
//                        offset{M-1}, size{M-1}, stride{M-1}]
//
// Then, the following equations hold:
//
// (1) f'(0, ..., 0)[0, ..., 0]{i}
//   = offset_expr{i} + stride_expr{i} * 0
//   = offset_expr{i}
//
// (2) f'(1, ..., 1)[1, ..., 1]{i} - f'(0, ..., 0)[0, ..., 0]{i}
//   = offset_expr{i} + stride_expr{i} * 1 - offset_expr{i}
//   = stride_expr{i}
//
// (3) If stride_expr{i} = 0, we automatically set size_expr{i} = 1. This
//     happens when the strided expression points to a single value that is the
//     same for all elements in the tile.
//
//     If stride_expr{i} != 0, then the relevant size expression can be obtained
//     by analyzing the index expression, which is known to be either a symbol
//     with known bound, or an index parameter. In the former case, we set the
//     size to be the upper bound of the symbol; in the latter case, we
//     substitute the index parameter by its corresponding size parameter.
//
// Strictly solving (1) may yield negative strides (e.g. in the case of
// reverse). Conceptually, negative strides denote of a decremental iteration
// order over indices {0, ..., size_expr{i} - 1}. Since all indices within a
// tile are captured at the same time (they are only explicit here as a
// convenience), we can reverse this iteration order by replacing
//   index_expr{i}
// with
//   (size_expr{i} - 1 - index_expr{i}).
//
// This gives us a new expression
//
//     change-iteration-order(offset_expr{i} + stride_expr{i} * index_expr{i})
//   = offset_expr{i} - |stride_expr{i}| * (size_expr{i} - 1 - index_expr{i})
//   = offset_expr{i} - |stride_expr{i}| * (size_expr{i} - 1) +
//     |stride_expr{i}| * index_expr{i}
//   = offset_expr'{i} + stride_expr'{i} * index_expr{i}
// where
//   offset_expr'{i} = offset_expr{i} - |stride_expr{i}| * (size_expr{i} - 1)
//   stride_expr'{i} = |stride_expr{i}| = -stride_expr{i}
// and size, offset, and stride expressions are positive.
//
// The resulting affine maps elide known symbols from the list of parameter
// symbols, since they will have been replaced by constants.
std::optional<RawSymbolicTile> RawSymbolicTileFromIndexingMap(
    const IndexingMap& indexing_map) {
  AffineMap affine_map = indexing_map.GetAffineMap();
  if (!AffineMapDescribesTile(affine_map)) {
    return std::nullopt;
  }

  MLIRContext* mlir_context = affine_map.getContext();
  int64_t num_known_symbols =
      affine_map.getNumSymbols() -
      affine_map.getNumDims() * kNumTileParametersPerInputDim;
  int64_t num_results = affine_map.getNumResults();

  // offsets_expr = f'(0, ..., 0)[0, ..., 0]
  AffineMap f_prime_0 = SubstituteAllIndicesAndKnownSymbolsWithSameValue(
      affine_map, getAffineConstantExpr(0, mlir_context), num_known_symbols);
  llvm::ArrayRef<AffineExpr> unnormalized_offset_expressions =
      f_prime_0.getResults();

  // Compute f'(1, ..., 1)[1, ..., 1].
  AffineMap f_prime_1 = SubstituteAllIndicesAndKnownSymbolsWithSameValue(
      affine_map, getAffineConstantExpr(1, mlir_context), num_known_symbols);

  // strides_expr = f'(1, ..., 1)[1, ..., 1] - f'(0, ..., 0)[0, ..., 0]
  std::vector<AffineExpr> signed_stride_expressions;
  signed_stride_expressions.reserve(num_results);
  for (auto [sub_lhs, sub_rhs] :
       llvm::zip(f_prime_1.getResults(), f_prime_0.getResults())) {
    signed_stride_expressions.push_back(
        simplifyAffineExpr(sub_lhs - sub_rhs, affine_map.getNumDims(),
                           affine_map.getNumSymbols()));
  }

  // Deduce size_expr. At each index, if the stride is non-zero, once rid of
  // the offset expression, the remaining expression can be one of two things;
  //   1. a single parameter---either an index parameter, or a symbol with
  //      known bounds. This parameter is the size expression, and in the
  //      case of a symbol with known bounds, we can directly make it a
  //      constant;
  //   2. the product of an index parameter (or symbol with known bound)
  //      with an expression consisting only of constants, and offsets
  //      and strides parameters. In that case, the index parameter/symbol
  //      with known bound is the size expression, and we do like in the
  //      first bullet.
  // This structure is guaranteed by the `AffineMapDescribesTile` filter
  // at the top of the function.
  std::vector<AffineExpr> size_expressions;
  size_expressions.reserve(num_results);
  constexpr int kSizePositionWithinTileParameters = 1;
  for (auto [offset_expr, stride_expr, input_expr] :
       llvm::zip(unnormalized_offset_expressions, signed_stride_expressions,
                 affine_map.getResults())) {
    AffineExpr size_expr;
    if (stride_expr == getAffineConstantExpr(0, mlir_context)) {
      size_expr = getAffineConstantExpr(1, mlir_context);
    } else {
      AffineExpr strided_size_expr =
          simplifyAffineExpr(input_expr - offset_expr, affine_map.getNumDims(),
                             affine_map.getNumSymbols());

      strided_size_expr.walk([&](AffineExpr expr) {
        auto symbol_expr = llvm::dyn_cast<AffineSymbolExpr>(expr);
        if (symbol_expr && symbol_expr.getPosition() < num_known_symbols) {
          CHECK(!size_expr);
          const Range& symbol_range =
              indexing_map.GetSymbolRange(symbol_expr.getPosition());
          size_expr = getAffineConstantExpr(
              symbol_range.upper_bound - symbol_range.lower_bound + 1,
              mlir_context);
        } else if (auto dim_expr = llvm::dyn_cast<AffineDimExpr>(expr)) {
          CHECK(!size_expr);
          size_expr = getAffineSymbolExpr(
              num_known_symbols +
                  dim_expr.getPosition() * kNumTileParametersPerInputDim +
                  kSizePositionWithinTileParameters,
              mlir_context);
        }
      });
    }
    size_expressions.push_back(size_expr);
  }

  // Normalize offsets and strides to be non-negative if possible.
  std::vector<AffineExpr> offset_expressions;
  offset_expressions.reserve(num_results);
  std::vector<AffineExpr> stride_expressions;
  stride_expressions.reserve(num_results);
  RangeEvaluator range_evaluator(indexing_map.GetDimensionRanges(),
                                 indexing_map.GetSymbolRanges(),
                                 indexing_map.GetMLIRContext());
  for (auto [offset_expr, stride_expr, size_expr] :
       llvm::zip(unnormalized_offset_expressions, signed_stride_expressions,
                 size_expressions)) {
    if (range_evaluator.IsAlwaysPositiveOrZero(stride_expr)) {
      offset_expressions.push_back(offset_expr);
      stride_expressions.push_back(stride_expr);
    } else if (range_evaluator.IsAlwaysNegativeOrZero(stride_expr)) {
      offset_expressions.push_back(offset_expr + stride_expr * size_expr);
      stride_expressions.push_back(-stride_expr);
    } else {
      // In that case, the comparison is inconclusive---the expression may be
      // both positive or negative depending on the parameters. We can not
      // produce a tile that satisfies the "non-negative" requirements.
      return std::nullopt;
    }
  }

  int64_t num_symbols = affine_map.getNumSymbols();
  return RawSymbolicTile(
      {.offset_map =
           AffineMap::get(0, num_symbols, offset_expressions, mlir_context)
               .shiftSymbols(-num_known_symbols),
       .size_map =
           AffineMap::get(0, num_symbols, size_expressions, mlir_context)
               .shiftSymbols(-num_known_symbols),
       .stride_map =
           AffineMap::get(0, num_symbols, stride_expressions, mlir_context)
               .shiftSymbols(-num_known_symbols)});
}

}  // anonymous namespace

/*static*/ std::optional<SymbolicTile> SymbolicTile::FromIndexingMap(
    const IndexingMap& indexing_map) {
  MLIRContext* mlir_context = indexing_map.GetAffineMap().getContext();
  int64_t num_input_dims = indexing_map.GetDimensionCount();
  std::vector<AffineExpr> exprs;
  exprs.reserve(num_input_dims);

  std::vector<Range> tile_dimension_ranges;
  tile_dimension_ranges.reserve(num_input_dims);
  std::vector<Range> tile_symbol_ranges;
  tile_symbol_ranges.reserve(kNumTileParametersPerInputDim * num_input_dims +
                             indexing_map.GetAffineMap().getNumSymbols());

  // The symbols declared in 'indexing_map.affine_map' will precede those
  // defined in the producer map we construct here.
  absl::c_copy(indexing_map.GetSymbolRanges(),
               std::back_inserter(tile_symbol_ranges));

  // For each input dims we add kNumTileParametersPerInputDim = 3 symbols, as
  // well as a single dim. Symbols are ordered in (offset, size, stride)
  // triplets.
  for (int64_t dim = 0; dim < num_input_dims; ++dim) {
    AffineExpr index = getAffineDimExpr(dim, mlir_context);
    AffineExpr offset =
        getAffineSymbolExpr(kNumTileParametersPerInputDim * dim, mlir_context);
    AffineExpr stride = getAffineSymbolExpr(
        kNumTileParametersPerInputDim * dim + 2, mlir_context);

    exprs.push_back(offset + stride * index);

    Range range = indexing_map.GetDimensionRange(dim);
    tile_dimension_ranges.push_back(range);

    for (int64_t symbol_index = 0; symbol_index < kNumTileParametersPerInputDim;
         ++symbol_index) {
      tile_symbol_ranges.push_back(range);
    }
  }

  AffineMap producer_map = AffineMap::get(
      num_input_dims, kNumTileParametersPerInputDim * num_input_dims, exprs,
      mlir_context);

  IndexingMap composed_indexing_map(
      indexing_map.GetAffineMap().compose(producer_map), tile_dimension_ranges,
      tile_symbol_ranges);

  composed_indexing_map.Simplify();

  std::optional<RawSymbolicTile> maybe_raw_symbolic_tile =
      RawSymbolicTileFromIndexingMap(composed_indexing_map);

  if (!maybe_raw_symbolic_tile.has_value()) {
    return std::nullopt;
  }

  return SymbolicTile(maybe_raw_symbolic_tile->offset_map,
                      maybe_raw_symbolic_tile->size_map,
                      maybe_raw_symbolic_tile->stride_map);
}

std::string SymbolicTile::ToString(const AffineMapPrinter& printer) const {
  std::string s;
  std::stringstream ss(s);
  Print(ss, printer);
  return ss.str();
}

void SymbolicTile::Print(std::ostream& out,
                         const AffineMapPrinter& printer) const {
  out << "Symbolic tile with \n";
  out << "\toffset_map: ";
  printer.Print(out, offset_map_);
  out << "\n\tsize_map: ";
  printer.Print(out, size_map_);
  out << "\n\tstride_map: ";
  printer.Print(out, stride_map_);
  out << "\n";
}

std::ostream& operator<<(std::ostream& out, const SymbolicTile& symbolic_tile) {
  AffineMapPrinter printer;

  // This utilizes the assumption that symbols are structured as triplets, i.e.
  // [offset0, size0, stride0, ... offset{N-1}, size{N-1}, stride{N-1}]
  // where N is the tensor rank.
  for (int64_t triplet_start = 0;
       triplet_start < symbolic_tile.offset_map().getNumSymbols();
       triplet_start += kNumTileParametersPerInputDim) {
    int64_t triplet_idx = triplet_start / kNumTileParametersPerInputDim;
    printer.SetSymbolName(triplet_start, StrCat("offset", triplet_idx));
    printer.SetSymbolName(triplet_start + 1, StrCat("size", triplet_idx));
    printer.SetSymbolName(triplet_start + 2, StrCat("stride", triplet_idx));
  }

  symbolic_tile.Print(out, printer);
  return out;
}

}  // namespace gpu
}  // namespace xla
