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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/service/gpu/model/affine_map_printer.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {
namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineExprKind;
using ::mlir::AffineMap;
using ::mlir::getAffineConstantExpr;
using ::mlir::getAffineDimExpr;
using ::mlir::MLIRContext;

// Gets a modified version of `expressions` where the indices of the RTVars are
// decreased, because no RangeVars appear anymore.
//
// (dimensions)[RangeVars, RTVars] -> (dimensions)[RTVars]
//
// Precondition: `expressions` must not contain any RangeVar symbols.
//
// This handles a vector of expressions at once, because we don't want to
// regenerate `symbol_map` every time.
std::vector<AffineExpr> WithoutRangeVars(std::vector<AffineExpr> expressions,
                                         const IndexingMap& indexing_map) {
  // Precondition check:
  for (AffineExpr expression : expressions) {
    expression.walk([&indexing_map](AffineExpr expr) {
      CHECK(!(expr.getKind() == AffineExprKind::SymbolId &&
              indexing_map.IsRangeVarSymbol(
                  llvm::cast<mlir::AffineSymbolExpr>(expr))));
    });
  }

  MLIRContext* mlir_context = indexing_map.GetMLIRContext();

  // Cannot use AffineExpr::shiftSymbols, because it doesn't support negative
  // shifts.
  llvm::DenseMap<AffineExpr, AffineExpr> symbol_map;
  for (int i = 0; i < indexing_map.GetRTVarsCount(); i++) {
    symbol_map[getAffineSymbolExpr(indexing_map.GetRangeVarsCount() + i,
                                   mlir_context)] =
        getAffineSymbolExpr(i, mlir_context);
  }
  for (AffineExpr& expression : expressions) {
    expression = expression.replace(symbol_map);
  }

  return expressions;
}

// Gets a modified version of `expressions` where both the original dimensions
// and symbols are replaced with symbols.
//
// (dimensions)[symbols] -> ()[dimensions, symbols]
std::vector<AffineExpr> DimsToSymbols(std::vector<AffineExpr> expressions,
                                      const IndexingMap& indexing_map) {
  MLIRContext* mlir_context = indexing_map.GetMLIRContext();

  // Move symbols right
  for (AffineExpr& expression : expressions) {
    expression =
        expression.shiftSymbols(/*numSymbols=*/indexing_map.GetSymbolCount(),
                                /*shift=*/indexing_map.GetDimensionCount());
  }

  // Convert dimensions to symbols
  llvm::DenseMap<AffineExpr, AffineExpr> dim_to_symbol_map;
  for (int i = 0; i < indexing_map.GetDimensionCount(); i++) {
    dim_to_symbol_map[getAffineDimExpr(i, mlir_context)] =
        getAffineSymbolExpr(i, mlir_context);
  }
  for (AffineExpr& expression : expressions) {
    expression = expression.replace(dim_to_symbol_map);
  }

  return expressions;
}

// Internal helper that checks whether an affine map describes a tileable space.
// In simple terms, this currently returns true if "dimensions don't mix", i.e.,
// every result expression only refers to a single dimension (or symbol).
//
// TODO(b/328427138): this is too restrictive for expressions involving e.g.
// (output-to-input) split reshapes, where several symbols may appear within the
// same expression but still yield a tileable space. This will be handled in a
// forthcoming change.
bool IndexingMapDescribesTileableSpace(const IndexingMap& indexing_map) {
  for (AffineExpr result_expr : indexing_map.GetAffineMap().getResults()) {
    // Using a simple integer here might be overly restrictive, since there may
    // be cases where the same symbol appears in several places within the
    // expression. It is a bit unclear whether this is a case that would happen
    // in practice and whether we would be able to handle it well in all cases
    // if it did. For that reason, we err on the side of conservatism and
    // explicitly do not support such cases.
    int64_t num_hits = 0;
    result_expr.walk([&num_hits, &indexing_map](AffineExpr expr) {
      if ((expr.getKind() == AffineExprKind::DimId) ||
          (expr.getKind() == AffineExprKind::SymbolId &&
           indexing_map.IsRangeVarSymbol(
               llvm::cast<mlir::AffineSymbolExpr>(expr)))) {
        ++num_hits;
      }
    });

    if (num_hits > 1) {
      return false;
    }
  }
  return true;
}

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

struct SizeAndStrideExpression {
  AffineExpr size;
  AffineExpr stride;
};

// Extracts size and stride expressions from the operands to a modulo
// expression.
//
// TODO(b/326998704): Currently, this fails when the stride is not exactly unit.
std::optional<SizeAndStrideExpression> ExtractSizeAndStrideFromMod(
    AffineExpr lhs, AffineExpr modulus) {
  // TODO(b/326998704): derive constraints here, as well as the non-one stride
  // case, both in the code and in the proof.
  // Let f(d0) = d0 mod c. Then, given an input tile size n,
  // {f(x) | x in Fin(n)} contains:
  //   * n elements if n < c (and we add a constraint such that c | n);
  //   * c elements if n >= c (and we add a constraint such that n | c).
  // Given these constraints and assumptions, we derive
  //   card({f(x) | x in Fin(n)}) = n - ((n - 1) floordiv n) * n.
  // Proof:
  //   * n < c (and c | n):
  //       n - ((n - 1) floordiv c) * c
  //     = n - 0 * c               (n < c => n floordiv c == 0)
  //     = n
  //   * n >= c (and n | c):
  //       n - ((n - 1) floordiv c) * c
  //     = n - (n / c - 1) * c     (n | c => (n - 1) floordiv c = n / c - 1)
  //     = n - (n - c)
  //     = c
  CHECK(modulus.getKind() == AffineExprKind::Constant);
  if (auto dim_expr = llvm::dyn_cast<mlir::AffineDimExpr>(lhs)) {
    AffineExpr size =
        dim_expr - mlir::getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                               dim_expr - 1, modulus) *
                       modulus;
    // In this case, stride is effectively 1 mod modulus = 1.
    return SizeAndStrideExpression{
        size, /*stride=*/getAffineConstantExpr(1, lhs.getContext())};
  }

  return std::nullopt;
}

// Extracts size and stride expressions from the operands to a floordiv
// expression.
//
// TODO(b/326998704): Currently, this fails when the numerator of the stride
// is not exactly unit.
std::optional<SizeAndStrideExpression> ExtractSizeAndStrideFromFloorDiv(
    AffineExpr num, AffineExpr den) {
  if (den.getKind() != AffineExprKind::Constant) {
    return std::nullopt;
  }

  if (auto dim_expr = llvm::dyn_cast<mlir::AffineDimExpr>(num)) {
    // Let f(d0) = d0 floordiv c. Then, given an input tile size n,
    // {f(x) | x in Fin(n)} contains n ceildiv c elements, with stride
    // (1 ceildiv c) = 1.
    //
    // We represent `a ceildiv b` as `(a + b - 1) floordiv b`, since indexing
    // maps are not compatible with CeilDiv affine expressions.
    AffineExpr size = mlir::getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                                  dim_expr + (den - 1), den);
    return SizeAndStrideExpression{
        size, /*stride=*/getAffineConstantExpr(1, num.getContext())};
  }

  return std::nullopt;
}

std::optional<SizeAndStrideExpression> ExtractSizeAndStride(
    AffineExpr strided_indexing, absl::Span<Interval const> symbol_intervals) {
  MLIRContext* ctx = strided_indexing.getContext();
  // Deal with the symbol case (capturing a whole untiled dimension).
  // TODO(b/330906085): concatenating across a reduction dimension needs to be
  // handled by this code.
  if (auto symbol = llvm::dyn_cast<mlir::AffineSymbolExpr>(strided_indexing)) {
    const Interval& symbol_interval = symbol_intervals[symbol.getPosition()];
    if (symbol_interval.lower != 0) {
      return std::nullopt;
    }

    return SizeAndStrideExpression{
        /*size=*/getAffineConstantExpr(symbol_interval.upper + 1, ctx),
        /*stride=*/getAffineConstantExpr(1, ctx)};
  }

  AffineMapPrinter printer;

  // TODO(b/328427138): support multivariate size expressions.
  switch (strided_indexing.getKind()) {
    case AffineExprKind::DimId:
      return SizeAndStrideExpression{/*size=*/strided_indexing,
                                     /*stride=*/getAffineConstantExpr(1, ctx)};
    case mlir::AffineExprKind::Mul: {
      const auto mul = llvm::cast<mlir::AffineBinaryOpExpr>(strided_indexing);
      AffineExpr lhs = mul.getLHS();
      // The stride may not be fully collapsed if it is negative; in that case,
      // we need to extract the negative multiplier first.
      if (const auto rhs =
              llvm::dyn_cast<mlir::AffineConstantExpr>(mul.getRHS());
          rhs && rhs.getValue() == -1) {
        std::optional<SizeAndStrideExpression> maybe_size_and_stride =
            ExtractSizeAndStride(lhs, symbol_intervals);
        if (!maybe_size_and_stride.has_value()) {
          return std::nullopt;
        }

        return SizeAndStrideExpression{
            /*size=*/maybe_size_and_stride->size,
            /*stride=*/maybe_size_and_stride->stride * rhs};
      }
      CHECK(lhs.getKind() == AffineExprKind::DimId);
      return SizeAndStrideExpression{/*size=*/lhs,
                                     /*stride=*/mul.getRHS()};
    }
    case mlir::AffineExprKind::Mod: {
      auto mod = llvm::cast<mlir::AffineBinaryOpExpr>(strided_indexing);
      return ExtractSizeAndStrideFromMod(mod.getLHS(), mod.getRHS());
    }
    case mlir::AffineExprKind::FloorDiv: {
      auto floor_div = llvm::cast<mlir::AffineBinaryOpExpr>(strided_indexing);
      return ExtractSizeAndStrideFromFloorDiv(floor_div.getLHS(),
                                              floor_div.getRHS());
    };
    case mlir::AffineExprKind::Constant:
      return SizeAndStrideExpression{/*size=*/getAffineConstantExpr(1, ctx),
                                     /*stride=*/getAffineConstantExpr(0, ctx)};
    case mlir::AffineExprKind::SymbolId:
      VLOG(1) << "Encountered complex size expression involving symbol "
              << printer.ToString(strided_indexing);
      // It's currently not checked separately, but RTVars shouldn't appear in
      // the strided indexing expressions.
      return std::nullopt;
    case mlir::AffineExprKind::Add:
      // TODO(b/328427138): this should only be necessary in the multivariate
      // case, and will be implemented later.
      VLOG(1) << "Encountered complex strided indexing expression "
              << printer.ToString(strided_indexing);
      return std::nullopt;
    case mlir::AffineExprKind::CeilDiv:
      break;
  };
  LOG(FATAL) << "unreachable";
}

}  // anonymous namespace

/*static*/ std::optional<SymbolicTile> SymbolicTile::FromIndexingMap(
    const IndexingMap& indexing_map) {
  VLOG(1) << "SymbolicTile::FromIndexingMap: " << indexing_map.ToString();

  // TODO(b/328427138): handle multiple symbols in a single tile to support
  // merging dimensions.
  if (!IndexingMapDescribesTileableSpace(indexing_map)) {
    VLOG(1) << "Not a tileable indexing map";
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

  std::vector<AffineExpr> size_expressions;
  std::vector<AffineExpr> stride_expressions;
  size_expressions.reserve(offset_expressions.size());
  stride_expressions.reserve(offset_expressions.size());

  // strided_indexing_expressions =
  //     f(x0, ..., x{M-1})[x{M}, ..., x{M+P-1}] - offset_expressions
  for (auto [composite_indexing, offset] :
       llvm::zip(input_affine_map.getResults(), offset_expressions)) {
    std::optional<SizeAndStrideExpression> maybe_size_and_stride =
        ExtractSizeAndStride(composite_indexing - offset,
                             indexing_map.GetSymbolBounds());
    if (!maybe_size_and_stride.has_value()) {
      // Retry with a simplified expression.
      // For example `(d0 + s0 - s0)` will be simplified to `d0`.
      // But the simplification doesn't help when it rewrites `mod` to
      // `floordiv` & `add`, so at first we try without simplification.
      maybe_size_and_stride = ExtractSizeAndStride(
          simplifyAffineExpr(composite_indexing - offset,
                             input_affine_map.getNumDims(),
                             input_affine_map.getNumSymbols()),
          indexing_map.GetSymbolBounds());
    }
    if (!maybe_size_and_stride.has_value()) {
      VLOG(1) << "No size and stride extracted";
      return std::nullopt;
    }
    size_expressions.push_back(maybe_size_and_stride->size);
    stride_expressions.push_back(maybe_size_and_stride->stride);
  }

  // Eliminate negative strides and recalculate offsets.
  std::vector<AffineExpr> dim_replacements, sym_replacements;
  for (auto [offset, size, stride] :
       llvm::zip(offset_expressions, size_expressions, stride_expressions)) {
    auto constant = llvm::dyn_cast<mlir::AffineConstantExpr>(stride);
    if (!constant) {
      AffineMapPrinter printer;
      VLOG(1) << "Unexpected non-constant stride expression: "
              << printer.ToString(stride);
      return std::nullopt;
    }
    if (constant.getValue() < 0) {
      offset = offset + size * stride - stride;
      stride = -stride;
    }
  }

  // DimVars in `indexing_map` represent indices, but in `tile_map` they will
  // represent the size of the tile. So we need to add 1 to the bounds.
  // For example: indices: [0, 9] -> sizes: [1, 10].
  std::vector<DimVar> tile_sizes = indexing_map.GetDimVars();
  for (DimVar& tile_size : tile_sizes) {
    tile_size.bounds.lower += 1;
    tile_size.bounds.upper += 1;
  }

  std::vector<AffineExpr> results;
  absl::c_move(WithoutRangeVars(std::move(offset_expressions), indexing_map),
               std::back_inserter(results));
  absl::c_move(WithoutRangeVars(std::move(size_expressions), indexing_map),
               std::back_inserter(results));
  absl::c_move(WithoutRangeVars(std::move(stride_expressions), indexing_map),
               std::back_inserter(results));

  AffineMap tile_affine_map =
      AffineMap::get(/*dimCount=*/tile_sizes.size(),
                     /*symbolCount=*/indexing_map.GetRTVarsCount(),
                     /*results=*/results,
                     /*context=*/indexing_map.GetMLIRContext());

  // TODO(b/326998704): Pass constraints derived in ExtractSizeAndStrideFromMod
  // (and possibly other places) to the constructor. Also consider if we can
  // derive any constraints from the constraints of the original indexing map.
  IndexingMap tile_map(
      /*affine_map=*/std::move(tile_affine_map),
      /*dimensions=*/std::move(tile_sizes),
      /*range_vars=*/{},
      /*rt_vars=*/indexing_map.GetRTVars());
  VLOG(1) << "tile_map: " << tile_map.ToString();
  return SymbolicTile(std::move(tile_map));
}

std::string SymbolicTile::RtVarsToString(
    const AffineMapPrinter& printer) const {
  std::string s;
  std::stringstream ss(s);
  PrintRTVars(tile_map_.GetRTVars(),
              /*first_rt_var_symbol_index=*/tile_map_.GetDimensionCount(), ss,
              printer);
  return ss.str();
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
  printer.Print(out, offset_map());
  out << "\n\tsize_map: ";
  printer.Print(out, size_map());
  out << "\n\tstride_map: ";
  printer.Print(out, stride_map());
  out << "\n\trt_vars: ";
  PrintRTVars(tile_map_.GetRTVars(),
              /*first_rt_var_symbol_index=*/tile_map_.GetDimensionCount(), out,
              printer);
  out << "\n";
}

namespace {
// The results of `SymbolicTile::tile_map_` can be split into 3 groups: offsets,
// sizes, and strides.
constexpr int kNumComponentsPerTiledDimension = 3;
}  // namespace

mlir::AffineMap SymbolicTile::offset_map() const {
  llvm::ArrayRef<AffineExpr> results = tile_map_.GetAffineMap().getResults();
  CHECK_EQ(results.size() % kNumComponentsPerTiledDimension, 0);
  llvm::ArrayRef<AffineExpr> offsets(
      results.begin(),
      results.begin() + results.size() / kNumComponentsPerTiledDimension);
  // RTVars are included in the symbols.
  return AffineMap::get(
      /*dimCount=*/0,
      /*symbolCount=*/tile_map_.GetAffineMap().getNumDims() +
          tile_map_.GetAffineMap().getNumSymbols(),
      /*results=*/DimsToSymbols(offsets, tile_map_),
      /*context=*/tile_map_.GetAffineMap().getContext());
}

mlir::AffineMap SymbolicTile::size_map() const {
  llvm::ArrayRef<AffineExpr> results = tile_map_.GetAffineMap().getResults();
  CHECK_EQ(results.size() % kNumComponentsPerTiledDimension, 0);
  llvm::ArrayRef<AffineExpr> offsets(
      results.begin() + results.size() / kNumComponentsPerTiledDimension,
      results.begin() + results.size() / kNumComponentsPerTiledDimension * 2);
  // RTVars are *not* included in the symbols.
  return AffineMap::get(
      /*dimCount=*/0,
      /*symbolCount=*/tile_map_.GetAffineMap().getNumDims() +
          tile_map_.GetAffineMap().getNumSymbols() - tile_map_.GetRTVarsCount(),
      /*results=*/DimsToSymbols(offsets, tile_map_),
      /*context=*/tile_map_.GetAffineMap().getContext());
}

mlir::AffineMap SymbolicTile::stride_map() const {
  llvm::ArrayRef<AffineExpr> results = tile_map_.GetAffineMap().getResults();
  CHECK_EQ(results.size() % kNumComponentsPerTiledDimension, 0);
  llvm::ArrayRef<AffineExpr> offsets(
      results.begin() + results.size() / kNumComponentsPerTiledDimension * 2,
      results.end());
  // RTVars are *not* included in the symbols.
  return AffineMap::get(
      /*dimCount=*/0,
      /*symbolCount=*/tile_map_.GetAffineMap().getNumDims() +
          tile_map_.GetAffineMap().getNumSymbols() - tile_map_.GetRTVarsCount(),
      /*results=*/DimsToSymbols(offsets, tile_map_),
      /*context=*/tile_map_.GetAffineMap().getContext());
}

}  // namespace gpu
}  // namespace xla
