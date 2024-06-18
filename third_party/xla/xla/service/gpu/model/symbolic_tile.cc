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
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
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

using ::mlir::AffineConstantExpr;
using ::mlir::AffineDimExpr;
using ::mlir::AffineExpr;
using ::mlir::AffineExprKind;
using ::mlir::AffineMap;
using ::mlir::AffineSymbolExpr;
using ::mlir::getAffineConstantExpr;
using ::mlir::getAffineDimExpr;
using ::mlir::MLIRContext;
using ConstraintMap = SymbolicTile::ConstraintMap;

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
  ConstraintMap constraints;
  bool is_satisfiable;

  SizeAndStrideExpression(AffineExpr size, AffineExpr stride,
                          ConstraintMap constraints = ConstraintMap(),
                          bool is_satisfiable = true)
      : size(std::move(size)),
        stride(std::move(stride)),
        constraints(std::move(constraints)),
        is_satisfiable(is_satisfiable) {}
};

// Extracts size and stride expressions from the operands to a modulo
// expression.
//
// TODO(b/326998704): Currently, this fails when the stride is not exactly unit.
std::optional<SizeAndStrideExpression> ExtractSizeAndStrideFromMod(
    AffineExpr lhs, AffineExpr modulus) {
  // TODO(b/326998704): finish deriving constraints here, as well as the non-one
  // stride case, both in the code and in the proof.
  // Let f(d0) = d0 mod c. Then, given an input tile size n,
  // {f(x) | x in Fin(n)} contains:
  //   * n elements if n < c (and we add a constraint that c % n == 0)
  //   * c elements if n >= c (and we add a constraint that n % c == 0)
  // Given these constraints and assumptions, we derive
  //   card({f(x) | x in Fin(n)}) = n - ((n - 1) floordiv n) * n.
  // Proof:
  //   * n < c (and c % n == 0):
  //       n - ((n - 1) floordiv c) * c
  //     = n - 0 * c              (n < c => n floordiv c == 0)
  //     = n
  //   * n >= c (and n % c == 0):
  //       n - ((n - 1) floordiv c) * c
  //     = n - (n / c - 1) * c    (n % c == 0 => (n - 1) floordiv c = n / c - 1)
  //     = n - (n - c)
  //     = c
  CHECK(modulus.getKind() == AffineExprKind::Constant);
  if (auto dim_expr = llvm::dyn_cast<mlir::AffineDimExpr>(lhs)) {
    AffineExpr size =
        dim_expr - mlir::getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                               dim_expr - 1, modulus) *
                       modulus;

    AffineExpr constrained_expr =
        getAffineSymbolExpr(dim_expr.getPosition(), lhs.getContext()) % modulus;
    ConstraintMap constraints;
    // TODO(b/334043867): we only add a constraint for n being a multiple of c
    // while we do not support disjunctions.
    constraints.insert({constrained_expr, Interval{/*lower=*/0, /*upper=*/0}});

    // In this case, stride is effectively 1 mod modulus = 1.
    return SizeAndStrideExpression(
        size, /*stride=*/getAffineConstantExpr(1, lhs.getContext()),
        std::move(constraints));
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
    return SizeAndStrideExpression(
        size, /*stride=*/getAffineConstantExpr(1, num.getContext()));
  }

  return std::nullopt;
}

// See documentation of `DestructureSummation` for an explanation of the
// algorithm.
void DestructureSummationImpl(AffineExpr expr,
                              std::vector<AffineExpr>& summands) {
  switch (expr.getKind()) {
    case AffineExprKind::Add: {
      const auto add = llvm::cast<mlir::AffineBinaryOpExpr>(expr);
      DestructureSummationImpl(add.getLHS(), summands);
      DestructureSummationImpl(add.getRHS(), summands);
      break;
    }
    default:
      // The expression is not a sum.
      summands.push_back(expr);
      break;
  }
}

// Given an n-ary summation of expressions `e0 + e1 + ... + e{n-1}` with
// arbitrary order of association, returns the vector `(e0, e1, ..., e{n-1})`.
// The order of the returned subexpressions is not guaranteed to match the order
// in which they appear in the original expression.
//
// AffineExprKind::Add should be the operation that binds the least tightly,
// allowing us to simply recursively destructure expressions until we reach an
// AffineExprKind that is not an AffineExprKind::Add.
//
// Note that this will only work correctly for expressions that do no
// factoring/grouping of summands such as `(d0 + d1) * c` or `(d0 + d1) mod c`.
// It's unclear at this point whether this restriction will prove problematic,
// but it isn't really worth thinking about until we are sure this actually
// has practical implications.
std::vector<AffineExpr> DestructureSummation(AffineExpr expr) {
  std::vector<AffineExpr> summands;
  DestructureSummationImpl(expr, summands);
  return summands;
}

std::optional<SizeAndStrideExpression> ExtractSizeAndStride(
    AffineExpr strided_indexing, absl::Span<Interval const> dimension_intervals,
    absl::Span<Interval const> symbol_intervals);

// Given a multivariate summation of strided indexing expression, extracts a
// size and a stride for each summand. Returns std::nullopt if extraction fails
// for any of the summands.
std::optional<std::vector<SizeAndStrideExpression>>
ExtractSizesAndStridesFromMultivariateSummation(
    AffineExpr summation, absl::Span<Interval const> dimension_intervals,
    absl::Span<Interval const> symbol_intervals) {
  std::vector<AffineExpr> summands = DestructureSummation(summation);

  std::vector<SizeAndStrideExpression> sizes_and_strides;
  sizes_and_strides.reserve(summands.size());
  for (AffineExpr summand : summands) {
    std::optional<SizeAndStrideExpression> maybe_size_and_stride =
        ExtractSizeAndStride(summand, dimension_intervals, symbol_intervals);
    if (!maybe_size_and_stride.has_value()) {
      VLOG(1) << "Couldn't extract size and stride from "
              << AffineMapPrinter().ToString(summand);
      return std::nullopt;
    }
    sizes_and_strides.push_back(*maybe_size_and_stride);
  }
  return sizes_and_strides;
}

// Given a list of sizes and strides, returns the product of all sizes.
AffineExpr CombineSizes(
    absl::Span<SizeAndStrideExpression const> sizes_and_strides) {
  CHECK(!sizes_and_strides.empty());
  AffineExpr product =
      getAffineConstantExpr(1, sizes_and_strides[0].size.getContext());
  for (const SizeAndStrideExpression& size_and_stride : sizes_and_strides) {
    product = product * size_and_stride.size;
  }
  return product;
}

// Returns an affine expression logically equivalent to
//   `eq_param != 1 ? true_expr : false_expr`.
// `eq_param` is assumed to be able to be in the inclusive range
//    {1, 2, ..., eq_param_inclusive_upper_bound}.
AffineExpr IfNeqOne(AffineExpr eq_param, AffineExpr true_expr,
                    AffineExpr false_expr,
                    int64_t eq_param_inclusive_upper_bound) {
  // Let e = eq_param, and b = eq_param_inclusive_bound, then we have:
  //     1 <= e <= b
  // <=> -b <= e - b - 1 <= -1              (subtract (b + 1))
  // <=> 1 <= b + 1 - e <= b                (negate)
  // <=> 0 <= (b + 1 - e) floordiv b <= 1   (divide by b)
  //
  // Since (b + 1 - e) floordiv b is an integer, it can only take values 0 or 1.
  // Let's prove that
  //   (b + 1 - e) floordiv b = 1 <=> e = 1.
  //
  // * If e = 1, then (b + 1 - e) floordiv b = (b + 1 - 1) floordiv b = 1.
  // * If e != 1, then 1 < e since 1 is the lower bound for e.
  //     1 < e <=> -e < -1                       (negate)
  //           <=> b + 1 - e < b                 (add b + 1)
  //           <=> (b - e + 1) floordiv b < 1.   (divide by b)
  //   We also know that 0 <= (b + 1 - e) floordiv b. Therefore, we have that
  //     (b - e + 1) floordiv b = 0.
  //
  // Thus,
  //   (b + 1 - e) floordiv b = 1 <=> e = 1, and
  //   (b + 1 - e) floordiv b = 0 <=> e != 1
  // hold.
  AffineExpr b = getAffineConstantExpr(eq_param_inclusive_upper_bound,
                                       eq_param.getContext());
  AffineExpr condition = mlir::getAffineBinaryOpExpr(AffineExprKind::FloorDiv,
                                                     b + 1 - eq_param, b);

  return condition * false_expr + (1 - condition) * true_expr;
}

// Sorts a list of `SizeAndStrideExpression`s by stride. There is a precondition
// that all strides are constant.
void SortByStride(std::vector<SizeAndStrideExpression>& sizes_and_strides) {
  absl::c_sort(sizes_and_strides, [](const SizeAndStrideExpression& sas1,
                                     const SizeAndStrideExpression& sas2) {
    int64_t stride1 = llvm::cast<AffineConstantExpr>(sas1.stride).getValue();
    int64_t stride2 = llvm::cast<AffineConstantExpr>(sas2.stride).getValue();
    return stride1 < stride2;
  });
}

// Given a list of sizes and strides, combines the strides into a single
// expression if it is possible.
//
// The current implementation expects that each size captures a single dimension
// parameter.
//
// Let s be an n-dimensional shape that we want to fully collapse. In order to
// be propagated successfully through the collapse, the pattern of the tiling of
// s will have to look like the following (in row-major order):
//   (1*, partial_dim?, full_dims*, 1*)
// where full_dims are dimensions along which we capture all the elements
// we can based on the corresponding stride, and partial_dim is a dimension that
// can be captured with an arbitrary tile.
//
// In that case, the stride will be the stride corresponding to the minormost
// dimension in which we capture more than a single element. This corresponds
// to the size expression `e` with the smallest stride such that `e` evaluates
// to another value than 1. Algorithmically, this can be represented as a series
// of nested if statements:
//   if size0 != 1 then stride0 else (if size1 != 1 then stride1 else ...)
// where {size,stride}i = size_and_strides[i].{size,stride} (sizes_and_strides
// being sorted in ascending order of stride).
//
// We generate this nest.
//
// If all the sizes are 1, then return a zero stride. Note that this
// value is arbitrarily chosen.
std::optional<AffineExpr> CombineStrides(
    std::vector<SizeAndStrideExpression> sizes_and_strides,
    absl::Span<Interval const> dimension_intervals) {
  CHECK(!sizes_and_strides.empty());
  for (const SizeAndStrideExpression& size_and_stride : sizes_and_strides) {
    if (size_and_stride.stride.getKind() != AffineExprKind::Constant) {
      VLOG(1) << "Attempted to combine non-constant stride: "
              << AffineMapPrinter().ToString(size_and_stride.stride);
      return std::nullopt;
    }

    // We know the exact bounds of dimension parameters, since they correspond
    // to parameters of the initial indexing map. It follows that if a size
    // expression is exactly a dimension parameter, we know its exact bounds.
    //
    // If a size is not exactly a dimension parameter, then it is dubious
    // whether we know the bounds---and may thus calculate wrong strides.
    if (size_and_stride.size.getKind() != AffineExprKind::DimId) {
      VLOG(1) << "Attempted to combine strides but got non-dimension size "
              << AffineMapPrinter().ToString(size_and_stride.size);
      return std::nullopt;
    }
  }

  SortByStride(sizes_and_strides);

  for (auto [dim_id, size_and_stride] : llvm::enumerate(sizes_and_strides)) {
    int64_t stride =
        llvm::cast<AffineConstantExpr>(size_and_stride.stride).getValue();

    // The minormost stride can be anything, but we expect every subsequent
    // stride to be exactly `p_stride * p_size` where `p_size` is the upper
    // bound of the size expression of the previous dimension and `p_stride` is
    // its stride expression.
    //
    // For simplicity, we assume that each size expression captures a single
    // dimension parameter.
    if (dim_id > 0) {
      const SizeAndStrideExpression& previous_size_and_stride =
          sizes_and_strides[dim_id - 1];
      const auto previous_dimension =
          llvm::cast<AffineDimExpr>(previous_size_and_stride.size);
      const Interval& previous_size_interval =
          dimension_intervals[previous_dimension.getPosition()];
      if (previous_size_interval.lower != 0) {
        // TODO(bchetioui): I think we may need to handle this to have reshapes
        // working well with concatenations. Nevertheless, we can take a look
        // later.
        VLOG(1) << "Attempted to combine strides but got dimension "
                << AffineMapPrinter().ToString(previous_dimension)
                << " with lower bound " << previous_size_interval.lower
                << " != 0";
        return std::nullopt;
      }

      int64_t previous_stride =
          llvm::cast<AffineConstantExpr>(previous_size_and_stride.stride)
              .getValue();
      // We need to add 1 to the upper bound of the interval to describe the
      // number of elements being captured, since the interval bounds are
      // inclusive.
      if ((previous_size_interval.upper + 1) * previous_stride != stride) {
        VLOG(1) << "Attempted to combine strides but stride did not grow "
                << "exactly as expected: got "
                << (previous_size_interval.upper + 1) << " * "
                << previous_stride << " != " << stride;
        return std::nullopt;
      }
    }
  }

  // Produce a nested if statement as described in the function's documentation.
  MLIRContext* ctx = sizes_and_strides[0].stride.getContext();
  AffineExpr nested_if = getAffineConstantExpr(0, ctx);
  for (auto size_and_stride_it = sizes_and_strides.rbegin();
       size_and_stride_it != sizes_and_strides.rend(); ++size_and_stride_it) {
    AffineExpr size = size_and_stride_it->size;
    AffineExpr stride = size_and_stride_it->stride;
    const Interval& size_interval =
        dimension_intervals[llvm::cast<AffineDimExpr>(size).getPosition()];
    nested_if = IfNeqOne(size, stride, nested_if, size_interval.upper + 1);
  }

  return nested_if;
}

// See documentation of `CombineSizes` and `CombineStrides` for an explanation
// of how sizes and strides are combined.
std::optional<SizeAndStrideExpression> CombineSizesAndStrides(
    std::vector<SizeAndStrideExpression> sizes_and_strides,
    absl::Span<Interval const> dimension_intervals) {
  CHECK(!sizes_and_strides.empty());

  std::optional<ConstraintMap> maybe_constraints = ConstraintMap();

  for (const SizeAndStrideExpression& size_and_stride : sizes_and_strides) {
    maybe_constraints = MergeConstraintMapIfPresentAndCompatible(
        std::move(maybe_constraints), size_and_stride.constraints);
    if (!maybe_constraints.has_value()) {
      break;
    }
  }

  ConstraintMap constraints;
  bool is_satisfiable = true;

  // Handle cases that we don't know how to process by constructing a
  // ConstraintMap with an unsatisfiable constraint.
  if (maybe_constraints.has_value()) {
    constraints = std::move(*maybe_constraints);
  } else {
    is_satisfiable = false;
  }

  AffineExpr size = CombineSizes(sizes_and_strides);
  std::optional<AffineExpr> stride =
      CombineStrides(std::move(sizes_and_strides), dimension_intervals);
  if (!stride.has_value()) {
    return std::nullopt;
  }

  // TODO(b/326998704): handle reshape constraints here.
  return SizeAndStrideExpression(size, *stride, std::move(constraints),
                                 is_satisfiable);
}

std::optional<SizeAndStrideExpression> ExtractSizeAndStride(
    AffineExpr strided_indexing, absl::Span<Interval const> dimension_intervals,
    absl::Span<Interval const> symbol_intervals) {
  MLIRContext* ctx = strided_indexing.getContext();
  // Deal with the symbol case (capturing a whole untiled dimension).
  // TODO(b/330906085): concatenating across a reduction dimension needs to be
  // handled by this code.
  if (auto symbol = llvm::dyn_cast<AffineSymbolExpr>(strided_indexing)) {
    const Interval& symbol_interval = symbol_intervals[symbol.getPosition()];
    if (symbol_interval.lower != 0) {
      return std::nullopt;
    }

    return SizeAndStrideExpression(
        /*size=*/getAffineConstantExpr(symbol_interval.upper + 1, ctx),
        /*stride=*/getAffineConstantExpr(1, ctx));
  }

  AffineMapPrinter printer;

  // TODO(b/328427138): support multivariate size expressions.
  switch (strided_indexing.getKind()) {
    case AffineExprKind::DimId:
      return SizeAndStrideExpression(/*size=*/strided_indexing,
                                     /*stride=*/getAffineConstantExpr(1, ctx));
    case mlir::AffineExprKind::Mul: {
      const auto mul = llvm::cast<mlir::AffineBinaryOpExpr>(strided_indexing);
      AffineExpr lhs = mul.getLHS();
      // The stride may not be fully collapsed if it is negative; in that case,
      // we need to extract the negative multiplier first.
      if (const auto rhs = llvm::dyn_cast<AffineConstantExpr>(mul.getRHS());
          rhs && rhs.getValue() == -1) {
        std::optional<SizeAndStrideExpression> maybe_size_and_stride =
            ExtractSizeAndStride(lhs, dimension_intervals, symbol_intervals);
        if (!maybe_size_and_stride.has_value()) {
          return std::nullopt;
        }

        return SizeAndStrideExpression(
            /*size=*/maybe_size_and_stride->size,
            /*stride=*/maybe_size_and_stride->stride * rhs);
      }
      CHECK(lhs.getKind() == AffineExprKind::DimId);
      return SizeAndStrideExpression(/*size=*/lhs,
                                     /*stride=*/mul.getRHS());
    }
    case mlir::AffineExprKind::Mod: {
      auto mod = llvm::cast<mlir::AffineBinaryOpExpr>(strided_indexing);
      return ExtractSizeAndStrideFromMod(mod.getLHS(), mod.getRHS());
    }
    case mlir::AffineExprKind::FloorDiv: {
      auto floor_div = llvm::cast<mlir::AffineBinaryOpExpr>(strided_indexing);
      return ExtractSizeAndStrideFromFloorDiv(floor_div.getLHS(),
                                              floor_div.getRHS());
    }
    case mlir::AffineExprKind::Constant:
      return SizeAndStrideExpression(/*size=*/getAffineConstantExpr(1, ctx),
                                     /*stride=*/getAffineConstantExpr(0, ctx));
    case mlir::AffineExprKind::SymbolId:
      VLOG(1) << "Encountered complex size expression involving symbol "
              << printer.ToString(strided_indexing);
      // It's currently not checked separately, but RTVars shouldn't appear in
      // the strided indexing expressions.
      return std::nullopt;
    case mlir::AffineExprKind::Add: {
      // TODO(b/328427138): this should only be necessary in the multivariate
      // case, and will be implemented later.
      std::optional<std::vector<SizeAndStrideExpression>>
          maybe_sizes_and_strides =
              ExtractSizesAndStridesFromMultivariateSummation(
                  strided_indexing, dimension_intervals, symbol_intervals);
      if (!maybe_sizes_and_strides.has_value()) {
        return std::nullopt;
      }
      return CombineSizesAndStrides(std::move(*maybe_sizes_and_strides),
                                    dimension_intervals);
    }
    case mlir::AffineExprKind::CeilDiv:
      break;
  };
  LOG(FATAL) << "unreachable";
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
  tmp_indexing_map.Simplify();

  CHECK_EQ(tmp_indexing_map.GetAffineMap().getResults().size(), 1);
  return tmp_indexing_map.GetAffineMap().getResults().back();
}

}  // anonymous namespace

std::optional<ConstraintMap> MergeConstraintMapIfPresentAndCompatible(
    std::optional<ConstraintMap> maybe_first_map,
    const ConstraintMap& second_map) {
  if (!maybe_first_map.has_value()) {
    return std::nullopt;
  }

  ConstraintMap& first_map = *maybe_first_map;

  for (const auto& [expr, interval] : second_map) {
    if (first_map.contains(expr)) {
      AffineMapPrinter printer;
      VLOG(1) << "Got two different constraints for expression "
              << printer.ToString(expr);
      return std::nullopt;
    }

    first_map.insert({expr, interval});
  }

  return first_map;
}

/*static*/ std::optional<SymbolicTile> SymbolicTile::FromIndexingMap(
    const IndexingMap& indexing_map) {
  VLOG(1) << "SymbolicTile::FromIndexingMap: " << indexing_map.ToString();

  // We do not handle indexing maps with pre-existing constraints for now.
  if (indexing_map.GetConstraintsCount() != 0) {
    VLOG(1) << "Deriving symbolic tile from indexing map with pre-existing "
            << "constraints might produce spurious constraints. Bailing out.";
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

  std::optional<ConstraintMap> maybe_constraints = ConstraintMap();
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

    maybe_constraints = MergeConstraintMapIfPresentAndCompatible(
        std::move(maybe_constraints), maybe_size_and_stride->constraints);
  }

  ConstraintMap constraints;
  bool is_satisfiable = true;

  // Handle cases that we don't know how to process by constructing a
  // ConstraintMap with an unsatisfiable constraint.
  if (maybe_constraints.has_value()) {
    constraints = std::move(*maybe_constraints);
  } else {
    is_satisfiable = false;
  }

  // Eliminate negative strides and recalculate offsets.
  // TODO(b/340555497): handle normalization of more complex expressions.
  std::vector<AffineExpr> dim_replacements, sym_replacements;
  for (auto [offset, size, stride] :
       llvm::zip(offset_expressions, size_expressions, stride_expressions)) {
    auto constant = llvm::dyn_cast<AffineConstantExpr>(stride);
    if (constant && constant.getValue() < 0) {
      offset = offset + size * stride - stride;
      stride = -stride;
    } else if (!constant) {
      AffineMapPrinter printer;
      VLOG(1) << "Unexpected non-constant stride expression: "
              << printer.ToString(stride);
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
  absl::c_move(std::move(offset_expressions), std::back_inserter(results));
  absl::c_move(std::move(size_expressions), std::back_inserter(results));
  absl::c_move(std::move(stride_expressions), std::back_inserter(results));

  AffineMap tile_affine_map =
      AffineMap::get(/*dimCount=*/tile_sizes.size(),
                     /*symbolCount=*/indexing_map.GetSymbolCount(),
                     /*results=*/results,
                     /*context=*/indexing_map.GetMLIRContext());

  // TODO(b/326998704): Can we derive any constraint from the constraints of
  // the original indexing map?
  IndexingMap tile_map(
      /*affine_map=*/std::move(tile_affine_map),
      /*dimensions=*/std::move(tile_sizes),
      /*range_vars=*/indexing_map.GetRangeVars(),
      /*rt_vars=*/indexing_map.GetRTVars());
  tile_map.RemoveUnusedSymbols();
  CHECK_EQ(tile_map.GetRangeVarsCount(), 0);

  VLOG(1) << "tile_map: " << tile_map.ToString();
  return SymbolicTile(std::move(tile_map), constraints, is_satisfiable);
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
  const std::vector<RTVar>& rt_vars = tile_map_.GetRTVars();
  if (!rt_vars.empty()) {
    out << "\n\trt_vars: ";
    PrintRTVars(rt_vars,
                /*first_rt_var_symbol_index=*/tile_map_.GetDimensionCount(),
                out, printer);
  }
  if (!constraints_.empty() && is_satisfiable_) {
    out << "\n\tconstraints: ";
    // Accumulate constraints in a vector in order to put them in lexicographic
    // order and to get deterministic output.
    std::vector<std::string> constraint_strings;
    constraint_strings.reserve(constraints_.size());
    for (const auto& [expr, interval] : constraints_) {
      std::stringstream ss;
      printer.Print(ss, expr);
      ss << " in ";
      interval.Print(ss);
      constraint_strings.push_back(ss.str());
    }
    std::sort(constraint_strings.begin(), constraint_strings.end());
    for (absl::string_view constraint_string : constraint_strings) {
      out << "\n\t" << constraint_string;
    }
  } else if (!is_satisfiable_) {
    out << "\n\tconstraints: ";
    out << "\n\tunsatisfiable";
  }
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
