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
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/service/gpu/model/affine_map_evaluator.h"

namespace xla {
namespace gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::AffineBinaryOpExpr;
using ::mlir::AffineConstantExpr;
using ::mlir::AffineDimExpr;
using ::mlir::AffineExpr;
using ::mlir::AffineExprKind;
using ::mlir::AffineMap;
using ::mlir::AffineSymbolExpr;
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

struct SizeAndStrideExpression {
  AffineExpr size;
  AffineExpr stride;
  ConstraintExpression constraints;

  explicit SizeAndStrideExpression(
      AffineExpr size, int64_t stride,
      ConstraintExpression constraints =
          ConstraintExpression::GetAlwaysSatisfied())
      : SizeAndStrideExpression(
            size, getAffineConstantExpr(stride, size.getContext()),
            std::move(constraints)) {}

  explicit SizeAndStrideExpression(
      AffineExpr size, AffineExpr stride,
      ConstraintExpression constraints =
          ConstraintExpression::GetAlwaysSatisfied())
      : size(size), stride(stride), constraints(std::move(constraints)) {}
};

// Extracts size and stride expressions from the operands to a modulo
// expression.
//
// TODO(b/349487906): Currently, this fails when the stride is not exactly unit.
std::optional<SizeAndStrideExpression> ExtractSizeAndStrideFromMod(
    AffineExpr lhs, AffineExpr modulus) {
  if (modulus.getKind() != AffineExprKind::Constant) {
    return std::nullopt;
  }
  // TODO(b/349487906): handle the non-one stride case, both in the code and in
  // the proof.

  // Let f(d0) = d0 mod c. Then, given an input tile size n,
  // {f(x) | x in Fin(n)} contains:
  //   * n elements if n < c (and we add a constraint that c % n == 0)
  //   * c elements if n >= c (and we add a constraint that n % c == 0)
  // Given these constraints and assumptions, we derive
  //   card({f(x) | x in Fin(n)}) = n - ((n - 1) floordiv c) * c.
  // Proof:
  //   * n < c (and c % n == 0):
  //       n - ((n - 1) floordiv c) * c
  //     = n - 0 * c              (n < c => n floordiv c == 0)
  //     = n
  //   * n >= c (and n % c == 0):
  //       n - ((n - 1) floordiv c) * c
  //     = n - (n / c - 1) * c    (n % c == 0 => (n - 1) floordiv c = n / c - 1)
  //     = n - (n - c) = c
  auto make_result = [&](AffineExpr dividend) {
    AffineExpr size = dividend - (dividend - 1).floorDiv(modulus) * modulus;
    // modulus % dividend == 0 || dividend % modulus == 0.
    Interval zero_interval{/*lower=*/0, /*upper=*/0};
    ConstraintExpression constraints =
        Constraint{modulus % dividend, zero_interval} ||
        Constraint{dividend % modulus, zero_interval};
    // Stride is effectively 1 % modulus = 1.
    return SizeAndStrideExpression(size, /*stride=*/1, std::move(constraints));
  };

  if (llvm::isa<AffineDimExpr>(lhs)) {
    return make_result(lhs);
  }

  if (lhs.getKind() != AffineExprKind::FloorDiv) {
    return std::nullopt;
  }
  AffineExpr dim = llvm::cast<AffineBinaryOpExpr>(lhs).getLHS();
  AffineExpr den = llvm::cast<AffineBinaryOpExpr>(lhs).getRHS();
  if (dim.getKind() != AffineExprKind::DimId ||
      den.getKind() != AffineExprKind::Constant) {
    return std::nullopt;
  }

  // Let f(d0) = d0 floordiv d mod c. Then, given an input tile size n,
  // {f(x) | x in Fin(n)} contains:
  //   * n ceildiv d elements if n ceildiv d < c
  //     (and we add a constraint that c % (n ceildiv d) == 0)
  //   * c elements if n ceildiv d >= c
  //     (and we add a constraint that (n ceildiv d) % c == 0)
  // Given these constraints and assumptions, we derive
  //   card({f(x) | x in Fin(n)}) =
  //       n ceildiv d - (n ceildiv d - 1) floordiv c * c
  // Proof:
  //   * n ceildiv d < c (and c % (n ceildiv d) == 0):
  //       n ceildiv d - (n ceildiv d - 1) floordiv c * c
  //         because (n ceildiv d) < c => (n ceildiv d - 1) floordiv c == 0
  //     = n ceildiv d - 0 * c
  //     = n ceildiv d
  //   * n ceildiv d >= c (and (n ceildiv d) % c == 0):
  //       n ceildiv d - (n ceildiv d - 1) floordiv c * c
  //         because (n ceildiv d) % c == 0
  //                           => n ceildiv d floordiv c == n ceildiv d / c
  //     = n ceildiv d - (n ceildiv d / c - 1) * c
  //     = n ceildiv d - (n ceildiv d - c)
  //     = c
  // We represent `n ceildiv d` as `(n + d - 1) floordiv d`, since indexing
  // maps are not compatible with CeilDiv affine expressions.
  return make_result((dim + (den - 1)).floorDiv(den));
}

// Extracts size and stride expressions from the operands to a floordiv
// expression.
//
// TODO(b/349487906): Currently, this fails when the numerator of the stride
// is not exactly unit.
std::optional<SizeAndStrideExpression> ExtractSizeAndStrideFromFloorDiv(
    AffineExpr num, AffineExpr den) {
  if (den.getKind() != AffineExprKind::Constant) {
    return std::nullopt;
  }
  if (num.getKind() != AffineExprKind::DimId) {
    return std::nullopt;
  }

  // Let f(d0) = d0 floordiv c. Then, given an input tile size n,
  // {f(x) | x in Fin(n)} contains n ceildiv c elements, with stride
  // (1 ceildiv c) = 1.
  //
  // We represent `a ceildiv b` as `(a + b - 1) floordiv b`, since indexing
  // maps are not compatible with CeilDiv affine expressions.
  return SizeAndStrideExpression((num + (den - 1)).floorDiv(den), /*stride=*/1);
}

// See documentation of `DestructureSummation` for an explanation of the
// algorithm.
void DestructureSummationImpl(AffineExpr expr,
                              std::vector<AffineExpr>& summands) {
  switch (expr.getKind()) {
    case AffineExprKind::Add: {
      const auto add = llvm::cast<AffineBinaryOpExpr>(expr);
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
      VLOG(1) << "Couldn't extract size and stride from " << ToString(summand);
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
void SortByStride(std::vector<SizeAndStrideExpression>& sizes_and_strides,
                  bool reverse = false) {
  absl::c_sort(sizes_and_strides, [&](const SizeAndStrideExpression& sas1,
                                      const SizeAndStrideExpression& sas2) {
    int64_t stride1 = llvm::cast<AffineConstantExpr>(sas1.stride).getValue();
    int64_t stride2 = llvm::cast<AffineConstantExpr>(sas2.stride).getValue();
    if (reverse) {
      return stride1 > stride2;
    }
    return stride1 < stride2;
  });
}

// Returns the range size of the given size expression.
//
// `size` must be a constant or dimension expression.
std::optional<int64_t> TryGetSizeExpressionRangeSize(
    AffineExpr size, absl::Span<Interval const> dimension_intervals) {
  if (size.getKind() == AffineExprKind::Constant) {
    return llvm::cast<AffineConstantExpr>(size).getValue();
  }
  CHECK(size.getKind() == AffineExprKind::DimId);
  auto dim_position = llvm::dyn_cast<AffineDimExpr>(size).getPosition();
  const Interval& interval = dimension_intervals.at(dim_position);
  if (interval.lower != 0) {
    // TODO(bchetioui): I think we may need to handle this to have reshapes
    // working well with concatenations. Nevertheless, we can take a look
    // later.
    VLOG(1) << "Attempted to combine strides but got dimension "
            << ToString(size) << " with lower bound " << interval.lower
            << " != 0";
    return std::nullopt;
  }
  // We need to add 1 to the upper bound of the interval to describe the
  // number of elements being captured, since the interval bounds are
  // inclusive.
  return interval.upper + 1;
};

// Given a list of sizes and strides, combines the strides into a single
// expression if it is possible.
//
// The current implementation expects that each size captures a single dimension
// parameter or a constant (coming from a RangeVar).
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
              << ToString(size_and_stride.stride);
      return std::nullopt;
    }

    // We know the exact bounds of dimension parameters, since they correspond
    // to parameters of the initial indexing map. It follows that if a size
    // expression is exactly a dimension parameter, we know its exact bounds.
    //
    // If a size is not a constant and not exactly a dimension parameter, then
    // it is dubious whether we know the bounds---and may thus calculate wrong
    // strides.
    if (size_and_stride.size.getKind() != AffineExprKind::Constant &&
        size_and_stride.size.getKind() != AffineExprKind::DimId) {
      VLOG(1) << "Attempted to combine strides but got non-constant, "
                 "non-dimension size "
              << ToString(size_and_stride.size);
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
      std::optional<int64_t> previous_size_expression_range_size =
          TryGetSizeExpressionRangeSize(previous_size_and_stride.size,
                                        dimension_intervals);
      if (!previous_size_expression_range_size.has_value()) {
        return std::nullopt;
      }

      int64_t previous_stride =
          llvm::cast<AffineConstantExpr>(previous_size_and_stride.stride)
              .getValue();

      if (*previous_size_expression_range_size * previous_stride != stride) {
        VLOG(1) << "Attempted to combine strides but stride did not grow "
                << "exactly as expected: got "
                << *previous_size_expression_range_size << " * "
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
    std::optional<int64_t> size_expression_range_size =
        TryGetSizeExpressionRangeSize(size, dimension_intervals);
    if (!size_expression_range_size.has_value()) {
      return std::nullopt;
    }
    nested_if = IfNeqOne(size, stride, nested_if, *size_expression_range_size);
  }

  return nested_if;
}

// Given a set of size expressions assumed to be sorted in descending order of
// associated stride, returns a conjunction such that:
//   - the first `partial_dim_index` size expressions are constrained to be
//     equal to 1;
//   - the `partial_dim_index`-th size expression is unconstrained;
//   - the next `num_full_dims` size expressions are constrained to be equal to
//     their upper bound;
//   - the remaining size expressions are constrained to be equal to 1.
//
// See also the documentation of
// `ConstructConstraintExpressionForDestructuredSummation` for broader context.
std::optional<ConstraintExpression>
TryConstructSingleConjointConstraintForDestructuredSummation(
    absl::Span<SizeAndStrideExpression const> sizes_and_strides,
    absl::Span<Interval const> dimension_intervals, int64_t partial_dim_index,
    int64_t num_full_dims) {
  CHECK_LE(partial_dim_index + num_full_dims, sizes_and_strides.size());

  ConstraintExpression constraints = ConstraintExpression::GetAlwaysSatisfied();
  Interval one = Interval{/*lower=*/1, /*upper=*/1};
  int64_t running_size_index = 0;

  // Add leading ones.
  while (running_size_index < partial_dim_index) {
    constraints = constraints &&
                  Constraint{sizes_and_strides[running_size_index].size, one};
    ++running_size_index;
  }

  // Skip partial dimension, since "partial" basically means unconstrained.
  ++running_size_index;

  // Add full dimensions.
  while (running_size_index <= partial_dim_index + num_full_dims) {
    AffineExpr size_expr = sizes_and_strides[running_size_index].size;
    std::optional<int64_t> max_size =
        TryGetSizeExpressionRangeSize(size_expr, dimension_intervals);
    if (!max_size.has_value()) {
      return std::nullopt;
    }
    constraints =
        constraints && Constraint{size_expr, Interval{/*lower=*/*max_size,
                                                      /*upper=*/*max_size}};
    ++running_size_index;
  }

  // Add trailing ones.
  while (running_size_index < sizes_and_strides.size()) {
    constraints = constraints &&
                  Constraint{sizes_and_strides[running_size_index].size, one};
    ++running_size_index;
  }

  return constraints;
}

// Constructs constraints for the summation expression
//   expr = sum(map(lambda [size, stride]: stride * size, sizes_and_strides)).
//
// In order to assign a single stride for the summation expression, we need to
// ensure that the parameters (sizes) involved in the expression are such that
// the gap between them is always the same. Concretely, given a list of sizes
// [s0, s1, ..., s{n}] ordered in descending order of associated strides, we
// expect that each size s{k} is either:
//   a) 1 (and the corresponding stride is irrelevant);
//   b) fully captured---i.e. s{k} = upper_bound(s{k}). Assume s{k} is the
//      leftmost fully captured dimension. In that case,
//      for i in {0, ..., n-k-1}, s{k+i+1} is allowed to be fully captured if
//      s{k+i} is also fully captured.  Otherwise, s{k+i+1} = 1. The resulting
//      stride is the smallest stride associated with a fully captured
//      dimension, or the stride of s{k};
//   c) partially captured---i.e. 1 < s{k} < upper_bound(s{k}). In that case,
//      for i in {0, ..., k-1}, s{i} = 1. s{k+1} is allowed to be fully
//      captured (and thus the leftmost fully captured dimension), in which case
//      we do as in b). If s{k+1} is not fully captured, then
//      for i in {k+1, ..., n}, s{i} = 1, and the stride of the expression is
//      the stride associated with s{k}.
//
// As a regex-like summary, we expect the sizes to be as follows in row-major
// order (i.e. strictly decreasing order of strides):
//   (1*, partial_dim?, full_dims*, 1*).
//
// See also the documentation of `CombineStrides`.
ConstraintExpression ConstructConstraintExpressionForDestructuredSummation(
    std::vector<SizeAndStrideExpression> sizes_and_strides,
    absl::Span<Interval const> dimension_intervals) {
  SortByStride(sizes_and_strides, /*reverse=*/true);
  ConstraintExpression result = ConstraintExpression::GetUnsatisfiable();

  int64_t num_components = sizes_and_strides.size();
  for (int64_t partial_dim_index = 0; partial_dim_index < num_components;
       ++partial_dim_index) {
    for (int64_t num_full_dims = 0;
         num_full_dims < num_components - partial_dim_index; ++num_full_dims) {
      std::optional<ConstraintExpression> single_conjoint_constraint =
          TryConstructSingleConjointConstraintForDestructuredSummation(
              sizes_and_strides, dimension_intervals, partial_dim_index,
              num_full_dims);
      if (!single_conjoint_constraint.has_value()) {
        // Even if we fail to derive a single conjunction, we can still recover
        // if we are able to derive another one. The constraint system will
        // just end up being more restricted (since one of the branches of the
        // overall disjunction will disappear).
        continue;
      }
      result = result || *single_conjoint_constraint;
    }
  }

  return result;
}

// See documentation of `CombineSizes` and `CombineStrides` for an explanation
// of how sizes and strides are combined.
std::optional<SizeAndStrideExpression> CombineSizesAndStrides(
    std::vector<SizeAndStrideExpression> sizes_and_strides,
    absl::Span<Interval const> dimension_intervals) {
  CHECK(!sizes_and_strides.empty());

  if (VLOG_IS_ON(1)) {
    for (const SizeAndStrideExpression& size_and_stride : sizes_and_strides) {
      LOG(INFO) << "CombineSizesAndStrides:";
      LOG(INFO) << "size: " << ToString(size_and_stride.size)
                << " stride: " << ToString(size_and_stride.stride);
    }
  }

  ConstraintExpression constraints = ConstraintExpression::GetAlwaysSatisfied();
  for (SizeAndStrideExpression& size_and_stride : sizes_and_strides) {
    constraints = constraints && size_and_stride.constraints;
  }

  AffineExpr size = CombineSizes(sizes_and_strides);
  std::optional<AffineExpr> stride =
      CombineStrides(sizes_and_strides, dimension_intervals);
  if (!stride.has_value()) {
    return std::nullopt;
  }

  // Derive necessary constraints for the summation expression. These
  // constraints are explained in the documentation of
  // `ConstructConstraintExpressionForDestructuredSummation` and
  // `CombineStrides`.
  constraints =
      constraints && ConstructConstraintExpressionForDestructuredSummation(
                         std::move(sizes_and_strides), dimension_intervals);

  return SizeAndStrideExpression(size, *stride, std::move(constraints));
}

std::optional<SizeAndStrideExpression> ExtractSizeAndStride(
    AffineExpr strided_indexing, absl::Span<Interval const> dimension_intervals,
    absl::Span<Interval const> symbol_intervals) {
  MLIRContext* ctx = strided_indexing.getContext();

  switch (strided_indexing.getKind()) {
    case AffineExprKind::DimId:
      return SizeAndStrideExpression(/*size=*/strided_indexing, /*stride=*/1);
    case AffineExprKind::Mul: {
      const auto mul = llvm::cast<AffineBinaryOpExpr>(strided_indexing);
      AffineExpr lhs = mul.getLHS();
      std::optional<SizeAndStrideExpression> maybe_size_and_stride =
          ExtractSizeAndStride(lhs, dimension_intervals, symbol_intervals);
      if (!maybe_size_and_stride.has_value()) {
        return std::nullopt;
      }

      return SizeAndStrideExpression(
          /*size=*/maybe_size_and_stride->size,
          /*stride=*/maybe_size_and_stride->stride * mul.getRHS());
    }
    case AffineExprKind::Mod: {
      auto mod = llvm::cast<AffineBinaryOpExpr>(strided_indexing);
      return ExtractSizeAndStrideFromMod(mod.getLHS(), mod.getRHS());
    }
    case AffineExprKind::FloorDiv: {
      auto floor_div = llvm::cast<AffineBinaryOpExpr>(strided_indexing);
      return ExtractSizeAndStrideFromFloorDiv(floor_div.getLHS(),
                                              floor_div.getRHS());
    }
    case AffineExprKind::Constant:
      return SizeAndStrideExpression(/*size=*/getAffineConstantExpr(1, ctx),
                                     /*stride=*/0);
    case AffineExprKind::SymbolId: {
      auto symbol = llvm::cast<AffineSymbolExpr>(strided_indexing);
      const Interval& symbol_interval = symbol_intervals[symbol.getPosition()];
      if (symbol_interval.lower != 0) {
        return std::nullopt;
      }

      return SizeAndStrideExpression(
          /*size=*/getAffineConstantExpr(symbol_interval.upper + 1, ctx),
          /*stride=*/1);
    }
    case AffineExprKind::Add: {
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
    case AffineExprKind::CeilDiv:
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
  tmp_indexing_map.Simplify(IndexingMap::SimplifyPointDimensions::kPreserve);

  CHECK_EQ(tmp_indexing_map.GetAffineMap().getResults().size(), 1);
  return tmp_indexing_map.GetAffineMap().getResults().back();
}

// Tries to take the conjunction of `conjunction_1` and `conjunction_2`.
// Fails and returns `std::nullopt` if and only if the conjunction attempt
// results in an unsatisfiable constraint.
std::optional<ConjointConstraints> TryIntersectConjointConstraints(
    const ConjointConstraints& conjunction_1,
    const ConjointConstraints& conjunction_2) {
  if (conjunction_1.empty()) {
    return conjunction_2;
  }

  if (conjunction_2.empty()) {
    return conjunction_1;
  }

  ConjointConstraints result = conjunction_1;
  for (const auto& constraint : conjunction_2) {
    Constraint* result_it =
        llvm::find_if(result, [&](const Constraint& result_constraint) {
          return result_constraint.expr == constraint.expr;
        });
    const auto& [expr, interval] = constraint;
    if (result_it != result.end()) {
      auto& [result_expr, result_interval] = *result_it;
      result_interval = result_interval.Intersect(interval);
      if (!result_interval.IsFeasible()) {
        VLOG(1) << "Got two incompatible intervals for expression "
                << ToString(expr);
        return std::nullopt;
      }
    } else {
      result.push_back(Constraint{expr, interval});
    }
  }

  return result;
}

}  // anonymous namespace

ConstraintExpression operator&&(const ConstraintExpression& first,
                                const ConstraintExpression& second) {
  // When either one of the expressions is unsatisfiable, their conjunction is
  // necessarily unsatisfiable.
  if (!first.is_satisfiable() || !second.is_satisfiable()) {
    return ConstraintExpression::GetUnsatisfiable();
  }

  // Both first and second are satisfiable. Handle here explicitly the case
  // where one (or both) of the maps are trivially satisfied.
  if (first.IsAlwaysSatisfied()) {
    return second;
  }

  if (second.IsAlwaysSatisfied()) {
    return first;
  }

  // `IsAlwaysSatisfied()` is true if and only if the map holds literally no
  // useful information and is equivalent to a default-constructed
  // `ConstraintExpression`---one that is neither unsatisfiable, nor contains
  // any constraints. Therefore, we can assume below that both of the provided
  // `ConstraintExpression`s are satisfiable and each contain at least one
  // constraint.
  //
  // By distributivity, we have that:
  //     (conj0 || conj1 || ...) && (conj2 || conj3 || ...)
  //   = (conj0 && conj2 || conj0 && conj3 || ... ||
  //      conj1 && conj2 || conj1 && conj3 ...)
  // which allows us to construct the result by essentially taking the cartesian
  // product of the disjoint conjunctions of `first` with those of `second`.
  decltype(std::declval<ConstraintExpression>()
               .disjoint_conjoint_constraints_) conjunctions;
  for (const ConjointConstraints& conjunction_1 :
       first.disjoint_conjoint_constraints_) {
    for (const ConjointConstraints& conjunction_2 :
         second.disjoint_conjoint_constraints_) {
      std::optional<ConjointConstraints> maybe_conjunction =
          TryIntersectConjointConstraints(conjunction_1, conjunction_2);
      // We only add the resulting conjunction to the result
      // `ConstraintExpression` if it is satisfiable, since it is otherwise
      // redundant: (conj || false) == conj.
      if (maybe_conjunction.has_value()) {
        conjunctions.push_back(*maybe_conjunction);
      }
    }
  }

  // If all the resulting conjunctions are unsatisfiable, the result itself is
  // unsatisfiable: (false || false) == false.
  // In our case, this manifests as an empty list of constraints in the result.
  ConstraintExpression result(/*is_satisfiable=*/!conjunctions.empty());
  result.disjoint_conjoint_constraints_ = std::move(conjunctions);
  return result;
}

ConstraintExpression operator||(const ConstraintExpression& first,
                                const ConstraintExpression& second) {
  // If either one of the expressions is always satisfied, their disjunction is
  // always satisfied.
  if (first.IsAlwaysSatisfied() || second.IsAlwaysSatisfied()) {
    return ConstraintExpression::GetAlwaysSatisfied();
  }

  // When either one of the expressions is unsatisfiable, we can simply return
  // the other one.
  if (!first.is_satisfiable()) {
    return second;
  }
  if (!second.is_satisfiable()) {
    return first;
  }

  ConstraintExpression result = first;
  absl::c_copy(second.disjoint_conjoint_constraints_,
               std::back_inserter(result.disjoint_conjoint_constraints_));
  return result;
}

bool ConstraintExpression::IsSatisfiedBy(
    absl::Span<const int64_t> dim_values) const {
  if (disjoint_conjoint_constraints_.empty()) {
    return is_satisfiable_;
  }

  return absl::c_any_of(
      disjoint_conjoint_constraints_, [&](const auto& conjunction) {
        return absl::c_all_of(conjunction, [&](const Constraint& constraint) {
          int64_t value = EvaluateAffineExpr(constraint.expr, dim_values);
          return constraint.interval.Contains(value);
        });
      });
}

std::string ConstraintExpression::ToString() const {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}

void ConstraintExpression::Print(std::ostream& out) const {
  if (IsAlwaysSatisfied()) {
    out << "always satisfied\n";
    return;
  }
  if (!is_satisfiable()) {
    out << "unsatisfiable\n";
    return;
  }

  // Accumulate constraints in a vector in order to put them in lexicographic
  // order and to get deterministic output.
  std::vector<std::string> conjunction_strings;
  conjunction_strings.reserve(disjoint_conjoint_constraints_.size());
  for (const auto& disjunction : disjoint_conjoint_constraints_) {
    std::vector<std::string> constraint_strings;
    constraint_strings.reserve(disjunction.size());
    for (const auto& [expr, interval] : disjunction) {
      constraint_strings.push_back(
          absl::StrCat(xla::ToString(expr), " in ", interval.ToString()));
    }
    std::sort(constraint_strings.begin(), constraint_strings.end());
    conjunction_strings.push_back(absl::StrJoin(constraint_strings, " && "));
  }
  std::sort(conjunction_strings.begin(), conjunction_strings.end());
  out << absl::StrJoin(conjunction_strings, " || ") << "\n";
}

namespace {

bool IsConstraintAlwaysSatisfied(AffineExpr expr, Interval interval) {
  if (AffineConstantExpr constant = mlir::dyn_cast<AffineConstantExpr>(expr)) {
    return interval.Contains(constant.getValue());
  }
  return false;
}

bool IsConstraintUnsatisfiable(AffineExpr expr, Interval interval) {
  if (!interval.IsFeasible()) {
    return true;
  }
  if (AffineConstantExpr constant = mlir::dyn_cast<AffineConstantExpr>(expr)) {
    return !interval.Contains(constant.getValue());
  }
  return false;
}

struct Unsatisfiable {};
struct AlwaysSatisfied {};

std::variant<Unsatisfiable, AlwaysSatisfied, ConjointConstraints>
SimplifyConjointConstraints(const ConjointConstraints& conjunction) {
  ConjointConstraints result;
  for (const auto& [expr, interval] : conjunction) {
    if (IsConstraintAlwaysSatisfied(expr, interval)) {
      continue;
    }
    if (IsConstraintUnsatisfiable(expr, interval)) {
      return Unsatisfiable{};
    }
    result.push_back(Constraint{expr, interval});
  }
  if (result.empty()) {
    return AlwaysSatisfied{};
  }

  // A comparator to canonicalize the order of constraints, so we can easily
  // check if two ConjointConstraints are equal. The order is arbitrary (doesn't
  // depend on the structure of the constraints) and can change between runs,
  // but is stable during a single execution. The printed version of the
  // constraints relies on sorting strings, so string representation will be
  // always the same.
  auto comp = [](const Constraint& a, const Constraint& b) {
    if (a.expr != b.expr) {
      // AffineExpr are deduplicated and stored as immutable objects in
      // MLIRContext. Comparing pointers gives us a fast and easy way to get
      // stable ordering.
      CHECK_EQ(a.expr.getContext(), b.expr.getContext())
          << "AffineExpr should be from the same MLIRContext.";
      return a.expr.getImpl() < b.expr.getImpl();
    }

    // Default comparison for intervals will return nullopt if intervals are
    // overlapping. Here we do strict ordering by comparing lower bounds first
    // and then upper bounds.
    return std::make_pair(a.interval.lower, a.interval.upper) <
           std::make_pair(b.interval.lower, b.interval.upper);
  };

  // Canonicalize constraints order and remove duplicates.
  std::sort(result.begin(), result.end(), comp);
  result.erase(std::unique(result.begin(), result.end()), result.end());

  return result;
}

}  // namespace

void ConstraintExpression::Simplify() {
  if (disjoint_conjoint_constraints_.empty()) {
    return;  // unsatisfiable or always satisfied.
  }

  // Find and remove redundant constraints.
  absl::flat_hash_set<ConjointConstraints> unique_conjunctions;
  for (const auto& conjunction : disjoint_conjoint_constraints_) {
    auto simplified_conjunction = SimplifyConjointConstraints(conjunction);
    if (std::holds_alternative<Unsatisfiable>(simplified_conjunction)) {
      continue;
    }
    if (std::holds_alternative<AlwaysSatisfied>(simplified_conjunction)) {
      return disjoint_conjoint_constraints_.clear();
    }
    unique_conjunctions.insert(
        std::get<ConjointConstraints>(simplified_conjunction));
  }
  disjoint_conjoint_constraints_.assign(unique_conjunctions.begin(),
                                        unique_conjunctions.end());
  // If all the conjunctions are unsatisfiable, the result is unsatisfiable.
  is_satisfiable_ = !disjoint_conjoint_constraints_.empty();
}

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

  // Eliminate negative strides and recalculate offsets.
  // TODO(b/340555497): handle normalization of more complex expressions.
  for (auto [offset, size, stride] :
       llvm::zip(offset_expressions, size_expressions, stride_expressions)) {
    auto constant = llvm::dyn_cast<AffineConstantExpr>(stride);
    if (constant && constant.getValue() < 0) {
      offset = offset + size * stride - stride;
      stride = -stride;
    } else if (!constant) {
      VLOG(1) << "Unexpected non-constant stride expression: "
              << xla::ToString(stride);
    }
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

}  // namespace gpu
}  // namespace xla
