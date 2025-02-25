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

#include "xla/service/gpu/model/size_and_stride_expression.h"

#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/service/gpu/model/constraint_expression.h"

namespace xla {
namespace gpu {
namespace {

using ::llvm::SmallVector;
using ::mlir::AffineBinaryOpExpr;
using ::mlir::AffineConstantExpr;
using ::mlir::AffineDimExpr;
using ::mlir::AffineExpr;
using ::mlir::AffineExprKind;
using ::mlir::AffineSymbolExpr;
using ::mlir::getAffineConstantExpr;
using ::mlir::MLIRContext;
using Constraint = ConstraintExpression::Constraint;
using ConjointConstraints = llvm::SmallVector<Constraint, 2>;

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
    int64_t stride1 =
        std::abs(llvm::cast<AffineConstantExpr>(sas1.stride).getValue());
    int64_t stride2 =
        std::abs(llvm::cast<AffineConstantExpr>(sas2.stride).getValue());
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

      if (*previous_size_expression_range_size * std::abs(previous_stride) !=
          std::abs(stride)) {
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

}  // anonymous namespace

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

}  // namespace gpu
}  // namespace xla
