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

#include "xla/service/gpu/model/indexing_map.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "xla/service/gpu/model/affine_map_printer.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {
namespace gpu {
namespace {

using mlir::AffineBinaryOpExpr;
using mlir::AffineConstantExpr;
using mlir::AffineDimExpr;
using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::AffineMap;
using mlir::AffineSymbolExpr;
using mlir::getAffineBinaryOpExpr;
using mlir::getAffineConstantExpr;
using mlir::MLIRContext;

int64_t FloorDiv(int64_t dividend, int64_t divisor) {
  return dividend / divisor -
         (((dividend >= 0) != (divisor >= 0) && dividend % divisor) ? 1 : 0);
}

int64_t CeilDiv(int64_t dividend, int64_t divisor) {
  return dividend / divisor +
         (((dividend >= 0) == (divisor >= 0) && dividend % divisor) ? 1 : 0);
}

class AffineExprSimplifier {
 public:
  explicit AffineExprSimplifier(RangeEvaluator* range_evaluator)
      : range_evaluator_(range_evaluator) {}

  // Simplifies the map as much as possible.
  mlir::AffineMap Simplify(mlir::AffineMap affine_map);

  mlir::AffineExpr Simplify(mlir::AffineExpr expr);

 private:
  std::optional<int64_t> GetConstantRhsMultiplier(mlir::AffineExpr expr);

  // Simplifier for mod.
  // - Rewrites (a * 100 + ...) % 100 to (...) % 100
  // - Rewrites a % b to a if a is known to be less than b.
  mlir::AffineExpr RewriteMod(mlir::AffineBinaryOpExpr mod);

  // Simplifier for floordiv.
  // - Rewrites (a * 100 + ...) / 100 to a + (...) / 100
  // - Rewrites a / 100 to 0 when a is known to be less than 100.
  mlir::AffineExpr RewriteFloorDiv(mlir::AffineBinaryOpExpr div);

  mlir::AffineExpr RewriteSumIf(
      mlir::AffineExpr expr, const std::function<bool(mlir::AffineExpr)>& pred);

  // Attempts to simplify the expression, but doesn't attempt to simplify the
  // result further.
  mlir::AffineExpr SimplifyOnce(mlir::AffineExpr expr);

  RangeEvaluator* range_evaluator_;
};

AffineExpr AffineExprSimplifier::RewriteMod(AffineBinaryOpExpr mod) {
  auto lhs_simplified = SimplifyOnce(mod.getLHS());

  auto lhs = range_evaluator_->ComputeExpressionRange(lhs_simplified);
  auto rhs = range_evaluator_->ComputeExpressionRange(mod.getRHS());

  // a % b where b is always larger than a?
  if (0 <= lhs.lower_bound && lhs.upper_bound < rhs.upper_bound) {
    return lhs_simplified;
  }

  // The logic below assumes we have a constant RHS.
  if (!rhs.IsPoint()) {
    return mod;
  }
  int64_t m = rhs.lower_bound;

  auto new_lhs = RewriteSumIf(lhs_simplified, [&](AffineExpr expr) {
    if (expr.getKind() != AffineExprKind::Mul) {
      return true;
    }

    auto mul_rhs = range_evaluator_->ComputeExpressionRange(
        mlir::cast<AffineBinaryOpExpr>(expr).getRHS());
    bool remove = mul_rhs.IsPoint() && (mul_rhs.lower_bound % m) == 0;
    return !remove;  // We keep it if we don't remove it!
  });

  // If we weren't able to remove or simplify anything, return the original
  // expression.
  if (new_lhs == mod.getLHS()) {
    return mod;
  }
  // If we removed everything, return 0.
  if (!new_lhs) {
    return getAffineConstantExpr(0, range_evaluator_->GetMLIRContext());
  }
  // Otherwise, return new_sum % m.
  return new_lhs % mod.getRHS();
}

AffineExpr AffineExprSimplifier::RewriteFloorDiv(AffineBinaryOpExpr div) {
  auto mlir_context = range_evaluator_->GetMLIRContext();
  auto lhs_simplified = SimplifyOnce(div.getLHS());
  auto lhs = range_evaluator_->ComputeExpressionRange(lhs_simplified);
  auto rhs = range_evaluator_->ComputeExpressionRange(div.getRHS());

  if (0 <= lhs.lower_bound && lhs.upper_bound < rhs.lower_bound) {
    return getAffineConstantExpr(0, mlir_context);
  }

  // The logic below assumes we have a constant RHS.
  if (!rhs.IsPoint()) {
    return div;
  }
  int64_t d = rhs.lower_bound;

  // If the dividend's range has a single element, return its value.
  int64_t a = FloorDiv(lhs.lower_bound, d);
  int64_t b = FloorDiv(lhs.upper_bound, d);
  if (a == b) {
    return getAffineConstantExpr(a, mlir_context);
  }

  AffineExpr extracted = getAffineConstantExpr(0, mlir_context);
  auto new_dividend = RewriteSumIf(lhs_simplified, [&](AffineExpr expr) {
    if (auto multiplier = GetConstantRhsMultiplier(expr)) {
      // (x * 7 + ...) / 3 -> can't extract. We could extract x * 2 and keep
      // one x, but we currently have no reason to do that.
      if (*multiplier % d != 0) {
        return true;
      }
      int64_t factor = *multiplier / d;
      extracted =
          extracted + mlir::cast<AffineBinaryOpExpr>(expr).getLHS() * factor;
      // Remove from dividend.
      return false;
    }

    // Not a constant multiplier, keep in dividend.
    return true;
  });

  // If we removed everything, skip the div.
  if (!new_dividend) {
    return extracted;
  }
  // If we removed nothing, return the original division.
  if (extracted == getAffineConstantExpr(0, mlir_context) &&
      new_dividend == div.getLHS()) {
    return div;
  }

  return extracted + new_dividend.floorDiv(div.getRHS());
}

std::optional<int64_t> AffineExprSimplifier::GetConstantRhsMultiplier(
    AffineExpr expr) {
  if (expr.getKind() != AffineExprKind::Mul) {
    return std::nullopt;
  }
  auto bound = range_evaluator_->ComputeExpressionRange(
      mlir::cast<AffineBinaryOpExpr>(expr).getRHS());
  if (!bound.IsPoint()) {
    return std::nullopt;
  }
  return bound.lower_bound;
}

AffineExpr AffineExprSimplifier::RewriteSumIf(
    AffineExpr expr, const std::function<bool(AffineExpr)>& pred) {
  if (expr.getKind() == AffineExprKind::Add) {
    auto add = mlir::dyn_cast<AffineBinaryOpExpr>(expr);
    auto lhs = RewriteSumIf(add.getLHS(), pred);
    auto rhs = RewriteSumIf(add.getRHS(), pred);
    if (lhs == add.getLHS() && rhs == add.getRHS()) {
      return add;
    }
    if (lhs && rhs) {
      return lhs + rhs;
    }
    return lhs ? lhs : (rhs ? rhs : nullptr);
  }
  return pred(expr) ? expr : nullptr;
}

AffineExpr AffineExprSimplifier::SimplifyOnce(AffineExpr expr) {
  switch (expr.getKind()) {
    case AffineExprKind::Mul:
    case AffineExprKind::Add: {
      auto binop = mlir::cast<AffineBinaryOpExpr>(expr);
      auto lhs = SimplifyOnce(binop.getLHS());
      auto rhs = SimplifyOnce(binop.getRHS());
      if (lhs == binop.getLHS() && rhs == binop.getRHS()) {
        return expr;
      }
      return getAffineBinaryOpExpr(expr.getKind(), lhs, rhs);
    }
    case AffineExprKind::Mod:
      return RewriteMod(mlir::cast<AffineBinaryOpExpr>(expr));
    case AffineExprKind::FloorDiv:
      return RewriteFloorDiv(mlir::cast<AffineBinaryOpExpr>(expr));
    case AffineExprKind::DimId:
    case AffineExprKind::SymbolId: {
      auto bounds = range_evaluator_->ComputeExpressionRange(expr);
      if (bounds.IsPoint()) {
        return getAffineConstantExpr(bounds.lower_bound,
                                     range_evaluator_->GetMLIRContext());
      }
      return expr;
    }

    default:
      return expr;
  }
}

AffineExpr AffineExprSimplifier::Simplify(AffineExpr expr) {
  while (true) {
    auto simplified = SimplifyOnce(expr);
    if (simplified == expr) {
      return expr;
    }
    expr = simplified;
  }
}

AffineMap AffineExprSimplifier::Simplify(AffineMap affine_map) {
  mlir::SmallVector<AffineExpr, 4> results;
  results.reserve(affine_map.getNumResults());
  bool nothing_changed = true;
  for (AffineExpr expr : affine_map.getResults()) {
    AffineExpr simplified = Simplify(expr);
    nothing_changed &= simplified == expr;
    results.push_back(simplified);
  }
  if (nothing_changed) {
    return affine_map;
  }
  return mlir::simplifyAffineMap(
      AffineMap::get(affine_map.getNumDims(), affine_map.getNumSymbols(),
                     results, affine_map.getContext()));
}

// Computes intersection of two ranges.
Range Intersect(const Range& lhs, const Range& rhs) {
  return Range{std::max(lhs.lower_bound, rhs.lower_bound),
               std::min(lhs.upper_bound, rhs.upper_bound)};
}

// Simplifies a constraint range, i.e. a constraint d0 + x in [lb, ub] will
// become d0 in [lb - x, ub - x]. Also supports *, floorDiv.
bool SimplifyConstraintRangeOnce(AffineExpr* expr, Range* range) {
  switch (expr->getKind()) {
    case AffineExprKind::DimId:
    case AffineExprKind::SymbolId:
      // do the trick with constant
    case AffineExprKind::Constant: {
      return false;
    }
    default: {
      auto binary_op = mlir::cast<AffineBinaryOpExpr>(*expr);
      CHECK(binary_op);
      auto lhs = binary_op.getLHS();
      auto rhs = binary_op.getRHS();
      auto constant = mlir::dyn_cast<AffineConstantExpr>(rhs);
      if (!constant) {
        return false;
      }
      switch (expr->getKind()) {
        case AffineExprKind::Add: {
          int64_t shift = constant.getValue();
          range->lower_bound -= shift;
          range->upper_bound -= shift;
          *expr = lhs;
          return true;
        }
        case AffineExprKind::Mul: {
          int64_t factor = constant.getValue();
          if (factor < 0) {
            factor *= -1;
            range->lower_bound *= -1;
            range->upper_bound *= -1;
            std::swap(range->lower_bound, range->upper_bound);
          }
          range->lower_bound = CeilDiv(range->lower_bound, factor);
          range->upper_bound = FloorDiv(range->upper_bound, factor);
          *expr = lhs;
          return true;
        }
        case AffineExprKind::FloorDiv: {
          int64_t divisor = constant.getValue();
          if (divisor < 0) {
            divisor *= -1;
            range->lower_bound *= -1;
            range->upper_bound *= -1;
            std::swap(range->lower_bound, range->upper_bound);
          }
          range->lower_bound *= divisor;
          range->upper_bound = (range->upper_bound + 1) * divisor - 1;
          *expr = lhs;
          return true;
        }
        default: {
          return false;
        }
      }
    }
  }
}

// Repeatedly simplifies the range of the constraint.
bool SimplifyConstraintRange(AffineExpr* expr, Range* range) {
  bool is_simplified = false;
  while (SimplifyConstraintRangeOnce(expr, range)) {
    is_simplified = true;
  }
  return is_simplified;
}

}  // namespace

std::string Range::ToString() const {
  std::stringstream ss;
  Print(ss);
  return ss.str();
}

void Range::Print(std::ostream& out) const {
  out << '[' << lower_bound << ", " << upper_bound << "]";
}

std::ostream& operator<<(std::ostream& out, const Range& range) {
  range.Print(out);
  return out;
}

bool operator==(const Range& lhs, const Range& rhs) {
  return lhs.lower_bound == rhs.lower_bound &&
         lhs.upper_bound == rhs.upper_bound;
}

IndexingMap IndexingMap::FromTensorSizes(
    AffineMap affine_map, absl::Span<const int64_t> dim_upper_bounds,
    absl::Span<const int64_t> symbol_upper_bounds) {
  IndexingMap indexing_map;
  indexing_map.affine_map_ = affine_map;
  indexing_map.dim_ranges_.reserve(dim_upper_bounds.size());
  for (int64_t ub : dim_upper_bounds) {
    CHECK_GT(ub, 0);
    indexing_map.dim_ranges_.push_back(Range{0, ub - 1});
  }
  indexing_map.symbol_ranges_.reserve(symbol_upper_bounds.size());
  for (int64_t ub : symbol_upper_bounds) {
    CHECK_GT(ub, 0);
    indexing_map.symbol_ranges_.push_back(Range{0, ub - 1});
  }
  return indexing_map;
}

void IndexingMap::AddConstraint(mlir::AffineExpr expr, Range range) {
  if (auto dim_expr = mlir::dyn_cast<AffineDimExpr>(expr)) {
    Range& current_range = dim_ranges_[dim_expr.getPosition()];
    current_range = Intersect(current_range, range);
    return;
  }
  if (auto symbol_expr = mlir::dyn_cast<AffineSymbolExpr>(expr)) {
    Range& current_range = symbol_ranges_[symbol_expr.getPosition()];
    current_range = Intersect(current_range, range);
    return;
  }
  // TODO(b/322131639): Add a proper Constraints simplifier that will apply
  // simplification rules until it converges. For example, it should have a rule
  // for `symbol_or_dim floorDiv divisor`.
  if (SimplifyConstraintRange(&expr, &range)) {
    AddConstraint(expr, range);
    return;
  }
  auto [it, inserted] = expr_ranges_.insert({expr, range});
  if (!inserted) {
    it->second = Intersect(it->second, range);
  }
}

bool IndexingMap::IsKnownEmpty() const {
  auto is_infeasible = [](const Range& range) {
    return range.lower_bound > range.upper_bound;
  };
  return llvm::any_of(dim_ranges_, is_infeasible) ||
         llvm::any_of(symbol_ranges_, is_infeasible) ||
         llvm::any_of(expr_ranges_,
                      [&](const std::pair<AffineExpr, Range>& item) {
                        return is_infeasible(item.second);
                      });
}

RangeEvaluator::RangeEvaluator(absl::Span<const Range> dim_ranges,
                               absl::Span<const Range> symbol_ranges,
                               MLIRContext* mlir_context)
    : mlir_context_(mlir_context) {
  for (const auto& [index, range] : llvm::enumerate(dim_ranges)) {
    expression_ranges_cache_[getAffineDimExpr(index, mlir_context_)] = range;
  }
  for (const auto& [index, range] : llvm::enumerate(symbol_ranges)) {
    expression_ranges_cache_[getAffineSymbolExpr(index, mlir_context_)] = range;
  }
}

bool RangeEvaluator::IsAlwaysPositiveOrZero(mlir::AffineExpr expr) {
  return ComputeExpressionRange(expr).lower_bound >= 0;
}

bool RangeEvaluator::IsAlwaysNegativeOrZero(mlir::AffineExpr expr) {
  return ComputeExpressionRange(expr).upper_bound <= 0;
}

Range RangeEvaluator::ComputeExpressionRange(AffineExpr expr) {
  switch (expr.getKind()) {
    case AffineExprKind::Constant: {
      int64_t value = mlir::cast<AffineConstantExpr>(expr).getValue();
      return Range{value, value};
    }
    case AffineExprKind::DimId: {
      return expression_ranges_cache_[expr];
    }
    case AffineExprKind::SymbolId: {
      return expression_ranges_cache_[expr];
    }
    default:
      auto bound = expression_ranges_cache_.find(expr);
      if (bound != expression_ranges_cache_.end()) {
        return bound->second;
      }
      auto binary_op = mlir::dyn_cast<AffineBinaryOpExpr>(expr);
      CHECK(binary_op);
      auto lhs = ComputeExpressionRange(binary_op.getLHS());
      auto rhs = ComputeExpressionRange(binary_op.getRHS());

      auto& result = expression_ranges_cache_[expr];
      switch (expr.getKind()) {
        case AffineExprKind::Add:
          return result = {lhs.lower_bound + rhs.lower_bound,
                           lhs.upper_bound + rhs.upper_bound};
        case AffineExprKind::Mul: {
          int64_t a = lhs.lower_bound * rhs.lower_bound;
          int64_t b = lhs.upper_bound * rhs.upper_bound;
          return result = {std::min(a, b), std::max(a, b)};
        }
        case AffineExprKind::Mod: {
          CHECK(rhs.IsPoint()) << "RHS of mod must be a constant";
          int64_t m = rhs.lower_bound;
          if (0 <= lhs.lower_bound && lhs.upper_bound < m) {
            return result = lhs;
          }
          return result = {0, m - 1};
        }
        case AffineExprKind::FloorDiv: {
          CHECK(rhs.IsPoint()) << "RHS of floor_div must be a constant";
          int64_t d = rhs.lower_bound;
          int64_t a = FloorDiv(lhs.lower_bound, d);
          int64_t b = FloorDiv(lhs.upper_bound, d);
          return result = {std::min(a, b), std::max(a, b)};
        }
        default:
          // We don't use ceildiv, so we don't support it.
          LOG(FATAL) << "Unsupported expression";
      }
  }
}

std::string IndexingMap::ToString(const AffineMapPrinter& printer) const {
  std::stringstream ss;
  Print(ss, printer);
  return ss.str();
}

void IndexingMap::Print(std::ostream& out,
                        const AffineMapPrinter& printer) const {
  printer.Print(out, affine_map_);
  out << "\ndomain:\n";
  for (const auto& [index, range] : llvm::enumerate(dim_ranges_)) {
    out << printer.GetDimensionName(static_cast<int64_t>(index)) << " in ";
    range.Print(out);
    out << '\n';
  }
  for (const auto& [index, range] : llvm::enumerate(symbol_ranges_)) {
    out << printer.GetSymbolName(static_cast<int64_t>(index)) << " in ";
    range.Print(out);
    out << '\n';
  }
  std::vector<std::string> expr_range_strings;
  expr_range_strings.reserve(expr_ranges_.size());
  for (const auto& [expr, range] : expr_ranges_) {
    std::stringstream ss;
    printer.Print(ss, expr);
    ss << " in ";
    range.Print(ss);
    expr_range_strings.push_back(ss.str());
  }
  std::sort(expr_range_strings.begin(), expr_range_strings.end());
  for (const auto& expr_range_string : expr_range_strings) {
    out << expr_range_string << '\n';
  }
}

std::ostream& operator<<(std::ostream& out, const IndexingMap& indexing_map) {
  AffineMapPrinter printer;
  indexing_map.Print(out, printer);
  return out;
}

bool operator==(const IndexingMap& lhs, const IndexingMap& rhs) {
  return lhs.GetAffineMap() == rhs.GetAffineMap() &&
         lhs.GetDimensionRanges() == rhs.GetDimensionRanges() &&
         lhs.GetSymbolRanges() == rhs.GetSymbolRanges();
}

bool IndexingMap::Simplify() {
  RangeEvaluator range_evaluator(dim_ranges_, symbol_ranges_, GetMLIRContext());
  AffineMap simplified_affine_map =
      AffineExprSimplifier(&range_evaluator).Simplify(affine_map_);
  if (simplified_affine_map == affine_map_) {
    return false;
  }
  affine_map_ = simplified_affine_map;
  return true;
}

std::optional<IndexingMap> ComposeIndexingMaps(
    const std::optional<IndexingMap>& producer_map,
    const std::optional<IndexingMap>& consumer_map) {
  if (!producer_map.has_value() || !consumer_map.has_value()) {
    return std::nullopt;
  }
  // AffineMap::compose(some_affine_map) actually computes some_affine_map ∘
  // this.
  AffineMap composed_map = mlir::simplifyAffineMap(
      producer_map->GetAffineMap().compose(consumer_map->GetAffineMap()));

  // After the composition some of the symbols might become unused, e.g. when a
  // dimension was added by broadcasting as then reduced. We should remove these
  // dimensions from the composed affine map and also from the resulting
  // `domain.symbol_ranges_`.
  //
  // For example, if there is a reduction(broadcast):
  //
  //   param = f32[15] parameter(0)
  //   bcast = f32[15, 20] broadcast(p0), dimensions={0}
  //   reduce = f32[15, 20] reduce(bcast, init) dimensions={1}
  //
  // then `reduce` has (d0)[s0] -> (d0, s0) with s0 in [0, 20).
  // and  `bcast` has (d0, d1) -> (d0) indexing map.
  //
  // The composition of there two maps yields (d0)[s0] -> (d0),
  // although `s0` is not used in the mapping. In order to remove such symbols,
  // we get the indices of unused symbols and remove them from the composed
  // affine map and the `domain.symbol_ranges_`.
  auto unused_symbols_bit_vector =
      mlir::getUnusedSymbolsBitVector({composed_map});
  composed_map = mlir::compressSymbols(composed_map, unused_symbols_bit_vector);

  // The symbols in the composed map, i.e. combined
  // producer_map.compose(consumer_map) are packed as [symbols(producer_map) |
  // symbols(consumer_map)]. In that order we are adding the symbol ranges while
  // skipping the symbols that are unused.
  std::vector<Range> combined_symbol_ranges;
  combined_symbol_ranges.reserve(producer_map->GetSymbolCount() +
                                 consumer_map->GetSymbolCount());
  int64_t symbol_id = 0;
  for (const Range& symbol_range : llvm::concat<const Range>(
           producer_map->GetSymbolRanges(), consumer_map->GetSymbolRanges())) {
    if (unused_symbols_bit_vector[symbol_id++]) {
      continue;
    }
    combined_symbol_ranges.push_back(symbol_range);
  }

  IndexingMap composed_indexing_map(composed_map,
                                    consumer_map->GetDimensionRanges(),
                                    std::move(combined_symbol_ranges));
  composed_indexing_map.Simplify();

  RangeEvaluator consumer_range_evaluator(consumer_map->GetDimensionRanges(),
                                          consumer_map->GetSymbolRanges(),
                                          consumer_map->GetMLIRContext());
  // Add constraints for consumer's codomain w.r.t. producer's domain.
  for (auto [index, expr] :
       llvm::enumerate(consumer_map->GetAffineMap().getResults())) {
    Range consumer_result_range =
        consumer_range_evaluator.ComputeExpressionRange(expr);
    Range producer_dim_range =
        producer_map->GetDimensionRange(static_cast<int64_t>(index));
    // If the constraint is always satisfied, we skip it.
    if (consumer_result_range.upper_bound <= producer_dim_range.upper_bound &&
        consumer_result_range.lower_bound >= producer_dim_range.lower_bound) {
      continue;
    }
    composed_indexing_map.AddConstraint(expr, producer_dim_range);
  }
  return composed_indexing_map;
}

}  // namespace gpu
}  // namespace xla
