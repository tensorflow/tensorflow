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

int64_t FloorDiv(int64_t dividend, int64_t divisor) {
  return dividend / divisor -
         (((dividend >= 0) != (divisor >= 0) && dividend % divisor) ? 1 : 0);
}

int64_t CeilDiv(int64_t dividend, int64_t divisor) {
  return dividend / divisor +
         (((dividend >= 0) == (divisor >= 0) && dividend % divisor) ? 1 : 0);
}

// Computes intersection of two ranges.
Range Intersect(const Range& lhs, const Range& rhs) {
  return Range{std::max(lhs.lower_bound, rhs.lower_bound),
               std::min(lhs.upper_bound, rhs.upper_bound)};
}

// Attempts to parse an expression dim_or_symbol * factor + shift.
bool ParseLinearFunction(AffineExpr expr, AffineExpr* symbol_or_dim,
                         int64_t* factor, int64_t* shift) {
  AffineExpr residual = expr;
  *shift = 0;
  *factor = 1;
  if (auto binop = mlir::dyn_cast<AffineBinaryOpExpr>(residual)) {
    if (binop.getKind() == AffineExprKind::Add) {
      auto constant = mlir::dyn_cast<AffineConstantExpr>(binop.getRHS());
      if (!constant) {
        return false;
      }
      *shift = constant.getValue();
      residual = binop.getLHS();
    }
  }
  if (auto binop = mlir::dyn_cast<AffineBinaryOpExpr>(residual)) {
    if (binop.getKind() == AffineExprKind::Mul) {
      auto constant = mlir::dyn_cast<AffineConstantExpr>(binop.getRHS());
      if (!constant) {
        return false;
      }
      *factor = constant.getValue();
      residual = binop.getLHS();
    }
  }
  if (residual.getKind() == AffineExprKind::DimId ||
      residual.getKind() == AffineExprKind::SymbolId) {
    *symbol_or_dim = residual;
    return true;
  }
  return false;
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

Domain Domain::FromUpperBounds(absl::Span<const int64_t> dim_upper_bounds,
                               absl::Span<const int64_t> symbol_upper_bounds) {
  Domain domain;
  domain.dim_ranges_.reserve(dim_upper_bounds.size());
  for (int64_t ub : dim_upper_bounds) {
    CHECK_GT(ub, 0);
    domain.dim_ranges_.push_back(Range{0, ub - 1});
  }
  domain.symbol_ranges_.reserve(symbol_upper_bounds.size());
  for (int64_t ub : symbol_upper_bounds) {
    CHECK_GT(ub, 0);
    domain.symbol_ranges_.push_back(Range{0, ub - 1});
  }
  return domain;
}

void Domain::AddConstraint(mlir::AffineExpr expr, const Range& range) {
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

  // Try to parse a linear function of type symbol_or_dim * factor + shift.
  AffineExpr symbol_or_dim;
  int64_t factor, shift;
  if (ParseLinearFunction(expr, &symbol_or_dim, &factor, &shift)) {
    Range new_range = factor > 0
                          ? Range{CeilDiv(range.lower_bound - shift, factor),
                                  FloorDiv(range.upper_bound - shift, factor)}
                          : Range{CeilDiv(range.upper_bound - shift, factor),
                                  FloorDiv(range.lower_bound - shift, factor)};
    AddConstraint(symbol_or_dim, new_range);
    return;
  }
  auto [it, inserted] = expr_ranges_.insert({expr, range});
  if (!inserted) {
    it->second = Intersect(it->second, range);
  }
}

bool Domain::IsKnownEmpty() const {
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

std::string Domain::ToString(const AffineMapPrinter& printer) const {
  std::stringstream ss;
  Print(ss, printer);
  return ss.str();
}

void Domain::Print(std::ostream& out, const AffineMapPrinter& printer) const {
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
    ss << '\n';
    expr_range_strings.push_back(ss.str());
  }
  std::sort(expr_range_strings.begin(), expr_range_strings.end());
  for (const auto& expr_range_string : expr_range_strings) {
    out << expr_range_string;
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
      return domain_->GetDimensionRange(
          mlir::cast<AffineDimExpr>(expr).getPosition());
    }
    case AffineExprKind::SymbolId: {
      return domain_->GetSymbolRange(
          mlir::cast<AffineSymbolExpr>(expr).getPosition());
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

std::ostream& operator<<(std::ostream& out, const Domain& domain) {
  AffineMapPrinter printer;
  domain.Print(out, printer);
  return out;
}

bool operator==(const Domain& lhs, const Domain& rhs) {
  return lhs.GetDimensionRanges() == rhs.GetDimensionRanges() &&
         lhs.GetSymbolRanges() == rhs.GetSymbolRanges();
}

std::string IndexingMap::ToString(const AffineMapPrinter& printer) const {
  std::stringstream ss;
  Print(ss, printer);
  return ss.str();
}

void IndexingMap::Print(std::ostream& out,
                        const AffineMapPrinter& printer) const {
  printer.Print(out, affine_map);
  out << "\ndomain:\n";
  domain.Print(out, printer);
  out << "\n";
}

std::ostream& operator<<(std::ostream& out, const IndexingMap& indexing_map) {
  AffineMapPrinter printer;
  indexing_map.Print(out, printer);
  return out;
}

bool operator==(const IndexingMap& lhs, const IndexingMap& rhs) {
  return lhs.affine_map == rhs.affine_map && lhs.domain == rhs.domain;
}

bool IndexingMap::Simplify() {
  RangeEvaluator range_evaluator(&domain);
  AffineMap simplified_affine_map =
      IndexingMapSimplifier(&range_evaluator, affine_map.getContext())
          .Simplify(affine_map);
  if (simplified_affine_map == affine_map) {
    return false;
  }
  affine_map = simplified_affine_map;
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
      producer_map->affine_map.compose(consumer_map->affine_map));

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
  combined_symbol_ranges.reserve(producer_map->domain.GetSymbolCount() +
                                 consumer_map->domain.GetSymbolCount());
  int64_t symbol_id = 0;
  for (const Range& symbol_range :
       llvm::concat<const Range>(producer_map->domain.GetSymbolRanges(),
                                 consumer_map->domain.GetSymbolRanges())) {
    if (unused_symbols_bit_vector[symbol_id++]) {
      continue;
    }
    combined_symbol_ranges.push_back(symbol_range);
  }

  IndexingMap composed_indexing_map{
      std::move(composed_map), Domain{consumer_map->domain.GetDimensionRanges(),
                                      combined_symbol_ranges}};
  composed_indexing_map.Simplify();

  RangeEvaluator consumer_range_evaluator(&consumer_map->domain);
  // Add constraints for consumer's codomain w.r.t. producer's domain.
  for (auto [index, expr] :
       llvm::enumerate(consumer_map->affine_map.getResults())) {
    Range consumer_result_range =
        consumer_range_evaluator.ComputeExpressionRange(expr);
    Range producer_dim_range =
        producer_map->domain.GetDimensionRange(static_cast<int64_t>(index));
    // If the constraint is always satisfied, we skip it.
    if (consumer_result_range.upper_bound <= producer_dim_range.upper_bound &&
        consumer_result_range.lower_bound >= producer_dim_range.lower_bound) {
      continue;
    }
    composed_indexing_map.domain.AddConstraint(expr, producer_dim_range);
  }
  return composed_indexing_map;
}

AffineExpr IndexingMapSimplifier::RewriteMod(AffineBinaryOpExpr mod) {
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
    return getAffineConstantExpr(0, mlir_context_);
  }
  // Otherwise, return new_sum % m.
  return new_lhs % mod.getRHS();
}

AffineExpr IndexingMapSimplifier::RewriteFloorDiv(AffineBinaryOpExpr div) {
  auto lhs_simplified = SimplifyOnce(div.getLHS());
  auto lhs = range_evaluator_->ComputeExpressionRange(lhs_simplified);
  auto rhs = range_evaluator_->ComputeExpressionRange(div.getRHS());

  if (0 <= lhs.lower_bound && lhs.upper_bound < rhs.lower_bound) {
    return getAffineConstantExpr(0, mlir_context_);
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
    return getAffineConstantExpr(a, mlir_context_);
  }

  AffineExpr extracted = getAffineConstantExpr(0, mlir_context_);
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
  if (extracted == getAffineConstantExpr(0, mlir_context_) &&
      new_dividend == div.getLHS()) {
    return div;
  }

  return extracted + new_dividend.floorDiv(div.getRHS());
}

std::optional<int64_t> IndexingMapSimplifier::GetConstantRhsMultiplier(
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

AffineExpr IndexingMapSimplifier::RewriteSumIf(
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

AffineExpr IndexingMapSimplifier::SimplifyOnce(AffineExpr expr) {
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
        return getAffineConstantExpr(bounds.lower_bound, mlir_context_);
      }
      return expr;
    }

    default:
      return expr;
  }
}

AffineExpr IndexingMapSimplifier::Simplify(AffineExpr expr) {
  while (true) {
    auto simplified = SimplifyOnce(expr);
    if (simplified == expr) {
      return expr;
    }
    expr = simplified;
  }
}

AffineMap IndexingMapSimplifier::Simplify(AffineMap affine_map) {
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

}  // namespace gpu
}  // namespace xla
