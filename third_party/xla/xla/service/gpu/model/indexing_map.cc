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

}  // namespace

std::string Range::ToString() const {
  std::string s;
  std::stringstream ss(s);
  Print(ss);
  return ss.str();
}

void Range::Print(std::ostream& out) const {
  out << '[' << lower_bound << ", " << upper_bound << ")";
}

std::ostream& operator<<(std::ostream& out, const Range& range) {
  range.Print(out);
  return out;
}

bool operator==(const Range& lhs, const Range& rhs) {
  return lhs.lower_bound == rhs.lower_bound &&
         lhs.upper_bound == rhs.upper_bound;
}

Domain Domain::FromUpperBounds(absl::Span<const int64_t> dimension_upper_bounds,
                               absl::Span<const int64_t> symbol_upper_bounds) {
  Domain domain;
  domain.dimension_ranges.reserve(dimension_upper_bounds.size());
  for (int64_t ub : dimension_upper_bounds) {
    CHECK_GT(ub, 0);
    domain.dimension_ranges.push_back(Range{0, ub - 1});
  }
  domain.symbol_ranges.reserve(symbol_upper_bounds.size());
  for (int64_t ub : symbol_upper_bounds) {
    CHECK_GT(ub, 0);
    domain.symbol_ranges.push_back(Range{0, ub - 1});
  }
  return domain;
}

std::string Domain::ToString(const AffineMapPrinter& printer) const {
  std::string s;
  std::stringstream ss(s);
  Print(ss, printer);
  return ss.str();
}

void Domain::Print(std::ostream& out, const AffineMapPrinter& printer) const {
  for (const auto& [index, range] : llvm::enumerate(dimension_ranges)) {
    out << printer.GetDimensionName(static_cast<int64_t>(index)) << " in "
        << range << '\n';
  }
  for (const auto& [index, range] : llvm::enumerate(symbol_ranges)) {
    out << printer.GetSymbolName(static_cast<int64_t>(index)) << " in " << range
        << '\n';
  }
}

std::ostream& operator<<(std::ostream& out, const Domain& domain) {
  AffineMapPrinter printer;
  domain.Print(out, printer);
  return out;
}

bool operator==(const Domain& lhs, const Domain& rhs) {
  return lhs.dimension_ranges == rhs.dimension_ranges &&
         lhs.symbol_ranges == rhs.symbol_ranges;
}

std::string IndexingMap::ToString(const AffineMapPrinter& printer) const {
  std::string s;
  std::stringstream ss(s);
  Print(ss, printer);
  return ss.str();
}

void IndexingMap::Print(std::ostream& out,
                        const AffineMapPrinter& printer) const {
  printer.Print(out, affine_map);
  out << " with domain\n";
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
  AffineMap simplified_affine_map =
      IndexingMapSimplifier::FromIndexingMap(*this).Simplify(affine_map);
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
  // `domain.symbol_ranges`.
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
  // affine map and the `domain.symbol_ranges`.
  auto unused_symbols_bit_vector =
      mlir::getUnusedSymbolsBitVector({composed_map});
  composed_map = mlir::compressSymbols(composed_map, unused_symbols_bit_vector);

  // The symbols in the composed map, i.e. combined
  // producer_map.compose(consumer_map) are packed as [symbols(producer_map) |
  // symbols(consumer_map)]. In that order we are adding the symbol ranges while
  // skipping the symbols that are unused.
  std::vector<Range> combined_symbol_ranges;
  combined_symbol_ranges.reserve(producer_map->domain.symbol_ranges.size() +
                                 consumer_map->domain.symbol_ranges.size());
  int64_t symbol_id = 0;
  for (const Range& symbol_range :
       llvm::concat<const Range>(producer_map->domain.symbol_ranges,
                                 consumer_map->domain.symbol_ranges)) {
    if (unused_symbols_bit_vector[symbol_id++]) {
      continue;
    }
    combined_symbol_ranges.push_back(symbol_range);
  }
  IndexingMap composed_indexing_map{
      std::move(composed_map),
      Domain{consumer_map->domain.dimension_ranges, combined_symbol_ranges}};
  composed_indexing_map.Simplify();
  return composed_indexing_map;
}

IndexingMapSimplifier IndexingMapSimplifier::FromIndexingMap(
    const IndexingMap& indexing_map) {
  mlir::MLIRContext* mlir_context = indexing_map.affine_map.getContext();
  IndexingMapSimplifier simplifier(mlir_context);

  const Domain& domain = indexing_map.domain;
  for (const auto& [index, range] : llvm::enumerate(domain.dimension_ranges)) {
    simplifier.SetRange(getAffineDimExpr(index, mlir_context),
                        range.lower_bound, range.upper_bound);
  }
  for (const auto& [index, range] : llvm::enumerate(domain.symbol_ranges)) {
    simplifier.SetRange(getAffineSymbolExpr(index, mlir_context),
                        range.lower_bound, range.upper_bound);
  }
  return simplifier;
}

bool IndexingMapSimplifier::IsAlwaysPositiveOrZero(mlir::AffineExpr expr) {
  return GetRange(expr).lower_bound >= 0;
}

bool IndexingMapSimplifier::IsAlwaysNegativeOrZero(mlir::AffineExpr expr) {
  return GetRange(expr).upper_bound <= 0;
}

void IndexingMapSimplifier::SetRange(AffineExpr expr, int64_t lower,
                                     int64_t upper) {
  ranges_[expr] = {lower, upper};
}

Range IndexingMapSimplifier::GetRange(AffineExpr expr) {
  auto bound = ranges_.find(expr);
  if (bound != ranges_.end()) {
    return bound->second;
  }

  switch (expr.getKind()) {
    case AffineExprKind::Constant: {
      int64_t value = mlir::cast<mlir::AffineConstantExpr>(expr).getValue();
      return ranges_[expr] = {value, value};
    }
    case AffineExprKind::DimId: {
      LOG(FATAL) << "Unknown dim "
                 << mlir::cast<mlir::AffineDimExpr>(expr).getPosition();
    }
    case AffineExprKind::SymbolId: {
      LOG(FATAL) << "Unknown symbol"
                 << mlir::cast<mlir::AffineSymbolExpr>(expr).getPosition();
    }
    default:
      auto binary_op = mlir::dyn_cast<AffineBinaryOpExpr>(expr);
      CHECK(binary_op);
      auto lhs = GetRange(binary_op.getLHS());
      auto rhs = GetRange(binary_op.getRHS());

      auto& result = ranges_[expr];
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

AffineExpr IndexingMapSimplifier::RewriteMod(AffineBinaryOpExpr mod) {
  auto lhs_simplified = SimplifyOnce(mod.getLHS());

  auto lhs = GetRange(lhs_simplified);
  auto rhs = GetRange(mod.getRHS());

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

    auto mul_rhs = GetRange(mlir::cast<AffineBinaryOpExpr>(expr).getRHS());
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
  auto lhs = GetRange(lhs_simplified);
  auto rhs = GetRange(div.getRHS());

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
  auto bound = GetRange(mlir::cast<AffineBinaryOpExpr>(expr).getRHS());
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
      auto bounds = GetRange(expr);
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
