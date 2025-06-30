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

#include "xla/hlo/analysis/indexing_map.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/int128.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {
namespace {

using llvm::ArrayRef;
using llvm::SmallBitVector;
using llvm::SmallVector;
using mlir::AffineBinaryOpExpr;
using mlir::AffineConstantExpr;
using mlir::AffineDimExpr;
using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::AffineMap;
using mlir::AffineSymbolExpr;
using mlir::getAffineConstantExpr;
using mlir::getAffineDimExpr;
using mlir::MLIRContext;

AffineExpr GetLhs(AffineExpr e) {
  return mlir::cast<AffineBinaryOpExpr>(e).getLHS();
}

AffineExpr GetRhs(AffineExpr e) {
  return mlir::cast<AffineBinaryOpExpr>(e).getRHS();
}

// Rewrites summands in arbitrarily nested sums (e.g, ((a+b)+c)) by applying
// `fn` to each one. In the example, the result is fn(a)+fn(b)+fn(c).
template <typename Fn>
AffineExpr MapSummands(AffineExpr expr, const Fn& fn) {
  if (expr.getKind() == AffineExprKind::Add) {
    auto add = mlir::cast<AffineBinaryOpExpr>(expr);
    auto lhs = MapSummands(add.getLHS(), fn);
    auto rhs = MapSummands(add.getRHS(), fn);
    if (lhs == add.getLHS() && rhs == add.getRHS()) {
      return add;
    }
    return lhs + rhs;
  }
  return fn(expr);
}

// Calls `visit` for each summand in an arbitrarily nested sum.
template <typename Fn>
void VisitSummands(mlir::AffineExpr expr, const Fn& visit) {
  if (expr.getKind() == AffineExprKind::Add) {
    VisitSummands(GetLhs(expr), visit);
    VisitSummands(GetRhs(expr), visit);
  } else {
    visit(expr);
  }
}

class AffineExprSimplifier {
 public:
  explicit AffineExprSimplifier(
      RangeEvaluator* range_evaluator,
      IndexingMap::SimplifyPointDimensions simplify_point_dimensions =
          IndexingMap::SimplifyPointDimensions::kReplace)
      : range_evaluator_(range_evaluator),
        zero_(getAffineConstantExpr(0, range_evaluator_->GetMLIRContext())),
        simplify_point_dimensions_(simplify_point_dimensions) {}

  // Simplifies the map as much as possible.
  mlir::AffineMap Simplify(mlir::AffineMap affine_map);

  mlir::AffineExpr Simplify(mlir::AffineExpr expr);

  // Performs AffineExpr simplification for all constraints.
  // Returns true if simplification was performed.
  bool SimplifyConstraintExprs(IndexingMap& map);

  // Performs range simplification for all constraints.
  // Returns true if simplification was performed.
  bool SimplifyConstraintRanges(IndexingMap& map);

 private:
  std::optional<int64_t> GetConstantRhs(mlir::AffineExpr expr,
                                        AffineExprKind kind);
  std::pair<mlir::AffineExpr, int64_t> ExtractMultiplier(
      mlir::AffineExpr expr) {
    if (auto mul = GetConstantRhs(expr, AffineExprKind::Mul)) {
      return {GetLhs(expr), *mul};
    }
    return {expr, 1};
  }

  // Simplifier for mod.
  // - Rewrites (a * 100 + ...) % 100 to (...) % 100
  // - Rewrites a % b to a if a is known to be less than b.
  mlir::AffineExpr RewriteMod(mlir::AffineBinaryOpExpr mod);

  // Simplifier for floordiv. Uses all the rules defined below.
  mlir::AffineExpr RewriteFloorDiv(mlir::AffineBinaryOpExpr div);

  // Rewrites `(c % ab) // a` to `(c // a) % b`. Returns nullptr on mismatch.
  AffineExpr SimplifyModDiv(AffineExpr dividend, int64_t divisor);

  // Rewrites `a // b // c` to `a // (b * c)` if `c` is positive. Returns
  // nullptr on mismatch.
  AffineExpr SimplifyDivDiv(AffineExpr dividend, int64_t divisor);

  // Rewrites `a // b` where a may be a sum.
  AffineExpr SimplifySumDiv(AffineExpr dividend, int64_t divisor);

  // Simplifier for mul.
  // - Distributes multiplications with constants over sums.
  mlir::AffineExpr RewriteMul(mlir::AffineBinaryOpExpr mul);

  // Simplifier for sums.
  mlir::AffineExpr RewriteSum(mlir::AffineBinaryOpExpr sum);

  // Attempts to simplify the expression, but doesn't attempt to simplify the
  // result further.
  mlir::AffineExpr SimplifyOnce(mlir::AffineExpr expr);

  // Simplifies the expression using MLIR's simplifier, except for mods.
  mlir::AffineExpr SimplifyWithMlir(mlir::AffineExpr expr, int num_dims,
                                    int num_symbols);

  bool SimplifyConstraintRangeOnce(AffineExpr* expr, Interval* range);
  bool SimplifyConstraintRange(AffineExpr* expr, Interval* range);
  bool SimplifyAddConstraint(AffineExpr* add, Interval* range);

  // Splits a nested sum into a * gcd + b.
  std::tuple<AffineExpr /*a*/, int64_t /*gcd*/, AffineExpr /*b*/> SplitSumByGcd(
      AffineExpr sum);

  RangeEvaluator* range_evaluator_;
  AffineExpr zero_;
  IndexingMap::SimplifyPointDimensions simplify_point_dimensions_;
};

AffineExpr AffineExprSimplifier::RewriteMod(AffineBinaryOpExpr mod) {
  auto rhs = range_evaluator_->ComputeExpressionRange(mod.getRHS());

  // The logic below assumes we have a constant RHS.
  if (!rhs.IsPoint()) {
    return mod;
  }
  int64_t m = rhs.lower;
  // Can only happen in cases where it doesn't matter, return 0.
  if (m == 0) {
    return zero_;
  }

  auto lhs_simplified = SimplifyOnce(mod.getLHS());
  auto lhs = range_evaluator_->ComputeExpressionRange(lhs_simplified);

  // Offset to add to lhs so the lower bound is between 0 and m-1.
  int64_t offset = llvm::divideFloorSigned(lhs.lower, m) * -m;
  // If there's no chance of wraparound, we can replace the mod with an add.
  if (lhs.upper + offset < m) {
    return lhs_simplified + offset;
  }

  // Rewrite `(c * a) % ab` to `(c % b) * a`.
  //   (c * a) % ab
  // = c * a - (c * a) // ab * ab
  // = c * a - c // b * ab
  // = (c - c // b * b) * a
  // = (c % b) * a
  if (auto mul = GetConstantRhs(lhs_simplified, AffineExprKind::Mul);
      mul && *mul > 0 && (m % *mul == 0)) {
    return (GetLhs(lhs_simplified) % (m / *mul)) * *mul;
  }

  int64_t extracted_constant = 0;
  auto new_lhs = MapSummands(lhs_simplified, [&](AffineExpr expr) {
    if (auto cst = mlir::dyn_cast<AffineConstantExpr>(expr)) {
      extracted_constant += cst.getValue();
      return zero_;
    }
    if (auto multiplier = GetConstantRhs(expr, AffineExprKind::Mul);
        multiplier && (*multiplier % m == 0)) {
      return zero_;
    }
    return expr;
  });

  if (extracted_constant % m != 0) {
    new_lhs = new_lhs + (extracted_constant % m);
  }

  // Split the sum into `multiplied * multiplier_gcd + not_multiplied`.
  auto [multiplied, multiplier_gcd, not_multiplied] = SplitSumByGcd(new_lhs);
  if (multiplier_gcd != 1 && m % multiplier_gcd == 0) {
    auto not_multiplied_range =
        range_evaluator_->ComputeExpressionRange(not_multiplied);
    if (not_multiplied_range == Interval{0, 0}) {
      // If b is zero, we can extract the gcd of `multiplier_gcd` and the
      // modulus from the mod.
      int64_t multiplier_mod_gcd = std::gcd(multiplier_gcd, m);
      if (multiplier_mod_gcd == multiplier_gcd) {
        // Special case of the next branch where the multiplications are all
        // * 1.
        new_lhs = multiplied;
      } else if (multiplier_mod_gcd > 1) {
        new_lhs = MapSummands(
            multiplied, [&, multiplier_gcd = multiplier_gcd](AffineExpr expr) {
              return expr * (multiplier_gcd / multiplier_mod_gcd);
            });
      }
      return (new_lhs % (m / multiplier_mod_gcd)) * multiplier_mod_gcd;
    }
    if (Interval{0, multiplier_gcd - 1}.Contains(not_multiplied_range)) {
      // Remove everything that doesn't have a multiplier.
      new_lhs = multiplied * multiplier_gcd;
      return new_lhs % mod.getRHS() + not_multiplied;
    }
  }

  return new_lhs == mod.getLHS() ? mod : (new_lhs % m);
}

AffineExpr AffineExprSimplifier::SimplifyModDiv(AffineExpr dividend,
                                                int64_t divisor) {
  if (auto mod = GetConstantRhs(dividend, AffineExprKind::Mod);
      mod && (*mod % divisor == 0)) {
    return GetLhs(dividend).floorDiv(divisor) % (*mod / divisor);
  }
  return nullptr;
}

AffineExpr AffineExprSimplifier::SimplifyDivDiv(AffineExpr dividend,
                                                int64_t divisor) {
  // The inner divisor here can be negative.
  if (auto inner_divisor = GetConstantRhs(dividend, AffineExprKind::FloorDiv)) {
    return GetLhs(dividend).floorDiv(divisor * *inner_divisor);
  }
  return nullptr;
}

AffineExpr AffineExprSimplifier::SimplifySumDiv(AffineExpr dividend,
                                                int64_t divisor) {
  AffineExpr extracted = zero_;
  auto new_dividend = MapSummands(dividend, [&](AffineExpr expr) {
    if (auto multiplier = GetConstantRhs(expr, AffineExprKind::Mul)) {
      // We can extract summands whose factor is a multiple of the divisor.
      if (*multiplier % divisor == 0) {
        int64_t factor = *multiplier / divisor;
        extracted = extracted + GetLhs(expr) * factor;
        // Remove from dividend.
        return zero_;
      }
    }
    // Not a constant multiplier, keep in dividend.
    return expr;
  });

  // Split `new_dividend` into `multiplied * multiplier_gcd + not_multiplied`.
  auto [multiplied, multiplier_gcd, not_multiplied] =
      SplitSumByGcd(new_dividend);
  int64_t multiplier_divisor_gcd = std::gcd(divisor, multiplier_gcd);

  // Consider an expression like: `(x * 6 + y) / 9`. if the range of `y` is at
  // most `[0; 3)`, we can rewrite it to `(x * 2) / 3`, since `y` can't affect
  // the result.
  auto no_multiplier_range =
      range_evaluator_->ComputeExpressionRange(not_multiplied);
  if (multiplier_divisor_gcd != 1 &&
      Interval{0, multiplier_divisor_gcd - 1}.Contains(no_multiplier_range)) {
    new_dividend = multiplied * (multiplier_gcd / multiplier_divisor_gcd);
    divisor /= multiplier_divisor_gcd;
  } else if (no_multiplier_range.IsPoint() && no_multiplier_range.lower != 0) {
    multiplier_divisor_gcd =
        std::gcd(no_multiplier_range.lower, multiplier_divisor_gcd);
    if (multiplier_divisor_gcd != 1) {
      new_dividend = multiplied * (multiplier_gcd / multiplier_divisor_gcd) +
                     (no_multiplier_range.lower / multiplier_divisor_gcd);
      divisor /= multiplier_divisor_gcd;
    }
  }

  // If we have an inner divisor whose value is equal to the GCD of all the
  // divisors, we can remove a division:
  //   `(a0 / c0 + ...) / c1` -> `(a0 + (...) * c0) / c0c1`
  // This potentially increases the number of multiplications, but it's
  // generally a win. It also matches what the MLIR simplifier does better, so
  // we can get more simplifications. Note that this rewrite is not correct if
  // there's more than one inner division, since each inner dividend may be
  // rounded down, whereas the sum might not be. For example, in
  //   `(a0 / 3 + a1 / 3) / 6)`
  // If a0 is 16 and a1 is 2, the result is `(5 + 0) / 6 = 0`, whereas the
  // rewritten form `(a0 + a1) / 18` evaluates to 1. This can only happen when
  // there is more than one division.
  std::optional<int64_t> inner_divisor = std::nullopt;
  int num_inner_divisors = 0;
  VisitSummands(new_dividend, [&](AffineExpr summand) {
    if (auto divisor = GetConstantRhs(summand, AffineExprKind::FloorDiv)) {
      inner_divisor = divisor;
      ++num_inner_divisors;
    }
  });
  if (num_inner_divisors == 1) {
    new_dividend = MapSummands(new_dividend, [&](AffineExpr summand) {
      if (auto inner_divisor =
              GetConstantRhs(summand, AffineExprKind::FloorDiv)) {
        return GetLhs(summand);
      }
      return summand * *inner_divisor;
    });
    divisor *= *inner_divisor;
  }

  if (new_dividend != dividend) {
    return new_dividend.floorDiv(divisor) + extracted;
  }
  return nullptr;
}

AffineExpr AffineExprSimplifier::RewriteFloorDiv(AffineBinaryOpExpr div) {
  auto rhs_range = range_evaluator_->ComputeExpressionRange(div.getRHS());
  auto lhs_simplified = SimplifyOnce(div.getLHS());
  if (!rhs_range.IsPoint()) {
    return lhs_simplified.floorDiv(SimplifyOnce(div.getRHS()));
  }

  int64_t d = rhs_range.lower;
  // The logic below assumes we have a constant positive RHS.
  if (d > 1) {
    // Rewrite `(c % ab) // a` to `(c // a) % b`.
    if (auto result = SimplifyModDiv(lhs_simplified, d)) {
      return result;
    }

    // Rewrite `((a // b) // c)` to `a // (b * c)`.
    if (auto result = SimplifyDivDiv(lhs_simplified, d)) {
      return result;
    }

    // Rewrite sums on the LHS.
    if (auto result = SimplifySumDiv(lhs_simplified, d)) {
      return result;
    }
  }
  return lhs_simplified != div.getLHS() ? lhs_simplified.floorDiv(d) : div;
}

mlir::AffineExpr AffineExprSimplifier::RewriteMul(
    mlir::AffineBinaryOpExpr mul) {
  auto rhs_range = range_evaluator_->ComputeExpressionRange(mul.getRHS());

  // The logic below assumes we have a constant RHS.
  if (!rhs_range.IsPoint()) {
    return mul;
  }

  int64_t multiplier = rhs_range.lower;
  auto lhs = SimplifyOnce(mul.getLHS());
  if (lhs.getKind() == AffineExprKind::Add) {
    return MapSummands(
        lhs, [&](AffineExpr summand) { return summand * rhs_range.lower; });
  }

  if (multiplier == 1) {
    return lhs;
  }
  if (lhs == mul.getLHS()) {
    return mul;
  }
  return lhs * multiplier;
}

std::optional<int64_t> AffineExprSimplifier::GetConstantRhs(
    AffineExpr expr, AffineExprKind kind) {
  if (expr.getKind() != kind) {
    return std::nullopt;
  }
  auto bound = range_evaluator_->ComputeExpressionRange(
      mlir::cast<AffineBinaryOpExpr>(expr).getRHS());
  if (!bound.IsPoint()) {
    return std::nullopt;
  }
  return bound.lower;
}

// Compares the two expression by their AST. The ordering is arbitrary but
// similar to what MLIR's simplifier does.
int CompareExprs(AffineExpr a, AffineExpr b) {
  if ((b.getKind() == AffineExprKind::Constant) !=
      (a.getKind() == AffineExprKind::Constant)) {
    return a.getKind() == AffineExprKind::Constant ? 1 : -1;
  }
  if (a.getKind() < b.getKind()) {
    return -1;
  }
  if (a.getKind() > b.getKind()) {
    return 1;
  }
  assert(a.getKind() == b.getKind());
  int64_t a_value = 0;
  int64_t b_value = 0;
  switch (a.getKind()) {
    case AffineExprKind::Add:
    case AffineExprKind::FloorDiv:
    case AffineExprKind::CeilDiv:
    case AffineExprKind::Mul:
    case AffineExprKind::Mod: {
      auto lhs = CompareExprs(GetLhs(a), GetLhs(b));
      if (lhs != 0) {
        return lhs;
      }
      return CompareExprs(GetRhs(a), GetRhs(b));
    }
    case AffineExprKind::Constant: {
      a_value = mlir::cast<AffineConstantExpr>(a).getValue();
      b_value = mlir::cast<AffineConstantExpr>(b).getValue();
      break;
    }
    case AffineExprKind::SymbolId: {
      a_value = mlir::cast<AffineSymbolExpr>(a).getPosition();
      b_value = mlir::cast<AffineSymbolExpr>(b).getPosition();
      break;
    }
    case AffineExprKind::DimId: {
      a_value = mlir::cast<AffineDimExpr>(a).getPosition();
      b_value = mlir::cast<AffineDimExpr>(b).getPosition();
      break;
    }
  }
  return a_value < b_value ? -1 : (a_value > b_value ? 1 : 0);
}

mlir::AffineExpr AffineExprSimplifier::RewriteSum(
    mlir::AffineBinaryOpExpr sum) {
  // TODO(jreiffers): Split this up more.
  // Rewrite `(x % c) * d + (x // c) * (c * d)` to `x * d`. We have to do it
  // in this rather convoluted way because we distribute multiplications.
  SmallVector<std::pair<AffineExpr, int64_t /*multiplier*/>> mods;
  SmallVector<std::pair<AffineExpr, int64_t /*multiplier*/>> divs;
  llvm::SmallDenseMap<AffineExpr, int64_t /* multiplier */> summands;
  VisitSummands(sum, [&](AffineExpr expr) {
    AffineExpr simplified = SimplifyOnce(expr);
    auto [lhs, multiplier] = ExtractMultiplier(simplified);
    if (lhs.getKind() == AffineExprKind::Mod) {
      mods.push_back({lhs, multiplier});
    } else if (lhs.getKind() == AffineExprKind::FloorDiv) {
      divs.push_back({lhs, multiplier});
    } else {
      summands[lhs] += multiplier;
    }
  });

  if (mods.size() * divs.size() >= 100) {
    std::string s;
    llvm::raw_string_ostream ss(s);
    ss << sum;
    LOG(WARNING) << "Unexpectedly large number of mods and divs in " << s
                 << ". Please open an issue on GitHub at "
                 << "https://github.com/openxla/xla.";
  }

  if (!divs.empty()) {
    for (int mod_i = 0; mod_i < mods.size(); ++mod_i) {
      auto [mod, mod_mul] = mods[mod_i];
      auto mod_c = GetConstantRhs(mod, AffineExprKind::Mod);
      if (!mod_c) {
        continue;
      }

      // In many cases, we could just compare the LHSes of the mod and the
      // div, but if x is a floorDiv itself, we need to check a bit more
      // carefully:
      //    ((x // c0) % c1) * d + (x // (c0 * c1)) * (c1 * d)`
      // `x // (c0 * c1)` will be simplified, so we we may not even have
      // `c0 * c1` in the expression, if `x` contains a multiplier.
      AffineExpr simplified_mod = Simplify(GetLhs(mod).floorDiv(*mod_c));
      for (int div_i = 0; div_i < divs.size(); ++div_i) {
        auto [div, div_mul] = divs[div_i];
        if (simplified_mod != div) {
          continue;
        }
        if ((div_mul % mod_mul) || (div_mul / mod_mul) != mod_c) {
          continue;
        }

        summands[GetLhs(mod)] += mod_mul;
        divs[div_i].first = nullptr;
        mods[mod_i].first = nullptr;
        break;
      }
    }

    // (x - (x floordiv div_c) * div_c) * b = (x mod a) * b.
    // We do this even if there is no x in the sum.
    for (int div_i = 0; div_i < divs.size(); ++div_i) {
      auto [div, div_mul] = divs[div_i];
      if (!div || div_mul > 0) {
        continue;
      }
      auto div_c = GetConstantRhs(div, AffineExprKind::FloorDiv);
      if (!div_c || *div_c < 0 || (div_mul % *div_c)) {
        continue;
      }

      int64_t b = div_mul / *div_c;
      auto x = GetLhs(div);
      VisitSummands(x, [&](AffineExpr summand) { summands[summand] += b; });
      mods.push_back({x % *div_c, -b});
      // Erase the div.
      divs[div_i].first = nullptr;
    }
  }

  for (auto [expr, mul] : mods) {
    if (expr) {
      summands[expr] += mul;
    }
  }
  for (auto [expr, mul] : divs) {
    if (expr) {
      summands[expr] += mul;
    }
  }

  SmallVector<AffineExpr, 4> expanded_summands;
  for (auto [expr, mul] : summands) {
    expanded_summands.push_back(expr * mul);
  }
  llvm::sort(expanded_summands,
             [](AffineExpr a, AffineExpr b) { return CompareExprs(a, b) < 0; });
  AffineExpr result = zero_;
  for (auto expr : expanded_summands) {
    result = result + expr;
  }
  return result;
}

AffineExpr AffineExprSimplifier::SimplifyOnce(AffineExpr expr) {
  if (expr.getKind() == AffineExprKind::Constant) {
    return expr;
  }

  if (simplify_point_dimensions_ ==
      IndexingMap::SimplifyPointDimensions::kReplace) {
    auto bounds = range_evaluator_->ComputeExpressionRange(expr);
    if (bounds.IsPoint()) {
      return getAffineConstantExpr(bounds.lower,
                                   range_evaluator_->GetMLIRContext());
    }
  }

  switch (expr.getKind()) {
    case AffineExprKind::Mul:
      return RewriteMul(mlir::cast<AffineBinaryOpExpr>(expr));
    case AffineExprKind::Add:
      return RewriteSum(mlir::cast<AffineBinaryOpExpr>(expr));
    case AffineExprKind::Mod:
      return RewriteMod(mlir::cast<AffineBinaryOpExpr>(expr));
    case AffineExprKind::FloorDiv:
      return RewriteFloorDiv(mlir::cast<AffineBinaryOpExpr>(expr));
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
  SmallVector<AffineExpr, 4> results;
  results.reserve(affine_map.getNumResults());
  for (AffineExpr expr : affine_map.getResults()) {
    results.push_back(Simplify(expr));
  }
  return AffineMap::get(affine_map.getNumDims(), affine_map.getNumSymbols(),
                        results, affine_map.getContext());
}

bool AffineExprSimplifier::SimplifyAddConstraint(AffineExpr* add,
                                                 Interval* range) {
  if (add->getKind() != AffineExprKind::Add) {
    return false;
  }

  auto rhs_range = range_evaluator_->ComputeExpressionRange(GetRhs(*add));
  if (rhs_range.IsPoint()) {
    *add = GetLhs(*add);
    range->lower -= rhs_range.lower;
    range->upper -= rhs_range.lower;
    return true;
  }

  if (range->lower != 0) {
    return false;
  }

  // Split the sum into `multiplied * multiplier_gcd + not_multiplied`.
  //   0 <= a * gcd + b <= ub
  //   0 <= a * gcd <= ub - b
  //   0 <= a <= (ub - b) floordiv gcd
  // If `(ub - b) floordiv gcd` is a constant, that means the value of b is
  // irrelevant to this constraint.
  auto [multiplied, multiplier_gcd, not_multiplied] = SplitSumByGcd(*add);
  if (multiplier_gcd == 1) {
    // If we didn't split anything, there's nothing to do.
    return false;
  }

  Interval difference_range =
      Interval{range->upper, range->upper} -
      range_evaluator_->ComputeExpressionRange(not_multiplied);
  if (!difference_range.FloorDiv(multiplier_gcd).IsPoint()) {
    return false;
  }

  *add = multiplied * multiplier_gcd;
  return true;
}

// Simplifies a constraint range, e.g. a constraint d0 + x in [lb, ub] will
// become d0 in [lb - x, ub - x]. Also supports *, floorDiv.
bool AffineExprSimplifier::SimplifyConstraintRangeOnce(AffineExpr* expr,
                                                       Interval* range) {
  switch (expr->getKind()) {
    case AffineExprKind::DimId:
    case AffineExprKind::SymbolId:
      // do the trick with constant
    case AffineExprKind::Constant: {
      return false;
    }
    case AffineExprKind::Add:
      return SimplifyAddConstraint(expr, range);
    default: {
      auto binary_op = mlir::cast<AffineBinaryOpExpr>(*expr);
      CHECK(binary_op);
      auto lhs = binary_op.getLHS();
      auto rhs_range = range_evaluator_->ComputeExpressionRange(GetRhs(*expr));
      if (!rhs_range.IsPoint()) {
        return false;
      }
      int64_t rhs_cst = rhs_range.lower;
      switch (expr->getKind()) {
        case AffineExprKind::Mul: {
          int64_t factor = rhs_cst;
          if (factor < 0) {
            factor *= -1;
            range->lower *= -1;
            range->upper *= -1;
            std::swap(range->lower, range->upper);
          }
          range->lower = llvm::divideCeilSigned(range->lower, factor);
          range->upper = llvm::divideFloorSigned(range->upper, factor);
          *expr = lhs;
          return true;
        }
        case AffineExprKind::FloorDiv: {
          int64_t divisor = rhs_cst;
          if (divisor < 0) {
            divisor *= -1;
            range->lower *= -1;
            range->upper *= -1;
            std::swap(range->lower, range->upper);
          }
          range->lower *= divisor;
          range->upper = (range->upper + 1) * divisor - 1;
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
bool AffineExprSimplifier::SimplifyConstraintRange(AffineExpr* expr,
                                                   Interval* range) {
  bool is_simplified = false;
  while (SimplifyConstraintRangeOnce(expr, range)) {
    is_simplified = true;
  }
  return is_simplified;
}

// Computes the symbols list replacement to go from
// [range_vars(second)|rt_vars(second)|range_vars(first)|rt_vars(first)]
// to
// [range_vars(second)|range_vars(first)|rt_vars(second)|rt_vars(first)].
// If an empty vector is returned, no replacement is needed.
SmallVector<AffineExpr, 4> GetComposedSymbolsPermutationToCorrectOrder(
    const IndexingMap& first, const IndexingMap& second) {
  // No permutation is needed if the second map has no RTVars.
  if (second.GetRTVarsCount() == 0) {
    return {};
  }
  SmallVector<AffineExpr, 4> symbol_replacements;
  MLIRContext* mlir_context = first.GetMLIRContext();
  for (int id = 0; id < second.GetRangeVarsCount(); ++id) {
    symbol_replacements.push_back(getAffineSymbolExpr(id, mlir_context));
  }
  int64_t first_range_vars_count = first.GetRangeVarsCount();
  int64_t second_range_vars_count = second.GetRangeVarsCount();
  int64_t first_rt_vars_count = first.GetRTVarsCount();
  int64_t second_rt_vars_count = second.GetRTVarsCount();
  int64_t rt_vars_second_start =
      first_range_vars_count + second_range_vars_count;
  for (int64_t id = 0; id < second_rt_vars_count; ++id) {
    symbol_replacements.push_back(
        getAffineSymbolExpr(rt_vars_second_start++, mlir_context));
  }
  int64_t range_vars_first_start = second_range_vars_count;
  for (int64_t id = 0; id < first_range_vars_count; ++id) {
    symbol_replacements.push_back(
        getAffineSymbolExpr(range_vars_first_start++, mlir_context));
  }
  int64_t rt_vars_first_start =
      first_range_vars_count + second_range_vars_count + second_rt_vars_count;
  for (int64_t id = 0; id < first_rt_vars_count; ++id) {
    symbol_replacements.push_back(
        getAffineSymbolExpr(rt_vars_first_start++, mlir_context));
  }
  return symbol_replacements;
}

// Computes the symbols list mapping to go from
// [range_vars(map)|rt_vars(map)]
// to
// [range_vars(second)|range_vars(first)|rt_vars(second)|rt_vars(first)].
SmallVector<AffineExpr, 4> MapSymbolsToComposedSymbolsList(
    const IndexingMap& map, const IndexingMap& composed) {
  SmallVector<AffineExpr, 4> symbol_replacements;

  MLIRContext* mlir_context = map.GetMLIRContext();
  int64_t range_vars_start =
      composed.GetRangeVarsCount() - map.GetRangeVarsCount();
  for (int64_t id = 0; id < map.GetRangeVarsCount(); ++id) {
    symbol_replacements.push_back(
        getAffineSymbolExpr(range_vars_start++, mlir_context));
  }
  int64_t rt_vars_start = composed.GetSymbolCount() - map.GetRTVarsCount();
  for (int64_t id = 0; id < map.GetRTVarsCount(); ++id) {
    symbol_replacements.push_back(
        getAffineSymbolExpr(rt_vars_start++, mlir_context));
  }
  return symbol_replacements;
}

}  // namespace

// TODO(willfroom): Change the names to work items/groups.
static constexpr absl::string_view kVarKindDefault = "default";
static constexpr absl::string_view kVarKindThreadX = "th_x";
static constexpr absl::string_view kVarKindThreadY = "th_y";
static constexpr absl::string_view kVarKindThreadZ = "th_z";
static constexpr absl::string_view kVarKindBlockX = "bl_x";
static constexpr absl::string_view kVarKindBlockY = "bl_y";
static constexpr absl::string_view kVarKindBlockZ = "bl_z";
static constexpr absl::string_view kVarKindWarp = "warp";
static constexpr absl::string_view kVarKindWarpThread = "th_w";

absl::string_view ToVariableName(VariableKind var_kind) {
  switch (var_kind) {
    case VariableKind::kDefault:
      return kVarKindDefault;
    case VariableKind::kThreadX:
      return kVarKindThreadX;
    case VariableKind::kThreadY:
      return kVarKindThreadY;
    case VariableKind::kThreadZ:
      return kVarKindThreadZ;
    case VariableKind::kBlockX:
      return kVarKindBlockX;
    case VariableKind::kBlockY:
      return kVarKindBlockY;
    case VariableKind::kBlockZ:
      return kVarKindBlockZ;
    case VariableKind::kWarp:
      return kVarKindWarp;
    case VariableKind::kWarpThread:
      return kVarKindWarpThread;
  }
  llvm_unreachable("Unknown VariableType");
}

VariableKind ToVariableType(absl::string_view var_name) {
  if (var_name == kVarKindThreadX) {
    return VariableKind::kThreadX;
  }
  if (var_name == kVarKindThreadY) {
    return VariableKind::kThreadY;
  }
  if (var_name == kVarKindThreadZ) {
    return VariableKind::kThreadZ;
  }
  if (var_name == kVarKindBlockX) {
    return VariableKind::kBlockX;
  }
  if (var_name == kVarKindBlockY) {
    return VariableKind::kBlockY;
  }
  if (var_name == kVarKindBlockZ) {
    return VariableKind::kBlockZ;
  }
  if (var_name == kVarKindWarp) {
    return VariableKind::kWarp;
  }
  if (var_name == kVarKindWarpThread) {
    return VariableKind::kWarpThread;
  }
  return VariableKind::kDefault;
}

std::ostream& operator<<(std::ostream& out, VariableKind var_type) {
  out << ToVariableName(var_type);
  return out;
}

std::ostream& operator<<(std::ostream& out, const Interval& interval) {
  out << absl::StrFormat("[%d, %d]", interval.lower, interval.upper);
  return out;
}

std::string Interval::ToString() const {
  std::stringstream ss;
  ss << *this;
  return ss.str();
}

inline llvm::raw_ostream& operator<<(llvm::raw_ostream& os,
                                     const Interval& interval) {
  os << absl::StrFormat("[%d, %d]", interval.lower, interval.upper);
  return os;
}

int64_t Interval::GetLoopTripCount() const {
  if (!IsFeasible()) {
    return 0;
  }
  DCHECK((static_cast<absl::int128>(upper) - lower + 1) <=
         std::numeric_limits<int64_t>::max());
  return upper - lower + 1;
}

Interval::ComparisonResult Interval::Gt(const Interval& b) const {
  if (!IsFeasible() || !b.IsFeasible()) {
    return {std::nullopt};
  }
  if (lower > b.upper) {
    return {true};
  }
  if (upper <= b.lower) {
    return {false};
  }
  return {std::nullopt};
}

Interval::ComparisonResult Interval::Eq(const Interval& b) const {
  Interval intersection = Intersect(b);
  if (!intersection.IsFeasible()) {
    return {false};
  }
  if (intersection.IsPoint() && IsPoint() && b.IsPoint()) {
    return {true};
  }
  return {std::nullopt};
}

Interval Interval::operator+(const Interval& rhs) const {
  int64_t out_lower;
  int64_t out_upper;

  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

  bool lower_overflow = llvm::AddOverflow(lower, rhs.lower, out_lower);
  bool upper_overflow = llvm::AddOverflow(upper, rhs.upper, out_upper);

  if (lower_overflow || lower == kMin || rhs.lower == kMin) {
    if (lower < 0 || rhs.lower < 0) {
      out_lower = kMin;
    } else {
      out_lower = kMax;
      out_upper = kMax;
    }
  }

  if (upper_overflow || upper == kMax || rhs.upper == kMax) {
    if (upper > 0 || rhs.upper > 0) {
      out_upper = kMax;
    } else {
      out_upper = kMin;
      out_lower = kMin;
    }
  }

  return {out_lower, out_upper};
}

Interval Interval::operator*(const Interval& rhs) const {
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

  auto mul = [&](int64_t p) {
    int64_t l = lower;
    int64_t u = upper;
    if (p < 0) {
      std::swap(l, u);
    }
    int64_t out_lower;
    int64_t out_upper;
    if (llvm::MulOverflow(l, p, out_lower) ||
        // -1 * max is min + 1, and doesn't overflow. We consider max a
        // special sentinel value, so the result should be min (= saturated).
        (p == -1 && l == kMax)) {
      out_lower = kMin;
    }
    if (llvm::MulOverflow(u, p, out_upper)) {
      out_upper = kMax;
    }
    return Interval{out_lower, out_upper};
  };

  return mul(rhs.lower).Union(mul(rhs.upper));
}

Interval Interval::operator-() const {
  int64_t ub = lower == std::numeric_limits<int64_t>::min()
                   ? std::numeric_limits<int64_t>::max()
                   : -lower;
  int64_t lb = upper == std::numeric_limits<int64_t>::max()
                   ? std::numeric_limits<int64_t>::min()
                   : -upper;
  return Interval{lb, ub};
}

Interval Interval::FloorDiv(int64_t rhs) const {
  auto saturate_div = [](int64_t lhs, int64_t rhs) {
    constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
    if (lhs == kMin) {
      return rhs > 0 ? kMin : kMax;
    }
    if (lhs == kMax) {
      return rhs > 0 ? kMax : kMin;
    }
    return llvm::divideFloorSigned(lhs, rhs);
  };

  int64_t a = saturate_div(lower, rhs);
  int64_t b = saturate_div(upper, rhs);
  return {std::min(a, b), std::max(a, b)};
}

bool operator==(const IndexingMap::Variable& lhs,
                const IndexingMap::Variable& rhs) {
  return lhs.bounds == rhs.bounds;
}

std::vector<IndexingMap::Variable> DimVarsFromTensorSizes(
    absl::Span<const int64_t> tensor_sizes) {
  std::vector<IndexingMap::Variable> ranges;
  ranges.reserve(tensor_sizes.size());
  for (int64_t size : tensor_sizes) {
    ranges.emplace_back(0, size - 1);
  }
  return ranges;
}

std::vector<IndexingMap::Variable> DimVarsFromGPUGrid(
    absl::Span<const int64_t> grid_sizes) {
  CHECK_EQ(grid_sizes.size(), 6)
      << "Grid must be 6-dimensional (th_x, th_y, th_z, bl_x, bl_y, bl_z)";
  return {
      IndexingMap::Variable{0, grid_sizes[0] - 1, kVarKindThreadX},
      IndexingMap::Variable{0, grid_sizes[1] - 1, kVarKindThreadY},
      IndexingMap::Variable{0, grid_sizes[2] - 1, kVarKindThreadZ},
      IndexingMap::Variable{0, grid_sizes[3] - 1, kVarKindBlockX},
      IndexingMap::Variable{0, grid_sizes[4] - 1, kVarKindBlockY},
      IndexingMap::Variable{0, grid_sizes[5] - 1, kVarKindBlockZ},
  };
}

std::vector<IndexingMap::Variable> RangeVarsFromTensorSizes(
    absl::Span<const int64_t> tensor_sizes) {
  return DimVarsFromTensorSizes(tensor_sizes);
}

IndexingMap::IndexingMap(
    AffineMap affine_map, std::vector<IndexingMap::Variable> dimensions,
    std::vector<IndexingMap::Variable> range_vars,
    std::vector<IndexingMap::Variable> rt_vars,
    absl::Span<std::pair<AffineExpr, Interval> const> constraints)
    : affine_map_(affine_map),
      dim_vars_(std::move(dimensions)),
      range_vars_(std::move(range_vars)),
      rt_vars_(std::move(rt_vars)) {
  if (!VerifyVariableIntervals()) {
    ResetToKnownEmpty();
    return;
  }
  for (const auto& [expr, range] : constraints) {
    AddConstraint(expr, range);
  }
}

IndexingMap::IndexingMap(
    AffineMap affine_map, std::vector<IndexingMap::Variable> dimensions,
    std::vector<IndexingMap::Variable> range_vars,
    std::vector<IndexingMap::Variable> rt_vars,
    const llvm::DenseMap<AffineExpr, Interval>& constraints)
    : affine_map_(affine_map),
      dim_vars_(std::move(dimensions)),
      range_vars_(std::move(range_vars)),
      rt_vars_(std::move(rt_vars)),
      constraints_(constraints) {
  if (!VerifyVariableIntervals() || !VerifyConstraintIntervals()) {
    ResetToKnownEmpty();
    return;
  }
}

IndexingMap IndexingMap::FromTensorSizes(
    AffineMap affine_map, absl::Span<const int64_t> dim_upper_bounds,
    absl::Span<const int64_t> symbol_upper_bounds) {
  return IndexingMap{affine_map, DimVarsFromTensorSizes(dim_upper_bounds),
                     RangeVarsFromTensorSizes(symbol_upper_bounds),
                     /*rt_vars=*/{}};
}

RangeEvaluator IndexingMap::GetRangeEvaluator() const {
  return RangeEvaluator(*this, GetMLIRContext());
}

const Interval& IndexingMap::GetDimensionBound(int64_t dim_id) const {
  return dim_vars_[dim_id].bounds;
}

Interval& IndexingMap::GetMutableDimensionBound(int64_t dim_id) {
  return dim_vars_[dim_id].bounds;
}

std::vector<Interval> IndexingMap::GetDimensionBounds() const {
  std::vector<Interval> bounds;
  bounds.reserve(affine_map_.getNumDims());
  for (const auto& dim : dim_vars_) {
    bounds.push_back(dim.bounds);
  }
  return bounds;
}

const Interval& IndexingMap::GetSymbolBound(int64_t symbol_id) const {
  // Because affine map symbols are packed like [range_vars, rt_vars],
  // we have to pick the correct bounds.
  int64_t range_var_count = GetRangeVarsCount();
  return symbol_id < range_var_count
             ? range_vars_[symbol_id].bounds
             : rt_vars_[symbol_id - range_var_count].bounds;
}

Interval& IndexingMap::GetMutableSymbolBound(int64_t symbol_id) {
  // Because affine map symbols are packed like [range_vars, rt_vars],
  // we have to pick the correct bounds.
  int64_t range_var_count = GetRangeVarsCount();
  return symbol_id < range_var_count
             ? range_vars_[symbol_id].bounds
             : rt_vars_[symbol_id - range_var_count].bounds;
}

std::vector<Interval> IndexingMap::GetSymbolBounds() const {
  std::vector<Interval> bounds;
  bounds.reserve(affine_map_.getNumSymbols());
  for (const auto& range_var : range_vars_) {
    bounds.push_back(range_var.bounds);
  }
  for (const auto& rt_var : rt_vars_) {
    bounds.push_back(rt_var.bounds);
  }
  return bounds;
}

void IndexingMap::AddConstraint(mlir::AffineExpr expr, Interval range) {
  // Do not add the constraint if the domain is already empty.
  if (IsKnownEmpty()) {
    return;
  }
  // If the range is empty, reset the indexing map to the canonical empty form.
  if (!range.IsFeasible()) {
    ResetToKnownEmpty();
    return;
  }
  if (auto dim_expr = mlir::dyn_cast<AffineDimExpr>(expr)) {
    Interval& current_range = GetMutableDimensionBound(dim_expr.getPosition());
    current_range = current_range.Intersect(range);
    if (!current_range.IsFeasible()) {
      ResetToKnownEmpty();
    }
    return;
  }
  if (auto symbol_expr = mlir::dyn_cast<AffineSymbolExpr>(expr)) {
    Interval& current_range = GetMutableSymbolBound(symbol_expr.getPosition());
    current_range = current_range.Intersect(range);
    if (!current_range.IsFeasible()) {
      ResetToKnownEmpty();
    }
    return;
  }
  if (auto constant_expr = mlir::dyn_cast<AffineConstantExpr>(expr)) {
    if (!range.Contains(constant_expr.getValue())) {
      ResetToKnownEmpty();
    }
    return;
  }
  auto [it, inserted] = constraints_.insert({expr, range});
  if (!inserted) {
    it->second = it->second.Intersect(range);
    if (!it->second.IsFeasible()) {
      ResetToKnownEmpty();
    }
  }
}

void IndexingMap::EraseConstraint(mlir::AffineExpr expr) {
  constraints_.erase(expr);
}

bool IndexingMap::ConstraintsSatisfied(
    ArrayRef<AffineExpr> dim_const_exprs,
    ArrayRef<AffineExpr> symbol_const_exprs) const {
  CHECK(dim_const_exprs.size() == affine_map_.getNumDims());
  CHECK(symbol_const_exprs.size() == affine_map_.getNumSymbols());
  if (IsKnownEmpty()) {
    return false;
  }
  for (auto& [expr, range] : constraints_) {
    int64_t expr_value =
        mlir::cast<AffineConstantExpr>(
            expr.replaceDimsAndSymbols(dim_const_exprs, symbol_const_exprs))
            .getValue();
    if (expr_value < range.lower || expr_value > range.upper) {
      return false;
    }
  }
  return true;
}

SmallVector<int64_t, 4> IndexingMap::Evaluate(
    ArrayRef<AffineExpr> dim_const_exprs,
    ArrayRef<AffineExpr> symbol_const_exprs) const {
  CHECK(dim_const_exprs.size() == GetDimensionCount());
  CHECK(symbol_const_exprs.size() == GetSymbolCount());
  AffineMap eval = affine_map_.replaceDimsAndSymbols(
      dim_const_exprs, symbol_const_exprs, dim_const_exprs.size(),
      symbol_const_exprs.size());
  return eval.getConstantResults();
}

bool IndexingMap::IsSymbolConstrained(int64_t symbol_id) const {
  for (const auto& [expr, _] : constraints_) {
    bool result = false;
    expr.walk([&](mlir::AffineExpr leaf) {
      auto sym = mlir::dyn_cast<mlir::AffineSymbolExpr>(leaf);
      if (sym && sym.getPosition() == symbol_id) {
        result = true;
      }
    });
    if (result) {
      return true;
    }
  }
  return false;
}

void IndexingMap::RenameDimVar(int64_t id, absl::string_view new_name) {
  CHECK_LT(id, dim_vars_.size());
  dim_vars_[id].name = new_name;
}

RangeEvaluator::RangeEvaluator(const IndexingMap& indexing_map,
                               MLIRContext* mlir_context, bool use_constraints)
    : mlir_context_(mlir_context),
      indexing_map_(indexing_map),
      use_constraints_(use_constraints) {}

bool RangeEvaluator::IsAlwaysPositiveOrZero(mlir::AffineExpr expr) {
  return ComputeExpressionRange(expr).lower >= 0;
}

bool RangeEvaluator::IsAlwaysNegativeOrZero(mlir::AffineExpr expr) {
  return ComputeExpressionRange(expr).upper <= 0;
}

Interval RangeEvaluator::ComputeExpressionRange(AffineExpr expr) {
  switch (expr.getKind()) {
    case AffineExprKind::Constant: {
      int64_t value = mlir::cast<AffineConstantExpr>(expr).getValue();
      return Interval{value, value};
    }
    case AffineExprKind::DimId:
      return indexing_map_.GetDimensionBound(
          mlir::cast<AffineDimExpr>(expr).getPosition());
    case AffineExprKind::SymbolId:
      return indexing_map_.GetSymbolBound(
          mlir::cast<AffineSymbolExpr>(expr).getPosition());
    default:
      break;
  }
  auto binary_op = mlir::dyn_cast<AffineBinaryOpExpr>(expr);
  CHECK(binary_op);
  auto lhs = ComputeExpressionRange(binary_op.getLHS());
  auto rhs = ComputeExpressionRange(binary_op.getRHS());

  Interval result;
  switch (expr.getKind()) {
    case AffineExprKind::Add:
      result = lhs + rhs;
      break;
    case AffineExprKind::Mul:
      result = lhs * rhs;
      break;
    case AffineExprKind::Mod: {
      CHECK(rhs.IsPoint()) << "RHS of mod must be a constant";
      int64_t m = rhs.lower;
      if (0 <= lhs.lower && lhs.upper < m) {
        result = lhs;
      } else {
        result = {0, m - 1};
      }
      break;
    }
    case AffineExprKind::FloorDiv: {
      CHECK(rhs.IsPoint()) << "RHS of floor_div must be a constant";
      int64_t d = rhs.lower;
      // TODO(jreiffers): Implement saturating semantics.
      int64_t a = llvm::divideFloorSigned(lhs.lower, d);
      int64_t b = llvm::divideFloorSigned(lhs.upper, d);
      result = {std::min(a, b), std::max(a, b)};
      break;
    }
    default:
      // We don't use ceildiv, so we don't support it.
      LOG(FATAL) << "Unsupported expression";
  }

  if (use_constraints_) {
    auto constraint = indexing_map_.GetConstraints().find(expr);
    if (constraint != indexing_map_.GetConstraints().end()) {
      return result.Intersect(constraint->second);
    }
  }
  return result;
}

MLIRContext* IndexingMap::GetMLIRContext() const {
  return IsUndefined() ? nullptr : affine_map_.getContext();
}

bool operator==(const IndexingMap& lhs, const IndexingMap& rhs) {
  return lhs.GetAffineMap() == rhs.GetAffineMap() &&
         lhs.GetDimVars() == rhs.GetDimVars() &&
         lhs.GetRangeVars() == rhs.GetRangeVars() &&
         lhs.GetRTVars() == rhs.GetRTVars() &&
         lhs.GetConstraints() == rhs.GetConstraints();
}

IndexingMap operator*(const IndexingMap& lhs, const IndexingMap& rhs) {
  return ComposeIndexingMaps(lhs, rhs);
}

bool IndexingMap::Verify(std::ostream& out) const {
  if (IsUndefined()) {
    return true;
  }
  if (affine_map_.getNumDims() != dim_vars_.size()) {
    out << absl::StrCat(
        "number of dim vars (", dim_vars_.size(),
        ") must match the number of dimensions in the affine map (",
        affine_map_.getNumDims(), ")");
    return false;
  }
  if (affine_map_.getNumSymbols() != range_vars_.size() + rt_vars_.size()) {
    out << absl::StrCat(
        "number of range (", range_vars_.size(), ") + runtime (",
        rt_vars_.size(),
        ") variables must match the number of symbols in the affine map (",
        affine_map_.getNumSymbols(), ")");
    return false;
  }
  return true;
}

// Simplification of IndexingMap has two main parts.
// At first we optimized constraints to make the domain as small and simple as
// possible. And only then we simplify the affine_map, because its
// simplification relies on lower/upper bounds of dimensions and symbols.

// Constraint simplification is performed in two stages repeated until
// convergence.
//   1. Simplify affine expressions in all constraints.
//   2. Simplify constraint ranges for all constraints.
// We don't optimize every constraint separately to avoid re-initialization of
// RangeEvaluator for every constraint. Note that we start with "expr"
// simplification, because the ranges of constraints were already optimized once
// when IndexingMap was constructed.
bool IndexingMap::Simplify(SimplifyPointDimensions simplify_point_dimensions) {
  if (IsUndefined() || IsKnownEmpty()) {
    return false;
  }

  // Simplify constraints to shrink the lower/upper bounds of dims and symbols.
  bool constraints_were_simplified = false;

  // Simplify affine_map using the optimized ranges.
  // Potentially, we can be smarter about recreating the range_evaluator.
  RangeEvaluator constraint_range_evaluator(*this, GetMLIRContext(),
                                            /*use_constraints=*/false);
  AffineExprSimplifier constraint_simplifier(&constraint_range_evaluator);
  while (true) {
    bool did_simplify = false;
    did_simplify |= constraint_simplifier.SimplifyConstraintExprs(*this);
    did_simplify |= constraint_simplifier.SimplifyConstraintRanges(*this);
    if (!did_simplify) {
      break;
    }
    constraints_were_simplified = true;
  }
  // Simplify dependent constraints.
  constraints_were_simplified |= MergeModConstraints();
  RangeEvaluator range_evaluator(*this, GetMLIRContext(),
                                 /*use_constraints=*/true);
  AffineMap simplified_affine_map =
      AffineExprSimplifier(&range_evaluator, simplify_point_dimensions)
          .Simplify(affine_map_);
  bool affine_map_was_simplified = simplified_affine_map != affine_map_;
  if (affine_map_was_simplified) {
    affine_map_ = simplified_affine_map;
  }
  return affine_map_was_simplified || constraints_were_simplified;
}

bool AffineExprSimplifier::SimplifyConstraintExprs(IndexingMap& map) {
  // Simplify affine expression in the constraints_.
  std::vector<AffineExpr> to_remove;
  std::vector<std::pair<AffineExpr, Interval>> to_add;
  for (const auto& [expr, range] : map.GetConstraints()) {
    AffineExpr simplified = Simplify(expr);

    // Skip constraints that are always satisfied.
    Interval evaluated_range =
        range_evaluator_->ComputeExpressionRange(simplified);
    if (evaluated_range.upper <= range.upper &&
        evaluated_range.lower >= range.lower) {
      to_remove.push_back(expr);
      continue;
    }
    if (simplified == expr) {
      continue;
    }
    to_add.push_back({simplified, range});
    to_remove.push_back(expr);
  }
  for (const auto& expr : to_remove) {
    map.EraseConstraint(expr);
  }
  for (const auto& [expr, range] : to_add) {
    map.AddConstraint(expr, range);
  }
  return !to_add.empty();
}

bool AffineExprSimplifier::SimplifyConstraintRanges(IndexingMap& map) {
  std::vector<AffineExpr> to_remove;
  std::vector<std::pair<AffineExpr, Interval>> to_add;
  for (const auto& [expr, range] : map.GetConstraints()) {
    AffineExpr simplified_expr = expr;
    Interval simplified_range = range;
    if (SimplifyConstraintRange(&simplified_expr, &simplified_range)) {
      to_add.push_back({simplified_expr, simplified_range});
      to_remove.push_back(expr);
    }
  }
  for (const auto& expr : to_remove) {
    map.EraseConstraint(expr);
  }
  for (const auto& [expr, range] : to_add) {
    map.AddConstraint(expr, range);
  }
  return !to_add.empty();
}

std::tuple<AffineExpr, int64_t, AffineExpr> AffineExprSimplifier::SplitSumByGcd(
    AffineExpr sum) {
  std::optional<int64_t> multiplier_gcd = std::nullopt;
  AffineExpr no_multiplier = zero_;
  VisitSummands(sum, [&](AffineExpr expr) {
    if (auto multiplier = GetConstantRhs(expr, AffineExprKind::Mul)) {
      if (multiplier_gcd.has_value()) {
        multiplier_gcd = std::gcd(*multiplier_gcd, *multiplier);
      } else {
        multiplier_gcd = *multiplier;
      }
    }
  });

  // If nothing had a multiplier, or the GCD was 1, there's nothing to split.
  if (multiplier_gcd.value_or(1) == 1) {
    return {zero_, 1, sum};
  }

  auto scaled = MapSummands(sum, [&](AffineExpr expr) {
    if (auto multiplier = GetConstantRhs(expr, AffineExprKind::Mul)) {
      // Rescale the multiplier.
      return GetLhs(expr) * (*multiplier / *multiplier_gcd);
    }
    // Extract the summand.
    no_multiplier = no_multiplier + expr;
    return zero_;
  });

  return {scaled, *multiplier_gcd, no_multiplier};
}

namespace {

struct UsedParameters {
  llvm::DenseSet<int64_t> dimension_ids;
  llvm::DenseSet<int64_t> symbol_ids;
};

void GetUsedParametersImpl(const AffineExpr& expr,
                           UsedParameters& used_parameters) {
  if (auto dim_expr = mlir::dyn_cast<AffineDimExpr>(expr)) {
    used_parameters.dimension_ids.insert(dim_expr.getPosition());
    return;
  }
  if (auto symbol_expr = mlir::dyn_cast<AffineSymbolExpr>(expr)) {
    used_parameters.symbol_ids.insert(symbol_expr.getPosition());
    return;
  }
  if (auto binary_expr = mlir::dyn_cast<AffineBinaryOpExpr>(expr)) {
    GetUsedParametersImpl(binary_expr.getLHS(), used_parameters);
    GetUsedParametersImpl(binary_expr.getRHS(), used_parameters);
  }
}

// Returns IDs of dimensions and symbols that participate in AffineExpr.
UsedParameters GetUsedParameters(const mlir::AffineExpr& expr) {
  UsedParameters used_parameters;
  GetUsedParametersImpl(expr, used_parameters);
  return used_parameters;
}

bool IsFunctionOfUnusedVarsOnly(const UsedParameters& used_parameters,
                                const SmallBitVector& unused_dims_bit_vector,
                                const SmallBitVector& unused_symbols_bit_vector,
                                bool removing_dims, bool removing_symbols) {
  if (!used_parameters.dimension_ids.empty() && !removing_dims) {
    return false;
  }
  if (!used_parameters.symbol_ids.empty() && !removing_symbols) {
    return false;
  }

  for (int64_t dim_id : used_parameters.dimension_ids) {
    if (!unused_dims_bit_vector[dim_id]) {
      return false;
    }
  }
  for (int64_t symbol_id : used_parameters.symbol_ids) {
    if (!unused_symbols_bit_vector[symbol_id]) {
      return false;
    }
  }
  return true;
}

struct UnusedVariables {
  SmallBitVector unused_dims;
  SmallBitVector unused_symbols;
  SmallVector<AffineExpr> constraints_with_unused_vars_only;
};

// Detects unused dimensions and symbols in the inde
UnusedVariables DetectUnusedVariables(const IndexingMap& indexing_map,
                                      bool removing_dims,
                                      bool removing_symbols) {
  AffineMap affine_map = indexing_map.GetAffineMap();

  UnusedVariables unused_vars;
  // Find unused dimensions and symbols in the affine_map.
  unused_vars.unused_dims = mlir::getUnusedDimsBitVector({affine_map});
  unused_vars.unused_symbols = mlir::getUnusedSymbolsBitVector({affine_map});

  // Check if the symbols that are unused in `affine_map` are also unused in
  // expressions.
  SmallVector<std::pair<AffineExpr, UsedParameters>, 2>
      unused_constraints_candidates;
  for (const auto& [expr, range] : indexing_map.GetConstraints()) {
    UsedParameters used_parameters = GetUsedParameters(expr);
    // If the expression uses only symbols that are unused in `affine_map`, then
    // we can remove it (because we will remove the symbols as well). Note that
    // the same is not true for dimensions, because of the existence of the
    // `RemoveUnusedSymbols` function.
    if (IsFunctionOfUnusedVarsOnly(used_parameters, unused_vars.unused_dims,
                                   unused_vars.unused_symbols, removing_dims,
                                   removing_symbols)) {
      unused_constraints_candidates.push_back({expr, used_parameters});
      continue;
    }
    // Otherwise, we need to mark all dims and symbols of these expr as "used".
    for (int64_t dim_id : used_parameters.dimension_ids) {
      unused_vars.unused_dims[dim_id] = false;
    }
    for (int64_t symbol_id : used_parameters.symbol_ids) {
      unused_vars.unused_symbols[symbol_id] = false;
    }
  }
  for (const auto& [expr, used_parameters] : unused_constraints_candidates) {
    if (IsFunctionOfUnusedVarsOnly(used_parameters, unused_vars.unused_dims,
                                   unused_vars.unused_symbols, removing_dims,
                                   removing_symbols)) {
      unused_vars.constraints_with_unused_vars_only.push_back(expr);
    }
  }
  return unused_vars;
}

SmallBitVector ConcatenateBitVectors(const SmallBitVector& lhs,
                                     const SmallBitVector& rhs) {
  SmallBitVector concat(lhs.size() + rhs.size(), false);
  int id = 0;
  for (int i = 0; i < lhs.size(); ++i, ++id) {
    concat[id] = lhs[i];
  }
  for (int i = 0; i < rhs.size(); ++i, ++id) {
    concat[id] = rhs[i];
  }
  return concat;
}

}  // namespace

bool IndexingMap::CompressVars(const llvm::SmallBitVector& unused_dims,
                               const llvm::SmallBitVector& unused_symbols) {
  MLIRContext* mlir_context = GetMLIRContext();

  bool num_dims_changed = unused_dims.count() > 0;
  bool num_symbols_changed = unused_symbols.count() > 0;
  if (!num_dims_changed && !num_symbols_changed) {
    return false;
  }

  unsigned num_dims_before = GetDimensionCount();
  unsigned num_symbols_before = GetSymbolCount();

  // Compress DimVars.
  SmallVector<AffineExpr, 2> dim_replacements;
  if (num_dims_changed) {
    affine_map_ = mlir::compressDims(affine_map_, unused_dims);
    std::vector<IndexingMap::Variable> compressed_dim_vars;
    dim_replacements = SmallVector<AffineExpr, 2>(
        num_dims_before, getAffineConstantExpr(0, mlir_context));
    int64_t used_dims_count = 0;
    for (int i = 0; i < unused_dims.size(); ++i) {
      if (!unused_dims[i]) {
        compressed_dim_vars.push_back(dim_vars_[i]);
        dim_replacements[i] = getAffineDimExpr(used_dims_count++, mlir_context);
      }
    }
    dim_vars_ = std::move(compressed_dim_vars);
  }

  // Compress RangeVars and RTVars.
  SmallVector<AffineExpr, 2> symbol_replacements;
  if (num_symbols_changed) {
    affine_map_ = mlir::compressSymbols(affine_map_, unused_symbols);
    symbol_replacements = SmallVector<AffineExpr, 2>(
        num_symbols_before, getAffineConstantExpr(0, mlir_context));
    std::vector<IndexingMap::Variable> compressed_range_vars;
    std::vector<IndexingMap::Variable> compressed_rt_vars;
    MLIRContext* mlir_context = GetMLIRContext();
    int64_t used_symbols_count = 0;
    auto range_vars_count = range_vars_.size();
    for (int i = 0; i < unused_symbols.size(); ++i) {
      if (!unused_symbols[i]) {
        if (i < range_vars_count) {
          compressed_range_vars.push_back(range_vars_[i]);
        } else {
          compressed_rt_vars.push_back(rt_vars_[i - range_vars_count]);
        }
        symbol_replacements[i] =
            getAffineSymbolExpr(used_symbols_count++, mlir_context);
      }
    }
    range_vars_ = std::move(compressed_range_vars);
    rt_vars_ = std::move(compressed_rt_vars);
  }

  // Remove constraints.
  std::vector<AffineExpr> to_remove;
  std::vector<std::pair<AffineExpr, Interval>> to_add;
  for (const auto& [expr, range] : constraints_) {
    auto updated_expr =
        expr.replaceDimsAndSymbols(dim_replacements, symbol_replacements);
    if (updated_expr == expr) {
      continue;
    }
    to_add.push_back({updated_expr, range});
    to_remove.push_back(expr);
  }
  for (const auto& expr : to_remove) {
    constraints_.erase(expr);
  }
  for (const auto& [expr, range] : to_add) {
    AddConstraint(expr, range);
  }
  return true;
}

SmallBitVector IndexingMap::RemoveUnusedSymbols() {
  if (IsUndefined() || GetSymbolCount() == 0) {
    return {};
  }
  UnusedVariables unused_vars = DetectUnusedVariables(
      *this, /*removing_dims=*/false, /*removing_symbols=*/true);
  for (AffineExpr expr : unused_vars.constraints_with_unused_vars_only) {
    constraints_.erase(expr);
  }
  if (!CompressVars(/*unused_dims=*/{}, unused_vars.unused_symbols)) {
    return {};
  }
  return std::move(unused_vars).unused_symbols;
}

void IndexingMap::ResetToKnownEmpty() {
  auto zero = getAffineConstantExpr(0, GetMLIRContext());
  affine_map_ = AffineMap::get(
      affine_map_.getNumDims(), affine_map_.getNumSymbols(),
      llvm::SmallVector<AffineExpr>(affine_map_.getNumResults(), zero),
      GetMLIRContext());
  for (auto& dim_var : dim_vars_) {
    dim_var.bounds = Interval{0, -1};
  }
  for (auto& range_var : range_vars_) {
    range_var.bounds = Interval{0, -1};
  }
  constraints_.clear();
  is_known_empty_ = true;
}

bool IndexingMap::VerifyVariableIntervals() {
  // TODO: Check if the variable names are unique.
  return llvm::all_of(dim_vars_,
                      [](const IndexingMap::Variable& dim_var) {
                        return dim_var.bounds.IsFeasible();
                      }) &&
         llvm::all_of(range_vars_,
                      [](const IndexingMap::Variable& range_var) {
                        return range_var.bounds.IsFeasible();
                      }) &&
         llvm::all_of(rt_vars_, [](const IndexingMap::Variable& rt_var) {
           return rt_var.bounds.IsFeasible();
         });
}

bool IndexingMap::VerifyConstraintIntervals() {
  return llvm::all_of(constraints_, [](const auto& constraint) {
    return constraint.second.IsFeasible();
  });
}

SmallBitVector IndexingMap::RemoveUnusedVars() {
  if (IsUndefined()) {
    return {};
  }

  UnusedVariables unused_vars = DetectUnusedVariables(
      *this, /*removing_dims=*/true, /*removing_symbols=*/true);
  for (AffineExpr expr : unused_vars.constraints_with_unused_vars_only) {
    constraints_.erase(expr);
  }
  if (!CompressVars(unused_vars.unused_dims, unused_vars.unused_symbols)) {
    return {};
  }
  return ConcatenateBitVectors(unused_vars.unused_dims,
                               unused_vars.unused_symbols);
}

bool IndexingMap::MergeModConstraints() {
  RangeEvaluator range_evaluator(*this, GetMLIRContext(),
                                 /*use_constraints=*/false);
  bool did_simplify = false;

  // Group constraints by LHS.
  llvm::DenseMap<AffineExpr, llvm::SmallVector<AffineBinaryOpExpr, 2>>
      grouped_constraints;
  for (const auto& [expr, _] : constraints_) {
    if (expr.getKind() != AffineExprKind::Mod) continue;
    auto binop = mlir::cast<AffineBinaryOpExpr>(expr);
    grouped_constraints[binop.getLHS()].push_back(binop);
  }

  // Merge constraints of type MOD.
  // (X mod 3 == 0) & (X mod 2 == 0) => (X mod 6 == 0)
  for (const auto& [lhs, binops] : grouped_constraints) {
    llvm::DenseMap<int64_t, llvm::SmallVector<AffineBinaryOpExpr, 2>>
        mod_groups;
    for (const auto& binop : binops) {
      Interval mod_result = constraints_[binop];
      if (mod_result.IsPoint()) {
        mod_groups[mod_result.lower].push_back(binop);
      }
    }
    if (mod_groups.empty()) continue;

    // Update domain for dimensions and symbols only.
    Interval* interval_to_update = nullptr;
    if (lhs.getKind() == AffineExprKind::DimId) {
      interval_to_update = &GetMutableDimensionBound(
          mlir::cast<AffineDimExpr>(lhs).getPosition());
    } else if (lhs.getKind() == AffineExprKind::SymbolId) {
      interval_to_update = &GetMutableSymbolBound(
          mlir::cast<AffineSymbolExpr>(lhs).getPosition());
    }
    for (const auto& [res, ops] : mod_groups) {
      // Calculate least common multiple for the divisors.
      int64_t div = 1;
      for (const auto& op : ops) {
        int64_t rhs_value =
            range_evaluator.ComputeExpressionRange(op.getRHS()).lower;
        div = std::lcm(div, rhs_value);
      }
      // Replace multiple constraints with a merged one.
      if (ops.size() > 1) {
        for (const auto& op : ops) {
          constraints_.erase(op);
        }
        constraints_[lhs % div] = Interval{res, res};
        did_simplify = true;
      }
      // Update dimension and symbol bounds.
      // TODO(b/347240603): If there are 2 constraints for the same dimension,
      // but we cannot merge them, then the final interval of the dimension may
      // depend on the order of iteration of mod_groups, and it may change
      // multiple times if we call MergeModConstraints() repeatedly, until
      // reaching a "sharp limit".
      if (interval_to_update != nullptr) {
        Interval old = *interval_to_update;
        int64_t l = (interval_to_update->lower / div) * div + res;
        interval_to_update->lower =
            l >= interval_to_update->lower ? l : l + div;
        int64_t h = (interval_to_update->upper / div) * div + res;
        interval_to_update->upper =
            h <= interval_to_update->upper ? h : h - div;
        if (*interval_to_update != old) {
          did_simplify = true;
        }
      }
    }
  }

  return did_simplify;
}

IndexingMap ComposeIndexingMaps(const IndexingMap& first,
                                const IndexingMap& second) {
  if (second.IsUndefined() || first.IsUndefined()) {
    return IndexingMap::GetUndefined();
  }
  MLIRContext* mlir_context = first.GetMLIRContext();
  AffineMap producer_affine_map = second.GetAffineMap();
  AffineMap composed_map = producer_affine_map.compose(first.GetAffineMap());

  // The symbols in the composed map, i.e. combined
  // producer_map.compose(consumer_map) are packed as
  // [range_vars(second)|rt_vars(second)|range_vars(first)|rt_vars(first)].
  std::vector<IndexingMap::Variable> combined_range_vars;
  combined_range_vars.reserve(second.GetRangeVarsCount() +
                              first.GetRangeVarsCount());
  for (const IndexingMap::Variable& range_var :
       llvm::concat<const IndexingMap::Variable>(second.GetRangeVars(),
                                                 first.GetRangeVars())) {
    combined_range_vars.push_back(range_var);
  }
  std::vector<IndexingMap::Variable> combined_rt_vars;
  combined_rt_vars.reserve(second.GetRTVarsCount() + first.GetRTVarsCount());
  for (const IndexingMap::Variable& rt_var :
       llvm::concat<const IndexingMap::Variable>(second.GetRTVars(),
                                                 first.GetRTVars())) {
    combined_rt_vars.push_back(rt_var);
  }
  // The symbols in the composed map have to be permuted to keep the invariant
  // that range_vars go before rt_vars in the composed affine map symbols list.
  SmallVector<AffineExpr, 4> symbol_replacements =
      GetComposedSymbolsPermutationToCorrectOrder(first, second);
  if (!symbol_replacements.empty()) {
    composed_map = composed_map.replaceDimsAndSymbols(
        /*dimReplacements=*/{}, symbol_replacements, composed_map.getNumDims(),
        composed_map.getNumSymbols());
  }
  IndexingMap composed_indexing_map(composed_map, first.GetDimVars(),
                                    std::move(combined_range_vars),
                                    std::move(combined_rt_vars));

  // Add constraints that are already present in the producer_map. We have to
  // compute consumer_map(producer_constraints). To keep all symbols and
  // dimension IDs the same as in the `composed_indexing_map.affine_map`, we
  // create an AffineMap
  // (dims of producer_affine_map)[symbols_of_producer_affine_map] =
  //   (constraint_1, ..., constraint_N) and then compose.
  std::vector<AffineExpr> constraints;
  std::vector<Interval> constraints_ranges;
  for (const auto& [expr, range] : second.GetConstraints()) {
    constraints.push_back(expr);
    constraints_ranges.push_back(range);
  }
  auto constraints_map = AffineMap::get(producer_affine_map.getNumDims(),
                                        producer_affine_map.getNumSymbols(),
                                        constraints, mlir_context);
  auto remapped_constraints =
      constraints_map.compose(first.GetAffineMap())
          .replaceDimsAndSymbols(/*dimReplacements=*/{}, symbol_replacements,
                                 composed_indexing_map.GetDimensionCount(),
                                 composed_indexing_map.GetSymbolCount());
  for (const auto& [expr, range] :
       llvm::zip(remapped_constraints.getResults(), constraints_ranges)) {
    composed_indexing_map.AddConstraint(expr, range);
  }
  // Remap symbol ids and add constraints that are already present in the
  // consumer_map.
  SmallVector<AffineExpr, 4> first_map_symbols_to_composed_symbols =
      MapSymbolsToComposedSymbolsList(first, composed_indexing_map);
  for (const auto& [expr, range] : first.GetConstraints()) {
    composed_indexing_map.AddConstraint(
        expr.replaceSymbols(first_map_symbols_to_composed_symbols), range);
  }
  // Add constraints for consumer's codomain w.r.t. producer's domain.
  for (auto [index, expr] :
       llvm::enumerate(first.GetAffineMap().getResults())) {
    Interval producer_dim_range =
        second.GetDimensionBound(static_cast<int64_t>(index));
    composed_indexing_map.AddConstraint(
        expr.replaceSymbols(first_map_symbols_to_composed_symbols),
        producer_dim_range);
  }
  return composed_indexing_map;
}

bool IndexingMap::RescaleSymbols() {
  MergeModConstraints();

  llvm::DenseSet<AffineExpr> to_delete;
  llvm::DenseMap<AffineExpr, AffineExpr> to_replace;

  for (const auto& [expr, range] : constraints_) {
    if (range.lower != range.upper) {
      continue;
    }
    auto shift_value = range.lower;

    if (expr.getKind() != AffineExprKind::Mod) {
      continue;
    }
    auto mod_expr = mlir::cast<AffineBinaryOpExpr>(expr);

    auto constant_expr = mlir::dyn_cast<AffineConstantExpr>(mod_expr.getRHS());
    if (!constant_expr) {
      continue;
    }

    // We don't rescale mod expressions with non-positive divisors.
    if (constant_expr.getValue() <= 0) {
      continue;
    }
    auto scaling_factor = constant_expr.getValue();

    if (mod_expr.getLHS().getKind() != AffineExprKind::SymbolId) {
      continue;
    }
    auto symbol_expr = mlir::cast<AffineSymbolExpr>(mod_expr.getLHS());

    // In case there are two mod constraints which were not merged, we only
    // support rescaling by one.
    // TODO(b/347240603): The result shouldn't depend on the hashmap's iteration
    // order.
    if (to_replace.contains(symbol_expr)) {
      continue;
    }

    to_replace[symbol_expr] = constant_expr * symbol_expr + shift_value;
    to_delete.insert(expr);

    affine_map_ = affine_map_.replace(
        symbol_expr, constant_expr * symbol_expr + shift_value,
        affine_map_.getNumDims(), affine_map_.getNumSymbols());

    auto& symbol_range = range_vars_[symbol_expr.getPosition()].bounds;
    symbol_range.lower = (symbol_range.lower - shift_value) / scaling_factor;
    symbol_range.upper = (symbol_range.upper - shift_value) / scaling_factor;
  }

  llvm::DenseMap<mlir::AffineExpr, Interval> new_constraints;
  for (const auto& [expr, range] : constraints_) {
    if (!to_delete.contains(expr)) {
      new_constraints[expr.replace(to_replace)] = range;
    }
  }
  constraints_ = std::move(new_constraints);

  return !to_delete.empty();
}

bool IndexingMap::IsRangeVarSymbol(mlir::AffineSymbolExpr symbol) const {
  unsigned int position = symbol.getPosition();
  CHECK_LE(position, GetSymbolCount());
  return position < range_vars_.size();
}

bool IndexingMap::IsRTVarSymbol(mlir::AffineSymbolExpr symbol) const {
  unsigned int position = symbol.getPosition();
  CHECK_LE(position, GetSymbolCount());
  return position >= range_vars_.size();
}

IndexingMap IndexingMap::ConvertSymbolsToDimensions() const {
  int num_symbols = GetSymbolCount();
  if (IsUndefined() || IsKnownEmpty() || num_symbols == 0) {
    return *this;
  }
  int num_dims = GetDimensionCount();

  MLIRContext* mlir_context = GetMLIRContext();
  int64_t num_vars = num_dims + num_symbols;

  std::vector<IndexingMap::Variable> new_dim_vars;
  new_dim_vars.reserve(num_vars);

  // // Populate the existing dims.
  llvm::append_range(new_dim_vars, GetDimVars());

  // Capture the existing symbols as dims.
  SmallVector<AffineExpr> syms_replacements;
  int64_t symbol_id = num_dims;
  for (const IndexingMap::Variable& var :
       llvm::concat<const IndexingMap::Variable>(range_vars_, rt_vars_)) {
    syms_replacements.push_back(getAffineDimExpr(symbol_id++, mlir_context));
    new_dim_vars.push_back(IndexingMap::Variable{var.bounds});
  }

  // Update constraints.
  SmallVector<std::pair<AffineExpr, Interval>, 4> new_constraints;
  for (const auto& [expr, range] : constraints_) {
    new_constraints.push_back(
        std::make_pair(expr.replaceSymbols(syms_replacements), range));
  }

  AffineMap canonical_map =
      affine_map_.replaceDimsAndSymbols({}, syms_replacements, num_vars, 0);
  IndexingMap new_indexing_map(canonical_map, new_dim_vars, /*range_vars=*/{},
                               /*rt_vars=*/{}, new_constraints);
  return new_indexing_map;
}

IndexingMap ConvertRangeVariablesToDimensions(
    const IndexingMap& map, ArrayRef<int64_t> range_var_indices) {
  CHECK(std::is_sorted(range_var_indices.begin(), range_var_indices.end()));
  auto* mlir_context = map.GetMLIRContext();

  AffineMap affine_map = map.GetAffineMap();
  // Update the affine map and the variables.
  std::vector<IndexingMap::Variable> dims = map.GetDimVars();
  std::vector<IndexingMap::Variable> range_vars;
  SmallVector<AffineExpr, 4> symbol_replacements;
  symbol_replacements.reserve(affine_map.getNumSymbols());
  int64_t updated_count = 0;
  int64_t range_var_indices_count = range_var_indices.size();
  for (int i = 0; i < map.GetRangeVarsCount(); ++i) {
    auto range_var = map.GetRangeVar(i);
    if (updated_count < range_var_indices_count &&
        i == range_var_indices[updated_count]) {
      symbol_replacements.push_back(getAffineDimExpr(
          affine_map.getNumDims() + updated_count, mlir_context));
      dims.push_back(range_var);
      updated_count++;
    } else {
      symbol_replacements.push_back(
          getAffineSymbolExpr(i - updated_count, mlir_context));
      range_vars.push_back(range_var);
    }
  }
  CHECK_EQ(updated_count, range_var_indices_count)
      << "Not all replacements were used";
  for (int i = 0; i < map.GetRTVarsCount(); ++i) {
    symbol_replacements.push_back(getAffineSymbolExpr(
        map.GetRangeVarsCount() - range_var_indices_count + i, mlir_context));
  }
  CHECK_EQ(symbol_replacements.size(), affine_map.getNumSymbols())
      << "All symbols must be updated";
  AffineMap converted_affine_map = affine_map.replaceDimsAndSymbols(
      /*dimReplacements=*/{}, symbol_replacements,
      /*numResultDims=*/affine_map.getNumDims() + range_var_indices_count,
      /*numResultSyms=*/affine_map.getNumSymbols() - range_var_indices_count);

  // Update the constraints.
  std::vector<std::pair<AffineExpr, Interval>> constraints;
  constraints.reserve(map.GetConstraintsCount());
  for (auto constraint : map.GetConstraints()) {
    constraints.push_back({constraint.first.replaceSymbols(symbol_replacements),
                           constraint.second});
  }
  return IndexingMap{converted_affine_map, std::move(dims),
                     std::move(range_vars), map.GetRTVars(), constraints};
}

}  // namespace xla
