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
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
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
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"
#include "xla/hlo/analysis/symbolic_map_converter.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {
namespace {

using llvm::ArrayRef;
using llvm::SmallBitVector;
using llvm::SmallVector;
using mlir::AffineExpr;
using mlir::AffineMap;
using mlir::MLIRContext;

SymbolicExpr GetLhs(SymbolicExpr e) { return e.GetLHS(); }

SymbolicExpr GetRhs(SymbolicExpr e) { return e.GetRHS(); }

// Rewrites summands in arbitrarily nested sums (e.g, ((a+b)+c)) by applying
// `fn` to each one. In the example, the result is fn(a)+fn(b)+fn(c).
template <typename Fn>
SymbolicExpr MapSummands(SymbolicExpr expr, const Fn& fn) {
  if (expr.GetType() == SymbolicExprType::kAdd) {
    auto lhs = MapSummands(expr.GetLHS(), fn);
    auto rhs = MapSummands(expr.GetRHS(), fn);
    if (lhs == expr.GetLHS() && rhs == expr.GetRHS()) {
      return expr;
    }
    return lhs + rhs;
  }
  return fn(expr);
}

// Calls `visit` for each summand in an arbitrarily nested sum.
template <typename Fn>
void VisitSummands(SymbolicExpr expr, const Fn& visit) {
  if (expr.GetType() == SymbolicExprType::kAdd) {
    VisitSummands(GetLhs(expr), visit);
    VisitSummands(GetRhs(expr), visit);
  } else {
    visit(expr);
  }
}

class SymbolicExprSimplifier {
 public:
  explicit SymbolicExprSimplifier(
      RangeEvaluator* range_evaluator,
      IndexingMap::SimplifyPointDimensions simplify_point_dimensions =
          IndexingMap::SimplifyPointDimensions::kReplace)
      : range_evaluator_(range_evaluator),
        zero_(CreateSymbolicConstant(0, range_evaluator_->GetMLIRContext())),
        simplify_point_dimensions_(simplify_point_dimensions) {}

  // Simplifies the map as much as possible.
  SymbolicMap Simplify(SymbolicMap symbolic_map);

  SymbolicExpr Simplify(SymbolicExpr expr);

  // Performs SymbolicExpr simplification for all constraints.
  // Returns true if simplification was performed.
  bool SimplifyConstraintExprs(IndexingMap& map);

  // Performs range simplification for all constraints.
  // Returns true if simplification was performed.
  bool SimplifyConstraintRanges(IndexingMap& map);

 private:
  std::optional<int64_t> GetConstantRhs(SymbolicExpr expr,
                                        SymbolicExprType type);
  std::pair<SymbolicExpr, int64_t> ExtractMultiplier(SymbolicExpr expr) {
    if (auto mul = GetConstantRhs(expr, SymbolicExprType::kMul)) {
      return {GetLhs(expr), *mul};
    }
    return {expr, 1};
  }

  // Simplifier for mod.
  // - Rewrites (a * 100 + ...) % 100 to (...) % 100
  // - Rewrites a % b to a if a is known to be less than b.
  SymbolicExpr RewriteMod(SymbolicExpr mod);

  // Simplifier for floordiv. Uses all the rules defined below.
  SymbolicExpr RewriteFloorDiv(SymbolicExpr div);

  // Rewrites `(c % ab) // a` to `(c // a) % b`. Returns nullptr on mismatch.
  SymbolicExpr SimplifyModDiv(SymbolicExpr dividend, int64_t divisor);

  // Rewrites `a // b // c` to `a // (b * c)` if `c` is positive. Returns
  // nullptr on mismatch.
  SymbolicExpr SimplifyDivDiv(SymbolicExpr dividend, int64_t divisor);

  // Rewrites `a // b` where a may be a sum.
  SymbolicExpr SimplifySumDiv(SymbolicExpr dividend, int64_t divisor);

  // Simplifier for mul.
  // - Distributes multiplications with constants over sums.
  SymbolicExpr RewriteMul(SymbolicExpr mul);

  // Simplifier for sums.
  SymbolicExpr RewriteSum(SymbolicExpr sum);

  // Attempts to simplify the expression, but doesn't attempt to simplify the
  // result further.
  SymbolicExpr SimplifyOnce(SymbolicExpr expr);

  // Simplifies the expression using MLIR's simplifier, except for mods.
  SymbolicExpr SimplifyWithMlir(SymbolicExpr expr, int num_dims,
                                int num_symbols);

  bool SimplifyConstraintRangeOnce(SymbolicExpr* expr, Interval* range);
  bool SimplifyConstraintRange(SymbolicExpr* expr, Interval* range);
  bool SimplifyAddConstraint(SymbolicExpr* add, Interval* range);

  // Splits a nested sum into a * gcd + b.
  std::tuple<SymbolicExpr /*a*/, int64_t /*gcd*/, SymbolicExpr /*b*/>
  SplitSumByGcd(SymbolicExpr sum);

  RangeEvaluator* range_evaluator_;
  SymbolicExpr zero_;
  IndexingMap::SimplifyPointDimensions simplify_point_dimensions_;
};

SymbolicExpr SymbolicExprSimplifier::RewriteMod(SymbolicExpr mod) {
  auto rhs = range_evaluator_->ComputeExpressionRange(mod.GetRHS());

  // The logic below assumes we have a constant RHS.
  if (!rhs.IsPoint()) {
    return mod;
  }
  int64_t m = rhs.lower;
  // Can only happen in cases where it doesn't matter, return 0.
  if (m == 0) {
    return zero_;
  }

  auto lhs_simplified = SimplifyOnce(mod.GetLHS());
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
  if (auto mul = GetConstantRhs(lhs_simplified, SymbolicExprType::kMul);
      mul && *mul > 0 && (m % *mul == 0)) {
    return (GetLhs(lhs_simplified) % (m / *mul)) * *mul;
  }

  int64_t extracted_constant = 0;
  auto new_lhs = MapSummands(lhs_simplified, [&](SymbolicExpr expr) {
    if (expr.GetType() == SymbolicExprType::kConstant) {
      extracted_constant += expr.GetValue();
      return zero_;
    }
    if (auto multiplier = GetConstantRhs(expr, SymbolicExprType::kMul);
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
        new_lhs = MapSummands(multiplied, [&, multiplier_gcd = multiplier_gcd](
                                              SymbolicExpr expr) {
          return expr * (multiplier_gcd / multiplier_mod_gcd);
        });
      }
      return (new_lhs % (m / multiplier_mod_gcd)) * multiplier_mod_gcd;
    }
    if (Interval{0, multiplier_gcd - 1}.Contains(not_multiplied_range)) {
      // Remove everything that doesn't have a multiplier.
      new_lhs = multiplied * multiplier_gcd;
      return new_lhs % mod.GetRHS() + not_multiplied;
    }
  }

  return new_lhs == mod.GetLHS() ? mod : (new_lhs % m);
}

SymbolicExpr SymbolicExprSimplifier::SimplifyModDiv(SymbolicExpr dividend,
                                                    int64_t divisor) {
  if (auto mod = GetConstantRhs(dividend, SymbolicExprType::kMod);
      mod && (*mod % divisor == 0)) {
    return GetLhs(dividend).floorDiv(divisor) % (*mod / divisor);
  }
  return nullptr;
}

SymbolicExpr SymbolicExprSimplifier::SimplifyDivDiv(SymbolicExpr dividend,
                                                    int64_t divisor) {
  // The inner divisor here can be negative.
  if (auto inner_divisor =
          GetConstantRhs(dividend, SymbolicExprType::kFloorDiv)) {
    return GetLhs(dividend).floorDiv(divisor * *inner_divisor);
  }
  return nullptr;
}

SymbolicExpr SymbolicExprSimplifier::SimplifySumDiv(SymbolicExpr dividend,
                                                    int64_t divisor) {
  SymbolicExpr extracted = zero_;
  auto new_dividend = MapSummands(dividend, [&](SymbolicExpr expr) {
    if (auto multiplier = GetConstantRhs(expr, SymbolicExprType::kMul)) {
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
  VisitSummands(new_dividend, [&](SymbolicExpr summand) {
    if (auto divisor = GetConstantRhs(summand, SymbolicExprType::kFloorDiv)) {
      inner_divisor = divisor;
      ++num_inner_divisors;
    }
  });
  if (num_inner_divisors == 1) {
    new_dividend = MapSummands(new_dividend, [&](SymbolicExpr summand) {
      if (auto inner_divisor =
              GetConstantRhs(summand, SymbolicExprType::kFloorDiv)) {
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

SymbolicExpr SymbolicExprSimplifier::RewriteFloorDiv(SymbolicExpr div) {
  auto rhs_range = range_evaluator_->ComputeExpressionRange(div.GetRHS());
  auto lhs_simplified = SimplifyOnce(div.GetLHS());
  if (!rhs_range.IsPoint()) {
    return lhs_simplified.floorDiv(SimplifyOnce(div.GetRHS()));
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
  return lhs_simplified != div.GetLHS() ? lhs_simplified.floorDiv(d) : div;
}

SymbolicExpr SymbolicExprSimplifier::RewriteMul(SymbolicExpr mul) {
  auto rhs_range = range_evaluator_->ComputeExpressionRange(mul.GetRHS());

  // The logic below assumes we have a constant RHS.
  if (!rhs_range.IsPoint()) {
    return mul;
  }

  int64_t multiplier = rhs_range.lower;
  auto lhs = SimplifyOnce(mul.GetLHS());
  if (lhs.GetType() == SymbolicExprType::kAdd) {
    return MapSummands(
        lhs, [&](SymbolicExpr summand) { return summand * multiplier; });
  }

  if (multiplier == 1) {
    return lhs;
  }
  if (lhs == mul.GetLHS()) {
    return mul;
  }
  return lhs * multiplier;
}

std::optional<int64_t> SymbolicExprSimplifier::GetConstantRhs(
    SymbolicExpr expr, SymbolicExprType type) {
  if (expr.GetType() != type) {
    return std::nullopt;
  }
  auto bound = range_evaluator_->ComputeExpressionRange(expr.GetRHS());
  if (!bound.IsPoint()) {
    return std::nullopt;
  }
  return bound.lower;
}

SymbolicExpr SymbolicExprSimplifier::RewriteSum(SymbolicExpr sum) {
  // TODO(jreiffers): Split this up more.
  // Rewrite `(x % c) * d + (x // c) * (c * d)` to `x * d`. We have to do it
  // in this rather convoluted way because we distribute multiplications.
  SmallVector<std::pair<SymbolicExpr, int64_t /*multiplier*/>> mods;
  SmallVector<std::pair<SymbolicExpr, int64_t /*multiplier*/>> divs;
  llvm::SmallDenseMap<SymbolicExpr, int64_t /* multiplier */> summands;
  VisitSummands(sum, [&](SymbolicExpr expr) {
    SymbolicExpr simplified = SimplifyOnce(expr);
    auto [lhs, multiplier] = ExtractMultiplier(simplified);
    if (lhs.GetType() == SymbolicExprType::kMod) {
      mods.push_back({lhs, multiplier});
    } else if (lhs.GetType() == SymbolicExprType::kFloorDiv) {
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
      auto mod_c = GetConstantRhs(mod, SymbolicExprType::kMod);
      if (!mod_c) {
        continue;
      }

      // In many cases, we could just compare the LHSes of the mod and the
      // div, but if x is a floorDiv itself, we need to check a bit more
      // carefully:
      //    ((x // c0) % c1) * d + (x // (c0 * c1)) * (c1 * d)`
      // `x // (c0 * c1)` will be simplified, so we we may not even have
      // `c0 * c1` in the expression, if `x` contains a multiplier.
      SymbolicExpr simplified_mod = Simplify(GetLhs(mod).floorDiv(*mod_c));
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
      auto div_c = GetConstantRhs(div, SymbolicExprType::kFloorDiv);
      if (!div_c || *div_c < 0 || (div_mul % *div_c)) {
        continue;
      }

      int64_t b = div_mul / *div_c;
      auto x = GetLhs(div);
      VisitSummands(x, [&](SymbolicExpr summand) { summands[summand] += b; });
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

  SmallVector<SymbolicExpr, 4> expanded_summands;
  for (auto [expr, mul] : summands) {
    expanded_summands.push_back(expr * mul);
  }
  llvm::sort(expanded_summands,
             [](SymbolicExpr a, SymbolicExpr b) { return a < b; });
  SymbolicExpr result = zero_;
  for (auto expr : expanded_summands) {
    result = result + expr;
  }
  return result;
}

SymbolicExpr SymbolicExprSimplifier::SimplifyOnce(SymbolicExpr expr) {
  if (expr.GetType() == SymbolicExprType::kConstant) {
    return expr;
  }

  if (simplify_point_dimensions_ ==
      IndexingMap::SimplifyPointDimensions::kReplace) {
    auto bounds = range_evaluator_->ComputeExpressionRange(expr);
    if (bounds.IsPoint()) {
      return CreateSymbolicConstant(bounds.lower,
                                    range_evaluator_->GetMLIRContext());
    }
  }

  switch (expr.GetType()) {
    case SymbolicExprType::kMul:
      return RewriteMul(expr);
    case SymbolicExprType::kAdd:
      return RewriteSum(expr);
    case SymbolicExprType::kMod:
      return RewriteMod(expr);
    case SymbolicExprType::kFloorDiv:
      return RewriteFloorDiv(expr);
    default:
      return expr;
  }
}

SymbolicExpr SymbolicExprSimplifier::Simplify(SymbolicExpr expr) {
  while (true) {
    auto simplified = SimplifyOnce(expr);
    if (simplified == expr) {
      return expr;
    }
    expr = simplified;
  }
}

SymbolicMap SymbolicExprSimplifier::Simplify(SymbolicMap symbolic_map) {
  SmallVector<SymbolicExpr, 4> results;
  results.reserve(symbolic_map.GetNumResults());
  for (SymbolicExpr expr : symbolic_map.GetResults()) {
    results.push_back(Simplify(expr));
  }
  return SymbolicMap::Get(symbolic_map.GetContext(), symbolic_map.GetNumDims(),
                          symbolic_map.GetNumSymbols(), std::move(results));
}

bool SymbolicExprSimplifier::SimplifyAddConstraint(SymbolicExpr* add,
                                                   Interval* range) {
  if (add->GetType() != SymbolicExprType::kAdd) {
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
bool SymbolicExprSimplifier::SimplifyConstraintRangeOnce(SymbolicExpr* expr,
                                                         Interval* range) {
  switch (expr->GetType()) {
    case SymbolicExprType::kVariable:
      // do the trick with constant
    case SymbolicExprType::kConstant: {
      return false;
    }
    case SymbolicExprType::kAdd:
      return SimplifyAddConstraint(expr, range);
    default: {
      SymbolicExpr lhs = expr->GetLHS();
      CHECK(lhs);
      auto rhs_range = range_evaluator_->ComputeExpressionRange(expr->GetRHS());
      if (!rhs_range.IsPoint()) {
        return false;
      }
      int64_t rhs_cst = rhs_range.lower;
      switch (expr->GetType()) {
        case SymbolicExprType::kMul: {
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
        case SymbolicExprType::kFloorDiv: {
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
bool SymbolicExprSimplifier::SimplifyConstraintRange(SymbolicExpr* expr,
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
SmallVector<SymbolicExpr, 4> GetComposedSymbolsPermutationToCorrectOrder(
    const IndexingMap& first, const IndexingMap& second, int64_t num_dims) {
  // No permutation is needed if the second map has no RTVars.
  if (second.GetRTVarsCount() == 0) {
    return {};
  }
  SmallVector<SymbolicExpr, 4> symbol_replacements;
  MLIRContext* context = first.GetMLIRContext();
  for (int id = 0; id < second.GetRangeVarsCount(); ++id) {
    symbol_replacements.push_back(CreateSymbolExpr(id, num_dims, context));
  }
  int64_t first_range_vars_count = first.GetRangeVarsCount();
  int64_t second_range_vars_count = second.GetRangeVarsCount();
  int64_t first_rt_vars_count = first.GetRTVarsCount();
  int64_t second_rt_vars_count = second.GetRTVarsCount();
  int64_t rt_vars_second_start =
      first_range_vars_count + second_range_vars_count;
  for (int64_t id = 0; id < second_rt_vars_count; ++id) {
    symbol_replacements.push_back(
        CreateSymbolExpr(rt_vars_second_start++, num_dims, context));
  }
  int64_t range_vars_first_start = second_range_vars_count;
  for (int64_t id = 0; id < first_range_vars_count; ++id) {
    symbol_replacements.push_back(
        CreateSymbolExpr(range_vars_first_start++, num_dims, context));
  }
  int64_t rt_vars_first_start =
      first_range_vars_count + second_range_vars_count + second_rt_vars_count;
  for (int64_t id = 0; id < first_rt_vars_count; ++id) {
    symbol_replacements.push_back(
        CreateSymbolExpr(rt_vars_first_start++, num_dims, context));
  }
  return symbol_replacements;
}

// Computes the symbols list mapping to go from
// [range_vars(map)|rt_vars(map)]
// to
// [range_vars(second)|range_vars(first)|rt_vars(second)|rt_vars(first)].
SmallVector<SymbolicExpr, 4> MapSymbolsToComposedSymbolsList(
    const IndexingMap& map, const IndexingMap& composed) {
  SmallVector<SymbolicExpr, 4> symbol_replacements;

  MLIRContext* context = map.GetMLIRContext();
  int64_t num_dims = map.GetSymbolicMap().GetNumDims();
  int64_t range_vars_start =
      composed.GetRangeVarsCount() - map.GetRangeVarsCount();
  for (int64_t id = 0; id < map.GetRangeVarsCount(); ++id) {
    symbol_replacements.push_back(
        CreateSymbolExpr(range_vars_start++, num_dims, context));
  }
  int64_t rt_vars_start = composed.GetSymbolCount() - map.GetRTVarsCount();
  for (int64_t id = 0; id < map.GetRTVarsCount(); ++id) {
    symbol_replacements.push_back(
        CreateSymbolExpr(rt_vars_start++, num_dims, context));
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

// TODO: b/446858351 - Remove this constructor after migrating all users to the
// symbolic map constructor.
IndexingMap::IndexingMap(
    AffineMap affine_map, std::vector<IndexingMap::Variable> dimensions,
    std::vector<IndexingMap::Variable> range_vars,
    std::vector<IndexingMap::Variable> rt_vars,
    absl::Span<std::pair<AffineExpr, Interval> const> constraints)
    : affine_map_(affine_map),
      dim_vars_(std::move(dimensions)),
      range_vars_(std::move(range_vars)),
      rt_vars_(std::move(rt_vars)) {
  symbolic_map_ = AffineMapToSymbolicMap(affine_map);
  for (const auto& [expr, range] : constraints) {
    AddConstraint(expr, range);
  }
  if (!VerifyVariableIntervals()) {
    ResetToKnownEmpty();
  }
}

// TODO (b/446858351): Remove this constructor after migrating all users to the
// symbolic map constructor.
IndexingMap::IndexingMap(
    AffineMap affine_map, std::vector<IndexingMap::Variable> dimensions,
    std::vector<IndexingMap::Variable> range_vars,
    std::vector<IndexingMap::Variable> rt_vars,
    const llvm::MapVector<AffineExpr, Interval>& constraints)
    : affine_map_(affine_map),
      dim_vars_(std::move(dimensions)),
      range_vars_(std::move(range_vars)),
      rt_vars_(std::move(rt_vars)) {
  symbolic_map_ = AffineMapToSymbolicMap(affine_map);
  constraints_ = ConvertAffineConstraintsToSymbolicConstraints(
      constraints, affine_map.getNumDims());
  if (!VerifyVariableIntervals() || !VerifyConstraintIntervals()) {
    ResetToKnownEmpty();
  }
}

IndexingMap::IndexingMap(
    SymbolicMap symbolic_map, std::vector<IndexingMap::Variable> dimensions,
    std::vector<IndexingMap::Variable> range_vars,
    std::vector<IndexingMap::Variable> rt_vars,
    absl::Span<std::pair<SymbolicExpr, Interval> const> constraints)
    : symbolic_map_(symbolic_map),
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
    SymbolicMap symbolic_map, std::vector<IndexingMap::Variable> dimensions,
    std::vector<IndexingMap::Variable> range_vars,
    std::vector<IndexingMap::Variable> rt_vars,
    const llvm::MapVector<SymbolicExpr, Interval>& constraints)
    : symbolic_map_(symbolic_map),
      dim_vars_(std::move(dimensions)),
      range_vars_(std::move(range_vars)),
      rt_vars_(std::move(rt_vars)),
      constraints_(constraints) {
  if (!VerifyVariableIntervals() || !VerifyConstraintIntervals()) {
    ResetToKnownEmpty();
    return;
  }
}

// TODO (b/446858351): Remove this constructor once all the users are migrated
// to the symbolic map constructor.
IndexingMap IndexingMap::FromTensorSizes(
    AffineMap affine_map, absl::Span<const int64_t> dim_upper_bounds,
    absl::Span<const int64_t> symbol_upper_bounds) {
  return IndexingMap{affine_map, DimVarsFromTensorSizes(dim_upper_bounds),
                     RangeVarsFromTensorSizes(symbol_upper_bounds),
                     /*rt_vars=*/{}};
}

IndexingMap IndexingMap::FromTensorSizes(
    SymbolicMap symbolic_map, absl::Span<const int64_t> dim_upper_bounds,
    absl::Span<const int64_t> symbol_upper_bounds) {
  return IndexingMap{symbolic_map, DimVarsFromTensorSizes(dim_upper_bounds),
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
  bounds.reserve(symbolic_map_.GetNumDims());
  for (const auto& dim : dim_vars_) {
    bounds.push_back(dim.bounds);
  }
  return bounds;
}

const Interval& IndexingMap::GetSymbolBound(int64_t symbol_id) const {
  // Because symbolic map symbols are packed like [range_vars, rt_vars],
  // we have to pick the correct bounds.
  int64_t range_var_count = GetRangeVarsCount();
  return symbol_id < range_var_count
             ? range_vars_[symbol_id].bounds
             : rt_vars_[symbol_id - range_var_count].bounds;
}

Interval& IndexingMap::GetMutableSymbolBound(int64_t symbol_id) {
  // Because symbolic map symbols are packed like [range_vars, rt_vars],
  // we have to pick the correct bounds.
  int64_t range_var_count = GetRangeVarsCount();
  return symbol_id < range_var_count
             ? range_vars_[symbol_id].bounds
             : rt_vars_[symbol_id - range_var_count].bounds;
}

std::vector<Interval> IndexingMap::GetSymbolBounds() const {
  std::vector<Interval> bounds;
  bounds.reserve(symbolic_map_.GetNumSymbols());
  for (const auto& range_var : range_vars_) {
    bounds.push_back(range_var.bounds);
  }
  for (const auto& rt_var : rt_vars_) {
    bounds.push_back(rt_var.bounds);
  }
  return bounds;
}

// TODO: b/446856820 - Remove this function once all the users are migrated to
// the symbolic map constructor.
llvm::MapVector<mlir::AffineExpr, Interval> IndexingMap::GetConstraints()
    const {
  llvm::MapVector<mlir::AffineExpr, Interval> affine_constraints;
  for (const auto& [expr, range] : constraints_) {
    affine_constraints[SymbolicExprToAffineExpr(
        expr, GetMLIRContext(), symbolic_map_.GetNumDims())] = range;
  }
  return affine_constraints;
}

// TODO: b/446856820 - Remove this function once all the users are migrated to
// the symbolic map constructor.
void IndexingMap::AddConstraint(mlir::AffineExpr expr, Interval range) {
  AddConstraint(AffineExprToSymbolicExpr(expr, GetDimensionCount()), range);
}

void IndexingMap::AddConstraint(SymbolicExpr expr, Interval range) {
  // Do not add the constraint if the domain is already empty.
  if (IsKnownEmpty()) {
    return;
  }
  // If the range is empty, reset the indexing map to the canonical empty form.
  if (!range.IsFeasible()) {
    ResetToKnownEmpty();
    return;
  }
  int64_t num_dims = GetDimVarsCount();
  if (IsDimension(expr, num_dims)) {
    Interval& current_range = GetMutableDimensionBound(expr.GetValue());
    current_range = current_range.Intersect(range);
    if (!current_range.IsFeasible()) {
      ResetToKnownEmpty();
    }
    return;
  }
  if (IsSymbol(expr, num_dims)) {
    Interval& current_range =
        GetMutableSymbolBound(GetSymbolIndex(expr, num_dims));
    current_range = current_range.Intersect(range);
    if (!current_range.IsFeasible()) {
      ResetToKnownEmpty();
    }
    return;
  }
  if (expr.GetType() == SymbolicExprType::kConstant) {
    if (!range.Contains(expr.GetValue())) {
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

void IndexingMap::EraseConstraint(SymbolicExpr expr) {
  constraints_.erase(expr);
}

// TODO: b/446856820  - Remove this function once all the users are migrated to
// the symbolic map getters.
bool IndexingMap::ConstraintsSatisfied(
    ArrayRef<mlir::AffineExpr> dim_const_exprs,
    ArrayRef<mlir::AffineExpr> symbol_const_exprs) const {
  return ConstraintsSatisfied(
      AffineExprsToSymbolicExprs(dim_const_exprs, symbolic_map_.GetNumDims()),
      AffineExprsToSymbolicExprs(symbol_const_exprs,
                                 symbolic_map_.GetNumDims()));
}

bool IndexingMap::ConstraintsSatisfied(
    ArrayRef<SymbolicExpr> dim_const_exprs,
    ArrayRef<SymbolicExpr> symbol_const_exprs) const {
  CHECK(dim_const_exprs.size() == symbolic_map_.GetNumDims());
  CHECK(symbol_const_exprs.size() == symbolic_map_.GetNumSymbols());
  if (IsKnownEmpty()) {
    return false;
  }
  for (auto& [expr, range] : constraints_) {
    int64_t expr_value =
        expr.ReplaceDimsAndSymbols(dim_const_exprs, symbol_const_exprs)
            .GetValue();
    if (expr_value < range.lower || expr_value > range.upper) {
      return false;
    }
  }
  return true;
}

SmallVector<int64_t, 4> IndexingMap::Evaluate(
    ArrayRef<SymbolicExpr> dim_const_exprs,
    ArrayRef<SymbolicExpr> symbol_const_exprs) const {
  CHECK(dim_const_exprs.size() == GetDimensionCount());
  CHECK(symbol_const_exprs.size() == GetSymbolCount());
  SymbolicMap eval = symbolic_map_.ReplaceDimsAndSymbols(
      dim_const_exprs, symbol_const_exprs, GetDimensionCount(),
      symbol_const_exprs.size());
  return eval.GetConstantResults();
}

bool IndexingMap::IsSymbolConstrained(int64_t symbol_id) const {
  int64_t num_dims = GetDimVarsCount();
  for (const auto& [expr, _] : constraints_) {
    bool result = false;
    expr.Walk([&](SymbolicExpr leaf) {
      if (IsSymbol(leaf, num_dims) &&
          (GetSymbolIndex(leaf, num_dims) == symbol_id)) {
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
  return IsAlwaysPositiveOrZero(
      AffineExprToSymbolicExpr(expr, indexing_map_.GetDimensionCount()));
}

bool RangeEvaluator::IsAlwaysPositiveOrZero(SymbolicExpr expr) {
  return ComputeExpressionRange(expr).lower >= 0;
}

bool RangeEvaluator::IsAlwaysNegativeOrZero(mlir::AffineExpr expr) {
  return IsAlwaysNegativeOrZero(
      AffineExprToSymbolicExpr(expr, indexing_map_.GetDimensionCount()));
}

bool RangeEvaluator::IsAlwaysNegativeOrZero(SymbolicExpr expr) {
  return ComputeExpressionRange(expr).upper <= 0;
}

Interval RangeEvaluator::ComputeExpressionRange(mlir::AffineExpr expr) {
  return ComputeExpressionRange(
      AffineExprToSymbolicExpr(expr, indexing_map_.GetDimensionCount()));
}

Interval RangeEvaluator::ComputeExpressionRange(SymbolicExpr expr) {
  switch (expr.GetType()) {
    case SymbolicExprType::kConstant: {
      int64_t value = expr.GetValue();
      return Interval{value, value};
    }
    case SymbolicExprType::kVariable: {
      int64_t num_dims = indexing_map_.GetDimensionCount();
      if (IsDimension(expr, num_dims)) {
        int64_t dim_id = GetDimensionIndex(expr, num_dims);
        return indexing_map_.GetDimensionBound(dim_id);
      }
      int64_t symbol_id = GetSymbolIndex(expr, num_dims);
      return indexing_map_.GetSymbolBound(symbol_id);
    }
    default:
      break;
  }
  CHECK(expr.IsBinaryOp());
  auto lhs = ComputeExpressionRange(expr.GetLHS());
  auto rhs = ComputeExpressionRange(expr.GetRHS());

  Interval result;
  switch (expr.GetType()) {
    case SymbolicExprType::kAdd:
      result = lhs + rhs;
      break;
    case SymbolicExprType::kMul:
      result = lhs * rhs;
      break;
    case SymbolicExprType::kMod: {
      CHECK(rhs.IsPoint()) << "RHS of mod must be a constant";
      int64_t m = rhs.lower;
      if (0 <= lhs.lower && lhs.upper < m) {
        result = lhs;
      } else {
        result = {0, m - 1};
      }
      break;
    }
    case SymbolicExprType::kFloorDiv: {
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
      LOG(FATAL) << "Unsupported expression type: "
                 << static_cast<int>(expr.GetType());
  }

  if (use_constraints_) {
    auto constraints_map = indexing_map_.GetSymbolicConstraints();
    auto constraint = constraints_map.find(expr);
    if (constraint != constraints_map.end()) {
      return result.Intersect(constraint->second);
    }
  }
  return result;
}

MLIRContext* IndexingMap::GetMLIRContext() const {
  return IsUndefined() ? nullptr : symbolic_map_.GetContext();
}

namespace {
bool EqualConstraints(const llvm::MapVector<SymbolicExpr, Interval>& lhs,
                      const llvm::MapVector<SymbolicExpr, Interval>& rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (const auto& [key, value] : lhs) {
    auto it = rhs.find(key);
    if (it == rhs.end() || it->second != value) {
      return false;
    }
  }

  return true;
}
}  // namespace

bool operator==(const IndexingMap& lhs, const IndexingMap& rhs) {
  return lhs.GetSymbolicMap() == rhs.GetSymbolicMap() &&
         lhs.GetDimVars() == rhs.GetDimVars() &&
         lhs.GetRangeVars() == rhs.GetRangeVars() &&
         lhs.GetRTVars() == rhs.GetRTVars() &&
         EqualConstraints(lhs.GetSymbolicConstraints(),
                          rhs.GetSymbolicConstraints());
}

IndexingMap operator*(const IndexingMap& lhs, const IndexingMap& rhs) {
  return ComposeIndexingMaps(lhs, rhs);
}

bool IndexingMap::Verify(std::ostream& out) const {
  if (IsUndefined()) {
    return true;
  }
  if (symbolic_map_.GetNumDims() != dim_vars_.size()) {
    out << absl::StrCat(
        "number of dim vars (", dim_vars_.size(),
        ") must match the number of dimensions in the symbolic map (",
        symbolic_map_.GetNumDims(), ")");
    return false;
  }
  if (symbolic_map_.GetNumSymbols() != range_vars_.size() + rt_vars_.size()) {
    out << absl::StrCat(
        "number of range (", range_vars_.size(), ") + runtime (",
        rt_vars_.size(),
        ") variables must match the number of symbols in the symbolic map (",
        symbolic_map_.GetNumSymbols(), ")");
    return false;
  }
  return true;
}

// Simplification of IndexingMap has two main parts.
// At first we optimized constraints to make the domain as small and simple as
// possible. And only then we simplify the symbolic_map, because its
// simplification relies on lower/upper bounds of dimensions and symbols.

// Constraint simplification is performed in two stages repeated until
// convergence.
//   1. Simplify symbolic expressions in all constraints.
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

  // Simplify symbolic_map using the optimized ranges.
  // Potentially, we can be smarter about recreating the range_evaluator.
  RangeEvaluator constraint_range_evaluator(*this, GetMLIRContext(),
                                            /*use_constraints=*/false);
  SymbolicExprSimplifier constraint_simplifier(&constraint_range_evaluator);
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
  SymbolicMap simplified_symbolic_map =
      SymbolicExprSimplifier(&range_evaluator, simplify_point_dimensions)
          .Simplify(symbolic_map_);
  bool symbolic_map_was_simplified = simplified_symbolic_map != symbolic_map_;
  if (symbolic_map_was_simplified) {
    symbolic_map_ = simplified_symbolic_map;
    // TODO: b/446856820 - Invalidate the cached affine_map_ by resetting it.
    // This forces GetAffineMap() to recompute it from the updated symbolic_map_
    // the next time it's called. This mechanism will be removed after the
    // migration to SymbolicMap is complete and GetAffineMap() is removed.
    affine_map_ = AffineMap();
  }
  return symbolic_map_was_simplified || constraints_were_simplified;
}

bool SymbolicExprSimplifier::SimplifyConstraintExprs(IndexingMap& map) {
  // Simplify symbolic expression in the constraints_.
  std::vector<SymbolicExpr> to_remove;
  std::vector<std::pair<SymbolicExpr, Interval>> to_add;
  for (const auto& [expr, range] : map.GetSymbolicConstraints()) {
    SymbolicExpr simplified = Simplify(expr);

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

bool SymbolicExprSimplifier::SimplifyConstraintRanges(IndexingMap& map) {
  std::vector<SymbolicExpr> to_remove;
  std::vector<std::pair<SymbolicExpr, Interval>> to_add;
  for (const auto& [expr, range] : map.GetSymbolicConstraints()) {
    SymbolicExpr simplified_expr = expr;
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

std::tuple<SymbolicExpr, int64_t, SymbolicExpr>
SymbolicExprSimplifier::SplitSumByGcd(SymbolicExpr sum) {
  std::optional<int64_t> multiplier_gcd = std::nullopt;
  SymbolicExpr no_multiplier = zero_;
  VisitSummands(sum, [&](SymbolicExpr expr) {
    if (auto multiplier = GetConstantRhs(expr, SymbolicExprType::kMul)) {
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

  auto scaled = MapSummands(sum, [&](SymbolicExpr expr) {
    if (auto multiplier = GetConstantRhs(expr, SymbolicExprType::kMul)) {
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

void GetUsedParametersImpl(const SymbolicExpr& expr,
                           UsedParameters& used_parameters, int64_t num_dims) {
  if (IsDimension(expr, num_dims)) {
    used_parameters.dimension_ids.insert(GetDimensionIndex(expr, num_dims));
    return;
  }
  if (IsSymbol(expr, num_dims)) {
    used_parameters.symbol_ids.insert(GetSymbolIndex(expr, num_dims));
    return;
  }
  if (expr.IsBinaryOp()) {
    GetUsedParametersImpl(expr.GetLHS(), used_parameters, num_dims);
    GetUsedParametersImpl(expr.GetRHS(), used_parameters, num_dims);
  }
}

// Returns IDs of dimensions and symbols that participate in SymbolicExpr.
UsedParameters GetUsedParameters(const SymbolicExpr& expr, int64_t num_dims) {
  UsedParameters used_parameters;
  GetUsedParametersImpl(expr, used_parameters, num_dims);
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
  SmallVector<SymbolicExpr> constraints_with_unused_vars_only;
};

// Detects unused dimensions and symbols in the inde
UnusedVariables DetectUnusedVariables(const IndexingMap& indexing_map,
                                      bool removing_dims,
                                      bool removing_symbols) {
  SymbolicMap symbolic_map = indexing_map.GetSymbolicMap();

  UnusedVariables unused_vars;
  // Find unused dimensions and symbols in the symbolic_map.
  unused_vars.unused_dims = GetUnusedDimensionsBitVector(symbolic_map);
  unused_vars.unused_symbols = GetUnusedSymbolsBitVector(symbolic_map);

  // Check if the symbols that are unused in `symbolic_map` are also unused in
  // expressions.
  SmallVector<std::pair<SymbolicExpr, UsedParameters>, 2>
      unused_constraints_candidates;
  for (const auto& [expr, range] : indexing_map.GetSymbolicConstraints()) {
    UsedParameters used_parameters =
        GetUsedParameters(expr, indexing_map.GetDimensionCount());
    // If the expression uses only symbols that are unused in `symbolic_map`,
    // then we can remove it (because we will remove the symbols as well). Note
    // that the same is not true for dimensions, because of the existence of the
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
  unsigned num_dims_after = num_dims_before - unused_dims.count();

  // Compress DimVars.
  SmallVector<SymbolicExpr> dim_replacements;
  if (num_dims_changed) {
    symbolic_map_ = CompressDims(symbolic_map_, unused_dims);
    // TODO: b/446856820 - Invalidate the cached affine_map_ by resetting it.
    // This forces GetAffineMap() to recompute it from the updated symbolic_map_
    // the next time it's called. This mechanism will be removed after the
    // migration to SymbolicMap is complete and GetAffineMap() is removed.
    affine_map_ = AffineMap();
    std::vector<IndexingMap::Variable> compressed_dim_vars;
    dim_replacements = SmallVector<SymbolicExpr, 2>(
        num_dims_before, CreateSymbolicConstant(0, mlir_context));
    int64_t used_dims_count = 0;
    for (int i = 0; i < unused_dims.size(); ++i) {
      if (!unused_dims[i]) {
        compressed_dim_vars.push_back(dim_vars_[i]);
        dim_replacements[i] = CreateDimExpr(used_dims_count++, mlir_context);
      }
    }
    dim_vars_ = std::move(compressed_dim_vars);
  }

  // Compress RangeVars and RTVars.
  SmallVector<SymbolicExpr> symbol_replacements;
  if (num_symbols_changed) {
    symbolic_map_ = CompressSymbols(symbolic_map_, unused_symbols);
    // TODO: b/446856820 - Invalidate the cached affine_map_ by resetting it.
    // This forces GetAffineMap() to recompute it from the updated symbolic_map_
    // the next time it's called. This mechanism will be removed after the
    // migration to SymbolicMap is complete and GetAffineMap() is removed.
    affine_map_ = AffineMap();
    symbol_replacements = SmallVector<SymbolicExpr, 2>(
        num_symbols_before, CreateSymbolicConstant(0, mlir_context));
    std::vector<IndexingMap::Variable> compressed_range_vars;
    std::vector<IndexingMap::Variable> compressed_rt_vars;
    int64_t used_symbols_count = 0;
    auto range_vars_count = range_vars_.size();
    for (int i = 0; i < unused_symbols.size(); ++i) {
      if (!unused_symbols[i]) {
        if (i < range_vars_count) {
          compressed_range_vars.push_back(range_vars_[i]);
        } else {
          compressed_rt_vars.push_back(rt_vars_[i - range_vars_count]);
        }
        symbol_replacements[i] = CreateSymbolExpr(used_symbols_count++,
                                                  num_dims_after, mlir_context);
      }
    }
    range_vars_ = std::move(compressed_range_vars);
    rt_vars_ = std::move(compressed_rt_vars);
  }

  // Remove constraints.
  std::vector<SymbolicExpr> to_remove;
  std::vector<std::pair<SymbolicExpr, Interval>> to_add;
  for (const auto& [expr, range] : constraints_) {
    auto updated_expr =
        (num_dims_changed
             ? expr.ReplaceDimsAndSymbols(dim_replacements, symbol_replacements)
             : expr.ReplaceSymbols(symbol_replacements, num_dims_before));
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
  for (SymbolicExpr expr : unused_vars.constraints_with_unused_vars_only) {
    constraints_.erase(expr);
  }
  if (!CompressVars(/*unused_dims=*/{}, unused_vars.unused_symbols)) {
    return {};
  }
  return std::move(unused_vars).unused_symbols;
}

void IndexingMap::ResetToKnownEmpty() {
  auto zero = CreateSymbolicConstant(0, GetMLIRContext());
  // TODO: b/446856820 - Invalidate the cached affine_map_ by resetting it.
  // This forces GetAffineMap() to recompute it from the updated symbolic_map_
  // the next time it's called. This mechanism will be removed after the
  // migration to SymbolicMap is complete and GetAffineMap() is removed.
  affine_map_ = AffineMap();
  symbolic_map_ = SymbolicMap::Get(
      GetMLIRContext(), symbolic_map_.GetNumDims(),
      symbolic_map_.GetNumSymbols(),
      llvm::SmallVector<SymbolicExpr>(symbolic_map_.GetNumResults(), zero));
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
  for (SymbolicExpr expr : unused_vars.constraints_with_unused_vars_only) {
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
  llvm::MapVector<SymbolicExpr, llvm::SmallVector<SymbolicExpr, 2>>
      grouped_constraints;
  for (const auto& [expr, _] : constraints_) {
    if (expr.GetType() != SymbolicExprType::kMod) {
      continue;
    }
    grouped_constraints[expr.GetLHS()].push_back(expr);
  }

  // Merge constraints of type MOD.
  // (X mod 3 == 0) & (X mod 2 == 0) => (X mod 6 == 0)
  for (const auto& [lhs, binops] : grouped_constraints) {
    llvm::MapVector<int64_t, llvm::SmallVector<SymbolicExpr, 2>> mod_groups;
    for (const auto& binop : binops) {
      Interval mod_result = constraints_[binop];
      if (mod_result.IsPoint()) {
        mod_groups[mod_result.lower].push_back(binop);
      }
    }
    if (mod_groups.empty()) continue;

    // Update domain for dimensions and symbols only.
    Interval* interval_to_update = nullptr;
    if (IsDimension(lhs, GetDimensionCount())) {
      interval_to_update = &GetMutableDimensionBound(
          GetDimensionIndex(lhs, GetDimensionCount()));
    } else if (IsSymbol(lhs, GetDimensionCount())) {
      interval_to_update =
          &GetMutableSymbolBound(GetSymbolIndex(lhs, GetDimensionCount()));
    }
    for (const auto& [res, ops] : mod_groups) {
      // Calculate least common multiple for the divisors.
      int64_t div = 1;
      for (const auto& op : ops) {
        int64_t rhs_value =
            range_evaluator.ComputeExpressionRange(op.GetRHS()).lower;
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
  SymbolicMap producer_symbolic_map = second.GetSymbolicMap();
  SymbolicMap composed_map =
      producer_symbolic_map.Compose(first.GetSymbolicMap());
  int64_t composed_dims = composed_map.GetNumDims();
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
  SmallVector<SymbolicExpr, 4> symbol_replacements =
      GetComposedSymbolsPermutationToCorrectOrder(first, second, composed_dims);
  if (!symbol_replacements.empty()) {
    composed_map = composed_map.ReplaceDimsAndSymbols(
        /*dim_replacements=*/{}, symbol_replacements, composed_dims,
        composed_map.GetNumSymbols());
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
  llvm::SmallVector<SymbolicExpr> constraints;
  llvm::SmallVector<Interval> constraints_ranges;
  for (const auto& [expr, range] : second.GetSymbolicConstraints()) {
    constraints.push_back(expr);
    constraints_ranges.push_back(range);
  }
  auto constraints_map =
      SymbolicMap::Get(mlir_context, producer_symbolic_map.GetNumDims(),
                       producer_symbolic_map.GetNumSymbols(), constraints);
  auto remapped_constraints =
      constraints_map.Compose(first.GetSymbolicMap())
          .ReplaceDimsAndSymbols(/*dim_replacements=*/{}, symbol_replacements,
                                 composed_indexing_map.GetDimensionCount(),
                                 composed_indexing_map.GetSymbolCount());
  for (const auto& [expr, range] :
       llvm::zip(remapped_constraints.GetResults(), constraints_ranges)) {
    composed_indexing_map.AddConstraint(expr, range);
  }
  // Remap symbol ids and add constraints that are already present in the
  // consumer_map.
  SmallVector<SymbolicExpr, 4> first_map_symbols_to_composed_symbols =
      MapSymbolsToComposedSymbolsList(first, composed_indexing_map);
  for (const auto& [expr, range] : first.GetSymbolicConstraints()) {
    composed_indexing_map.AddConstraint(
        expr.ReplaceSymbols(first_map_symbols_to_composed_symbols,
                            composed_dims),
        range);
  }
  // Add constraints for consumer's codomain w.r.t. producer's domain.

  for (int64_t index = 0; index < first.GetSymbolicMap().GetNumResults();
       ++index) {
    auto expr = first.GetSymbolicMap().GetResult(index);
    Interval producer_dim_range =
        second.GetDimensionBound(static_cast<int64_t>(index));
    composed_indexing_map.AddConstraint(
        expr.ReplaceSymbols(first_map_symbols_to_composed_symbols,
                            composed_dims),
        producer_dim_range);
  }
  return composed_indexing_map;
}

bool IndexingMap::RescaleSymbols() {
  MergeModConstraints();

  llvm::DenseSet<SymbolicExpr> to_delete;
  llvm::DenseMap<SymbolicExpr, SymbolicExpr> to_replace;

  for (const auto& [expr, range] : constraints_) {
    if (range.lower != range.upper) {
      continue;
    }
    auto shift_value = range.lower;

    if (expr.GetType() != SymbolicExprType::kMod) {
      continue;
    }
    auto mod_expr = expr;

    auto constant_expr = mod_expr.GetRHS();
    if (constant_expr.GetType() != SymbolicExprType::kConstant) {
      continue;
    }

    // We don't rescale mod expressions with non-positive divisors.
    if (constant_expr.GetValue() <= 0) {
      continue;
    }
    auto scaling_factor = constant_expr.GetValue();

    if (!IsSymbol(mod_expr.GetLHS(), GetDimensionCount())) {
      continue;
    }
    SymbolicExpr symbol_expr = mod_expr.GetLHS();

    // In case there are two mod constraints which were not merged, we only
    // support rescaling by one.
    // TODO(b/347240603): The result shouldn't depend on the hashmap's iteration
    // order.
    if (to_replace.contains(symbol_expr)) {
      continue;
    }

    to_replace[symbol_expr] = constant_expr * symbol_expr + shift_value;
    to_delete.insert(expr);

    symbolic_map_ = symbolic_map_.Replace(
        symbol_expr, constant_expr * symbol_expr + shift_value);
    // TODO: b/446856820 - Invalidate the cached affine_map_ by resetting it.
    // This forces GetAffineMap() to recompute it from the updated symbolic_map_
    // the next time it's called. This mechanism will be removed after the
    // migration to SymbolicMap is complete and GetAffineMap() is removed.
    affine_map_ = AffineMap();

    auto& symbol_range =
        range_vars_[GetSymbolIndex(symbol_expr, GetDimensionCount())].bounds;
    symbol_range.lower = (symbol_range.lower - shift_value) / scaling_factor;
    symbol_range.upper = (symbol_range.upper - shift_value) / scaling_factor;
  }

  llvm::MapVector<SymbolicExpr, Interval> new_constraints;
  for (const auto& [expr, range] : constraints_) {
    if (!to_delete.contains(expr)) {
      new_constraints[expr.Replace(to_replace)] = range;
    }
  }
  constraints_ = std::move(new_constraints);

  return !to_delete.empty();
}

bool IndexingMap::IsRangeVarSymbol(SymbolicExpr symbol) const {
  unsigned int position = GetSymbolIndex(symbol, GetDimensionCount());
  CHECK_LE(position, GetSymbolCount());
  return position < range_vars_.size();
}

bool IndexingMap::IsRTVarSymbol(SymbolicExpr symbol) const {
  unsigned int position = GetSymbolIndex(symbol, GetDimensionCount());
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
  SmallVector<SymbolicExpr> syms_replacements;
  int64_t symbol_id = num_dims;
  for (const IndexingMap::Variable& var :
       llvm::concat<const IndexingMap::Variable>(range_vars_, rt_vars_)) {
    syms_replacements.push_back(CreateDimExpr(symbol_id++, mlir_context));
    new_dim_vars.push_back(IndexingMap::Variable{var.bounds});
  }

  // Update constraints.
  SmallVector<std::pair<SymbolicExpr, Interval>, 4> new_constraints;
  for (const auto& [expr, range] : constraints_) {
    new_constraints.push_back(std::make_pair(
        expr.ReplaceSymbols(syms_replacements, num_dims), range));
  }

  SymbolicMap canonical_map =
      symbolic_map_.ReplaceDimsAndSymbols({}, syms_replacements, num_vars, 0);
  IndexingMap new_indexing_map(canonical_map, new_dim_vars, /*range_vars=*/{},
                               /*rt_vars=*/{}, new_constraints);
  return new_indexing_map;
}

IndexingMap ConvertRangeVariablesToDimensions(
    const IndexingMap& map, ArrayRef<int64_t> range_var_indices) {
  CHECK(std::is_sorted(range_var_indices.begin(), range_var_indices.end()));
  auto* mlir_context = map.GetMLIRContext();
  SymbolicMap symbolic_map = map.GetSymbolicMap();
  int64_t num_dims = symbolic_map.GetNumDims();
  int64_t num_symbols = symbolic_map.GetNumSymbols();
  int64_t num_result_dims = num_dims + range_var_indices.size();
  int64_t num_result_symbols = num_symbols - range_var_indices.size();
  // Update the symbolic map and the variables.
  std::vector<IndexingMap::Variable> dims = map.GetDimVars();
  std::vector<IndexingMap::Variable> range_vars;
  SmallVector<SymbolicExpr> symbol_replacements;
  symbol_replacements.reserve(num_symbols);
  int64_t updated_count = 0;
  int64_t range_var_indices_count = range_var_indices.size();
  for (int i = 0; i < map.GetRangeVarsCount(); ++i) {
    auto range_var = map.GetRangeVar(i);
    if (updated_count < range_var_indices_count &&
        i == range_var_indices[updated_count]) {
      symbol_replacements.push_back(
          CreateDimExpr(num_dims + updated_count, mlir_context));
      dims.push_back(range_var);
      updated_count++;
    } else {
      symbol_replacements.push_back(
          CreateSymbolExpr(i - updated_count, num_result_dims, mlir_context));
      range_vars.push_back(range_var);
    }
  }
  CHECK_EQ(updated_count, range_var_indices_count)
      << "Not all replacements were used";
  for (int i = 0; i < map.GetRTVarsCount(); ++i) {
    symbol_replacements.push_back(
        CreateSymbolExpr(map.GetRangeVarsCount() - range_var_indices_count + i,
                         num_result_dims, mlir_context));
  }
  CHECK_EQ(symbol_replacements.size(), symbolic_map.GetNumSymbols())
      << "All symbols must be updated";

  SymbolicMap converted_symbolic_map = symbolic_map.ReplaceDimsAndSymbols(
      /*dim_replacements=*/{}, symbol_replacements,
      /*num_result_dims=*/num_result_dims,
      /*num_result_symbols=*/num_result_symbols);

  // Update the constraints.
  std::vector<std::pair<SymbolicExpr, Interval>> constraints;
  constraints.reserve(map.GetConstraintsCount());
  for (auto constraint : map.GetSymbolicConstraints()) {
    constraints.push_back(
        {constraint.first.ReplaceSymbols(symbol_replacements, num_dims),
         constraint.second});
  }
  return IndexingMap{converted_symbolic_map, std::move(dims),
                     std::move(range_vars), map.GetRTVars(), constraints};
}

}  // namespace xla
