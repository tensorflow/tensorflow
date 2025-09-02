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

#include "xla/service/gpu/model/experimental/symbolic_expr_simplifier.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <tuple>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"
#include "xla/service/gpu/model/experimental/symbolic_map.h"

namespace xla {
namespace gpu {

namespace {

// Helper to get the constant value from the RHS of a binary operation.
std::optional<int64_t> GetConstantRhs(SymbolicExpr expr,
                                      SymbolicExprType kind) {
  if (expr.GetType() != kind) {
    return std::nullopt;
  }
  SymbolicExpr rhs = expr.GetRHS();
  if (rhs.GetType() == SymbolicExprType::kConstant) {
    return rhs.GetValue();
  }
  return std::nullopt;
}

// Extracts the multiplier from an expression, if it's a multiplication by a
// constant.
std::pair<SymbolicExpr, int64_t> ExtractMultiplier(SymbolicExpr expr) {
  if (expr.GetType() == SymbolicExprType::kMul) {
    if (auto mul = GetConstantRhs(expr, SymbolicExprType::kMul)) {
      return {expr.GetLHS(), *mul};
    }
  }
  return {expr, 1};
}

// Calls `visit` for each summand in an arbitrarily nested sum.
template <typename Fn>
void VisitSummands(SymbolicExpr expr, const Fn& visit) {
  if (expr.GetType() == SymbolicExprType::kAdd) {
    VisitSummands(expr.GetLHS(), visit);
    VisitSummands(expr.GetRHS(), visit);
  } else {
    visit(expr);
  }
}

}  // namespace

// Maps each summand in an arbitrarily nested sum.
template <typename Fn>
SymbolicExpr MapSummands(SymbolicExpr expr, const Fn& fn,
                         SymbolicExprContext* ctx) {
  if (expr.GetType() == SymbolicExprType::kAdd) {
    auto lhs = MapSummands(expr.GetLHS(), fn, ctx);
    auto rhs = MapSummands(expr.GetRHS(), fn, ctx);
    if (lhs == expr.GetLHS() && rhs == expr.GetRHS()) {
      return expr;
    }
    return ctx->CreateBinaryOp(SymbolicExprType::kAdd, lhs, rhs);
  }
  return fn(expr);
}

// Compares the two expression by their AST. The ordering is arbitrary but
// similar to what MLIR's simplifier does.
int CompareExprs(SymbolicExpr a, SymbolicExpr b) {
  if ((b.GetType() == SymbolicExprType::kConstant) !=
      (a.GetType() == SymbolicExprType::kConstant)) {
    return a.GetType() == SymbolicExprType::kConstant ? 1 : -1;
  }
  if (a.GetType() < b.GetType()) {
    return -1;
  }
  if (a.GetType() > b.GetType()) {
    return 1;
  }
  CHECK(a.GetType() == b.GetType());
  int64_t a_value = 0;
  int64_t b_value = 0;
  switch (a.GetType()) {
    case SymbolicExprType::kAdd:
    case SymbolicExprType::kMul:
    case SymbolicExprType::kFloorDiv:
    case SymbolicExprType::kCeilDiv:
    case SymbolicExprType::kMod:
    case SymbolicExprType::kMin:
    case SymbolicExprType::kMax: {
      auto lhs = CompareExprs(a.GetLHS(), b.GetLHS());
      if (lhs != 0) {
        return lhs;
      }
      return CompareExprs(a.GetRHS(), b.GetRHS());
    }
    case SymbolicExprType::kConstant:
    case SymbolicExprType::kVariable: {
      a_value = a.GetValue();
      b_value = b.GetValue();
      break;
    }
  }
  return a_value < b_value ? -1 : (a_value > b_value ? 1 : 0);
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

  auto scaled = MapSummands(
      sum,
      [&](SymbolicExpr expr) {
        if (auto multiplier = GetConstantRhs(expr, SymbolicExprType::kMul)) {
          // Rescale the multiplier.
          return expr.GetLHS() * (*multiplier / *multiplier_gcd);
        }
        // Extract the summand.
        no_multiplier = no_multiplier + expr;
        return zero_;
      },
      ctx_);

  return {scaled, *multiplier_gcd, no_multiplier};
}

SymbolicExpr SymbolicExprSimplifier::SimplifyModDiv(SymbolicExpr dividend,
                                                    int64_t divisor) {
  if (auto mod = GetConstantRhs(dividend, SymbolicExprType::kMod)) {
    if (*mod % divisor == 0) {
      return dividend.GetLHS().floorDiv(divisor) % (*mod / divisor);
    }
  }
  return {};  // Return null SymbolicExpr
}

SymbolicExpr SymbolicExprSimplifier::SimplifyDivDiv(SymbolicExpr dividend,
                                                    int64_t divisor) {
  // The inner divisor here can be negative.
  if (auto inner_divisor =
          GetConstantRhs(dividend, SymbolicExprType::kFloorDiv)) {
    return dividend.GetLHS().floorDiv(divisor * *inner_divisor);
  }
  return {};  // Return null SymbolicExpr
}

SymbolicExpr SymbolicExprSimplifier::SimplifySumDiv(SymbolicExpr dividend,
                                                    int64_t divisor) {
  SymbolicExpr extracted = zero_;
  bool changed = false;
  auto new_dividend = MapSummands(
      dividend,
      [&](SymbolicExpr expr) {
        auto [base, multiplier] = ExtractMultiplier(expr);
        if (multiplier % divisor == 0) {
          int64_t factor = multiplier / divisor;
          extracted = extracted + base * factor;
          changed = true;
          return zero_;
        }
        return expr;
      },
      ctx_);

  // TODO(b/442385842): Add range-based simplifications here.

  // Porting inner divisor logic
  std::optional<int64_t> inner_divisor = std::nullopt;
  int num_inner_divisors = 0;
  VisitSummands(new_dividend, [&](SymbolicExpr summand) {
    if (summand.GetType() == SymbolicExprType::kFloorDiv) {
      if (auto div_val = GetConstantRhs(summand, SymbolicExprType::kFloorDiv)) {
        inner_divisor = div_val;
        ++num_inner_divisors;
      }
    }
  });

  if (num_inner_divisors == 1 && inner_divisor.has_value()) {
    new_dividend = MapSummands(
        new_dividend,
        [&](SymbolicExpr summand) {
          if (summand.GetType() == SymbolicExprType::kFloorDiv &&
              GetConstantRhs(summand, SymbolicExprType::kFloorDiv) ==
                  inner_divisor) {
            return summand.GetLHS();
          }
          return summand * *inner_divisor;
        },
        ctx_);
    divisor *= *inner_divisor;
    changed = true;
  }

  if (!changed) {
    return {};  // Return null SymbolicExpr
  }

  return new_dividend.floorDiv(divisor) + extracted;
}

SymbolicMap SymbolicExprSimplifier::Simplify(SymbolicMap map) {
  if (map.IsEmpty()) {
    return map;
  }

  llvm::SmallVector<SymbolicExpr> results;
  results.reserve(map.GetNumResults());
  bool changed = false;
  for (SymbolicExpr expr : map.GetResults()) {
    SymbolicExpr simplified_expr = Simplify(expr);
    if (simplified_expr != expr) {
      changed = true;
    }
    results.push_back(simplified_expr);
  }

  if (!changed) {
    return map;
  }
  return SymbolicMap::Get(map.GetContext(), map.GetNumDims(),
                          map.GetNumSymbols(), results);
}

SymbolicExpr SymbolicExprSimplifier::Simplify(SymbolicExpr expr) {
  if (!expr) {
    return expr;
  }
  SymbolicExpr simplified = expr;
  while (true) {
    SymbolicExpr result = SimplifyOnce(simplified);
    if (result == simplified) {
      return result;
    }
    simplified = result;
  }
}

bool SymbolicExprSimplifier::SimplifyConstraintExprs(SymbolicMap& map) {
  // TODO(b/442385842): Implement constraint expression simplification.
  return false;
}

bool SymbolicExprSimplifier::SimplifyConstraintRanges(SymbolicMap& map) {
  // TODO(b/442385842): Implement constraint range simplification.
  return false;
}

SymbolicExpr SymbolicExprSimplifier::SimplifyOnce(SymbolicExpr expr) {
  if (!expr) {
    return expr;
  }

  SymbolicExprType type = expr.GetType();
  if (type == SymbolicExprType::kConstant ||
      type == SymbolicExprType::kVariable) {
    return expr;
  }

  SymbolicExpr lhs = SimplifyOnce(expr.GetLHS());
  SymbolicExpr rhs = SimplifyOnce(expr.GetRHS());

  // If children changed, create a new node and start simplification from there.
  if (lhs != expr.GetLHS() || rhs != expr.GetRHS()) {
    expr = ctx_->CreateBinaryOp(type, lhs, rhs);
  }

  switch (expr.GetType()) {
    case SymbolicExprType::kConstant:
    case SymbolicExprType::kVariable:
      return expr;  // Should not happen due to check above
    case SymbolicExprType::kAdd:
      return RewriteAdd(expr);
    case SymbolicExprType::kMul:
      return RewriteMul(expr);
    case SymbolicExprType::kFloorDiv:
      return RewriteFloorDiv(expr);
    case SymbolicExprType::kCeilDiv:
      return RewriteCeilDiv(expr);
    case SymbolicExprType::kMod:
      return RewriteMod(expr);
    case SymbolicExprType::kMin:
      return RewriteMin(expr);
    case SymbolicExprType::kMax:
      return RewriteMax(expr);
  }
}

SymbolicExpr SymbolicExprSimplifier::RewriteAdd(SymbolicExpr expr) {
  llvm::SmallVector<std::pair<SymbolicExpr, int64_t>, 4> mods;
  llvm::SmallVector<std::pair<SymbolicExpr, int64_t>, 4> divs;
  llvm::SmallDenseMap<SymbolicExpr, int64_t> summands;

  VisitSummands(expr, [&](SymbolicExpr e) {
    SymbolicExpr simplified = SimplifyOnce(e);
    auto [base, multiplier] = ExtractMultiplier(simplified);
    if (base.GetType() == SymbolicExprType::kMod) {
      mods.push_back({base, multiplier});
    } else if (base.GetType() == SymbolicExprType::kFloorDiv) {
      divs.push_back({base, multiplier});
    } else {
      summands[base] += multiplier;
    }
  });

  // TODO(b/442385842): Add warning for large number of mods/divs.

  if (!divs.empty()) {
    for (int mod_i = 0; mod_i < mods.size(); ++mod_i) {
      auto [mod, mod_mul] = mods[mod_i];
      if (!mod) {
        continue;
      }
      auto mod_c = GetConstantRhs(mod, SymbolicExprType::kMod);
      if (!mod_c) {
        continue;
      }

      SymbolicExpr mod_lhs = mod.GetLHS();
      SymbolicExpr simplified_mod_div = Simplify(mod_lhs.floorDiv(*mod_c));

      for (int div_i = 0; div_i < divs.size(); ++div_i) {
        auto [div, div_mul] = divs[div_i];
        if (!div) {
          continue;
        }

        if (simplified_mod_div != div) {
          continue;
        }

        if ((div_mul % mod_mul != 0) || (div_mul / mod_mul) != *mod_c) {
          continue;
        }

        summands[mod_lhs] += mod_mul;
        divs[div_i].first = {};
        mods[mod_i].first = {};
        break;
      }
    }

    for (int div_i = 0; div_i < divs.size(); ++div_i) {
      auto [div, div_mul] = divs[div_i];
      if (!div || div_mul > 0) {
        continue;
      }

      auto div_c = GetConstantRhs(div, SymbolicExprType::kFloorDiv);
      if (!div_c || *div_c <= 0 || (div_mul % *div_c != 0)) {
        continue;
      }

      int64_t b = div_mul / *div_c;
      SymbolicExpr x = div.GetLHS();
      VisitSummands(x, [&](SymbolicExpr summand) { summands[summand] += b; });
      mods.push_back({x % *div_c, -b});
      divs[div_i].first = {};
    }
  }

  for (auto const& [e, mul] : mods) {
    if (e) {
      summands[e] += mul;
    }
  }
  for (auto const& [e, mul] : divs) {
    if (e) {
      summands[e] += mul;
    }
  }

  llvm::SmallVector<SymbolicExpr, 4> new_terms;
  int64_t constant_term = 0;

  for (auto const& [base, coeff] : summands) {
    if (coeff == 0) {
      continue;
    }
    if (base.GetType() == SymbolicExprType::kConstant) {
      constant_term += base.GetValue() * coeff;
    } else {
      new_terms.push_back(base * coeff);
    }
  }

  if (constant_term != 0) {
    new_terms.push_back(ctx_->CreateConstant(constant_term));
  }

  if (new_terms.empty()) {
    return zero_;
  }

  std::sort(new_terms.begin(), new_terms.end(), CompareExprs);

  SymbolicExpr result = new_terms[0];
  for (size_t i = 1; i < new_terms.size(); ++i) {
    result = result + new_terms[i];
  }
  return result;
}

SymbolicExpr SymbolicExprSimplifier::RewriteMul(SymbolicExpr expr) {
  SymbolicExpr lhs = expr.GetLHS();
  SymbolicExpr rhs = expr.GetRHS();

  if (rhs.GetType() != SymbolicExprType::kConstant) {
    // TODO(b/442385842): Handle non-constant RHS?
    return expr;
  }
  int64_t multiplier = rhs.GetValue();

  if (multiplier == 0) {
    return zero_;
  }
  if (multiplier == 1) {
    return lhs;
  }

  if (lhs.GetType() == SymbolicExprType::kAdd) {
    return MapSummands(
        lhs,
        [&](SymbolicExpr summand) {
          return ctx_->CreateBinaryOp(SymbolicExprType::kMul, summand, rhs);
        },
        ctx_);
  }
  return expr;
}

SymbolicExpr SymbolicExprSimplifier::RewriteFloorDiv(SymbolicExpr expr) {
  SymbolicExpr lhs = expr.GetLHS();
  SymbolicExpr rhs = expr.GetRHS();

  if (rhs.GetType() != SymbolicExprType::kConstant) {
    return expr;
  }
  int64_t divisor = rhs.GetValue();
  CHECK_NE(divisor, 0);

  if (divisor == 1) {
    return lhs;
  }
  if (divisor == -1) {
    return -lhs;
  }

  if (auto result = SimplifyModDiv(lhs, divisor)) {
    return result;
  }
  if (auto result = SimplifyDivDiv(lhs, divisor)) {
    return result;
  }
  if (auto result = SimplifySumDiv(lhs, divisor)) {
    return result;
  }
  return expr;
}

SymbolicExpr SymbolicExprSimplifier::RewriteCeilDiv(SymbolicExpr expr) {
  // TODO(b/442385842): Implement rules for CeilDiv
  return expr;
}

SymbolicExpr SymbolicExprSimplifier::RewriteMod(SymbolicExpr expr) {
  SymbolicExpr lhs = expr.GetLHS();
  SymbolicExpr rhs = expr.GetRHS();

  if (rhs.GetType() != SymbolicExprType::kConstant) {
    return expr;
  }
  int64_t modulus = rhs.GetValue();
  CHECK_NE(modulus, 0);

  if (modulus == 1 || modulus == -1) {
    return zero_;
  }

  // Rewrite `(c * a) % ab` to `(c % b) * a`.
  if (lhs.GetType() == SymbolicExprType::kMul) {
    if (auto mul = GetConstantRhs(lhs, SymbolicExprType::kMul)) {
      if (*mul > 0 && (modulus % *mul == 0)) {
        return (lhs.GetLHS() % (modulus / *mul)) * *mul;
      }
    }
  }

  int64_t extracted_constant = 0;
  auto new_lhs = MapSummands(
      lhs,
      [&](SymbolicExpr e) {
        if (e.GetType() == SymbolicExprType::kConstant) {
          extracted_constant += e.GetValue();
          return zero_;
        }
        if (auto multiplier = GetConstantRhs(e, SymbolicExprType::kMul)) {
          if (*multiplier % modulus == 0) {
            return zero_;
          }
        }
        return e;
      },
      ctx_);

  if (extracted_constant % modulus != 0) {
    new_lhs = new_lhs + (extracted_constant % modulus);
  }

  // TODO(b/442385842): Port range-based simplifications for Mod.

  return new_lhs == lhs ? expr : new_lhs % modulus;
}

SymbolicExpr SymbolicExprSimplifier::RewriteMin(SymbolicExpr expr) {
  // TODO(b/442385842): Implement rules for Min
  return expr;
}

SymbolicExpr SymbolicExprSimplifier::RewriteMax(SymbolicExpr expr) {
  // TODO(b/442385842): Implement rules for Max
  return expr;
}

}  // namespace gpu
}  // namespace xla
