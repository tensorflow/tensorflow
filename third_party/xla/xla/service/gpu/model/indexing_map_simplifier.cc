/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/indexing_map_simplifier.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"  // from @llvm-project
#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project

namespace xla {
namespace gpu {
namespace {

using mlir::AffineBinaryOpExpr;
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

void IndexingMapSimplifier::SetInclusiveBounds(AffineExpr expr, int64_t lower,
                                               int64_t upper) {
  bounds_[expr] = {lower, upper};
}

IndexingMapSimplifier::Bounds IndexingMapSimplifier::GetInclusiveBounds(
    AffineExpr expr) {
  auto bound = bounds_.find(expr);
  if (bound != bounds_.end()) return bound->second;

  switch (expr.getKind()) {
    case AffineExprKind::Constant: {
      int64_t value = mlir::cast<mlir::AffineConstantExpr>(expr).getValue();
      return bounds_[expr] = {value, value};
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
      auto lhs = GetInclusiveBounds(binary_op.getLHS());
      auto rhs = GetInclusiveBounds(binary_op.getRHS());

      auto& result = bounds_[expr];
      switch (expr.getKind()) {
        case AffineExprKind::Add:
          return result = {lhs.lower + rhs.lower, lhs.upper + rhs.upper};
        case AffineExprKind::Mul: {
          int64_t a = lhs.lower * rhs.lower;
          int64_t b = lhs.upper * rhs.upper;
          return result = {std::min(a, b), std::max(a, b)};
        }
        case AffineExprKind::Mod: {
          CHECK_EQ(rhs.lower, rhs.upper) << "RHS of mod must be a constant";
          int64_t m = rhs.lower;
          if (0 <= lhs.lower && lhs.upper < m) {
            return result = lhs;
          }
          return result = {0, m - 1};
        }
        case AffineExprKind::FloorDiv: {
          CHECK_EQ(rhs.lower, rhs.upper)
              << "RHS of floor_div must be a constant";
          int64_t d = rhs.lower;
          int a = FloorDiv(lhs.lower, d);
          int b = FloorDiv(lhs.upper, d);
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

  auto lhs = GetInclusiveBounds(lhs_simplified);
  auto rhs = GetInclusiveBounds(mod.getRHS());

  // a % b where b is always larger than a?
  if (0 <= lhs.lower && lhs.upper < rhs.lower) return lhs_simplified;

  // The logic below assumes we have a constant RHS.
  if (rhs.lower != rhs.upper) return mod;
  int64_t m = rhs.lower;

  auto new_lhs = RewriteSumIf(lhs_simplified, [&](AffineExpr expr) {
    if (expr.getKind() != AffineExprKind::Mul) {
      return true;
    }

    auto mul_rhs =
        GetInclusiveBounds(mlir::cast<AffineBinaryOpExpr>(expr).getRHS());
    bool remove = mul_rhs.lower == mul_rhs.upper && (mul_rhs.lower % m) == 0;
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
  return getAffineBinaryOpExpr(AffineExprKind::Mod, new_lhs, mod.getRHS());
}

AffineExpr IndexingMapSimplifier::RewriteFloorDiv(AffineBinaryOpExpr div) {
  auto lhs_simplified = SimplifyOnce(div.getLHS());
  auto lhs = GetInclusiveBounds(lhs_simplified);
  auto rhs = GetInclusiveBounds(div.getRHS());

  if (0 <= lhs.lower && lhs.upper < rhs.lower) {
    return getAffineConstantExpr(0, mlir_context_);
  }

  // The logic below assumes we have a constant RHS.
  if (rhs.lower != rhs.upper) return div;
  int64_t d = rhs.lower;

  // If the dividend's range has a single element, return its value.
  int64_t a = FloorDiv(lhs.lower, d);
  int64_t b = FloorDiv(lhs.upper, d);
  if (a == b) {
    return getAffineConstantExpr(a, mlir_context_);
  }

  AffineExpr extracted = getAffineConstantExpr(0, mlir_context_);
  auto new_dividend = RewriteSumIf(lhs_simplified, [&](AffineExpr expr) {
    if (auto multiplier = GetConstantRhsMultiplier(expr)) {
      // (x * 7 + ...) / 3 -> can't extract. We could extract x * 2 and keep
      // one x, but we currently have no reason to do that.
      if (*multiplier % d != 0) return true;
      int64_t factor = *multiplier / d;
      extracted = getAffineBinaryOpExpr(
          AffineExprKind::Add, extracted,
          getAffineBinaryOpExpr(AffineExprKind::Mul,
                                mlir::cast<AffineBinaryOpExpr>(expr).getLHS(),
                                getAffineConstantExpr(factor, mlir_context_)));
      // Remove from dividend.
      return false;
    }

    // Not a constant multiplier, keep in dividend.
    return true;
  });

  // If we removed everything, skip the div.
  if (!new_dividend) return extracted;
  // If we removed nothing, return the original division.
  if (extracted == getAffineConstantExpr(0, mlir_context_) &&
      new_dividend == div.getLHS()) {
    return div;
  }

  return getAffineBinaryOpExpr(
      AffineExprKind::Add, extracted,
      getAffineBinaryOpExpr(AffineExprKind::FloorDiv, new_dividend,
                            div.getRHS()));
}

std::optional<int64_t> IndexingMapSimplifier::GetConstantRhsMultiplier(
    AffineExpr expr) {
  if (expr.getKind() != AffineExprKind::Mul) return std::nullopt;
  auto bound =
      GetInclusiveBounds(mlir::cast<AffineBinaryOpExpr>(expr).getRHS());
  if (bound.lower != bound.upper) return std::nullopt;
  return bound.lower;
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
      return getAffineBinaryOpExpr(AffineExprKind::Add, lhs, rhs);
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
      auto bounds = GetInclusiveBounds(expr);
      if (bounds.lower == bounds.upper) {
        return getAffineConstantExpr(bounds.lower, mlir_context_);
      }
      return expr;
    }

    default:
      return expr;
  }
}

std::string ToString(const AffineMap& affine_map) {
  std::string s;
  llvm::raw_string_ostream ss(s);
  affine_map.print(ss);
  return s;
}

AffineExpr IndexingMapSimplifier::Simplify(AffineExpr expr) {
  while (true) {
    auto simplified = SimplifyOnce(expr);
    if (simplified == expr) return expr;
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
