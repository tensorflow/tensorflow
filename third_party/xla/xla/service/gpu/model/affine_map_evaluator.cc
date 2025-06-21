/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/affine_map_evaluator.h"

#include <cstdint>

#include "absl/log/log.h"
#include "absl/types/span.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LLVM.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {
namespace gpu {

namespace {

using llvm::SmallVector;
using mlir::AffineBinaryOpExpr;
using mlir::AffineConstantExpr;
using mlir::AffineDimExpr;
using mlir::AffineExpr;
using mlir::AffineExprKind;
using mlir::AffineMap;
using mlir::AffineSymbolExpr;

}  // namespace

int64_t EvaluateAffineExpr(AffineExpr expr,
                           absl::Span<int64_t const> dim_values,
                           absl::Span<int64_t const> symbol_values) {
  AffineExprKind kind = expr.getKind();
  if (kind == AffineExprKind::Constant) {
    return mlir::cast<AffineConstantExpr>(expr).getValue();
  }
  if (kind == AffineExprKind::DimId) {
    return dim_values.at(mlir::cast<AffineDimExpr>(expr).getPosition());
  }
  if (kind == AffineExprKind::SymbolId) {
    return symbol_values.at(mlir::cast<AffineSymbolExpr>(expr).getPosition());
  }

  auto binary_expr = mlir::cast<AffineBinaryOpExpr>(expr);
  int64_t lhs =
      EvaluateAffineExpr(binary_expr.getLHS(), dim_values, symbol_values);
  int64_t rhs =
      EvaluateAffineExpr(binary_expr.getRHS(), dim_values, symbol_values);
  switch (kind) {
    case AffineExprKind::Add:
      return lhs + rhs;
    case AffineExprKind::Mul:
      return lhs * rhs;
    case AffineExprKind::FloorDiv:
      return llvm::divideFloorSigned(lhs, rhs);
    case AffineExprKind::Mod:
      return lhs % rhs;
    case AffineExprKind::CeilDiv:
      return llvm::divideCeilSigned(lhs, rhs);
    default:
      LOG(FATAL) << "Unsupported expression";
  }
}

SmallVector<int64_t> EvaluateAffineMap(
    AffineMap affine_map, absl::Span<int64_t const> dim_values,
    absl::Span<int64_t const> symbol_values) {
  CHECK_EQ(affine_map.getNumDims(), dim_values.size());
  CHECK_EQ(affine_map.getNumSymbols(), symbol_values.size());

  SmallVector<int64_t> results;
  results.reserve(affine_map.getNumResults());
  for (auto expr : affine_map.getResults()) {
    results.push_back(EvaluateAffineExpr(expr, dim_values, symbol_values));
  }
  return results;
}

}  // namespace gpu
}  // namespace xla
