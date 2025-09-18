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

#include "xla/service/gpu/model/experimental/symbolic_map_converter.h"

#include <cstdint>

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"
#include "xla/service/gpu/model/experimental/symbolic_map.h"

namespace xla {
namespace gpu {
namespace {

// Helper function to convert xla::gpu::SymbolicExpr to mlir::AffineExpr.
mlir::AffineExpr SymbolicToAffine(SymbolicExpr symbolic_expr,
                                  mlir::MLIRContext* context, int num_dims) {
  mlir::AffineExpr lhs, rhs;
  if (symbolic_expr.GetLHS() && symbolic_expr.GetRHS()) {
    lhs = SymbolicToAffine(symbolic_expr.GetLHS(), context, num_dims);
    rhs = SymbolicToAffine(symbolic_expr.GetRHS(), context, num_dims);
    if (!lhs || !rhs) {
      return mlir::AffineExpr();
    }
  }

  switch (symbolic_expr.GetType()) {
    case SymbolicExprType::kConstant:
      return mlir::getAffineConstantExpr(symbolic_expr.GetValue(), context);
    case SymbolicExprType::kVariable: {
      int64_t id = symbolic_expr.GetValue();
      if (id < num_dims) {
        return mlir::getAffineDimExpr(id, context);
      }
      return mlir::getAffineSymbolExpr(id - num_dims, context);
    }
    case SymbolicExprType::kAdd:
      return lhs + rhs;
    case SymbolicExprType::kMul:
      return lhs * rhs;
    case SymbolicExprType::kFloorDiv:
      return mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv, lhs,
                                         rhs);
    case SymbolicExprType::kCeilDiv:
      return mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::CeilDiv, lhs,
                                         rhs);
    case SymbolicExprType::kMod:
      return mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, lhs, rhs);
    default:
      // kMax and kMin are not supported in mlir::AffineExpr.
      return mlir::AffineExpr();
  }
}

}  // namespace

// Helper function to convert mlir::AffineExpr to xla::gpu::SymbolicExpr.
SymbolicExpr AffineToSymbolicExpr(mlir::AffineExpr affine_expr,
                                  SymbolicExprContext* context, int num_dims) {
  switch (affine_expr.getKind()) {
    case mlir::AffineExprKind::Constant:
      return context->CreateConstant(
          mlir::cast<mlir::AffineConstantExpr>(affine_expr).getValue());
    case mlir::AffineExprKind::DimId:
      return context->CreateVariable(
          mlir::cast<mlir::AffineDimExpr>(affine_expr).getPosition());
    case mlir::AffineExprKind::SymbolId:
      return context->CreateVariable(
          mlir::cast<mlir::AffineSymbolExpr>(affine_expr).getPosition() +
          num_dims);
    case mlir::AffineExprKind::Add: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(affine_expr);
      return AffineToSymbolicExpr(bin_op.getLHS(), context, num_dims) +
             AffineToSymbolicExpr(bin_op.getRHS(), context, num_dims);
    }
    case mlir::AffineExprKind::Mul: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(affine_expr);
      return AffineToSymbolicExpr(bin_op.getLHS(), context, num_dims) *
             AffineToSymbolicExpr(bin_op.getRHS(), context, num_dims);
    }
    case mlir::AffineExprKind::FloorDiv: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(affine_expr);
      return AffineToSymbolicExpr(bin_op.getLHS(), context, num_dims)
          .floorDiv(AffineToSymbolicExpr(bin_op.getRHS(), context, num_dims));
    }
    case mlir::AffineExprKind::CeilDiv: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(affine_expr);
      return AffineToSymbolicExpr(bin_op.getLHS(), context, num_dims)
          .ceilDiv(AffineToSymbolicExpr(bin_op.getRHS(), context, num_dims));
    }
    case mlir::AffineExprKind::Mod: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(affine_expr);
      return AffineToSymbolicExpr(bin_op.getLHS(), context, num_dims) %
             AffineToSymbolicExpr(bin_op.getRHS(), context, num_dims);
    }
  }
}

SymbolicMap AffineMapToSymbolicMap(const mlir::AffineMap& affine_map,
                                   SymbolicExprContext* context) {
  llvm::SmallVector<SymbolicExpr> results;
  results.reserve(affine_map.getNumResults());
  int num_dims = affine_map.getNumDims();
  for (mlir::AffineExpr expr : affine_map.getResults()) {
    results.push_back(AffineToSymbolicExpr(expr, context, num_dims));
  }
  return SymbolicMap::Get(context, num_dims, affine_map.getNumSymbols(),
                          results);
}

mlir::AffineMap SymbolicMapToAffineMap(SymbolicMap symbolic_map,
                                       mlir::MLIRContext* context) {
  int num_dims = symbolic_map.GetNumDims();
  int num_symbols = symbolic_map.GetNumSymbols();
  llvm::SmallVector<mlir::AffineExpr> results;
  results.reserve(symbolic_map.GetNumResults());
  for (SymbolicExpr expr : symbolic_map.GetResults()) {
    mlir::AffineExpr affine_expr =
        SymbolicToAffine(expr, context, symbolic_map.GetNumDims());
    if (!affine_expr) {
      // Conversion failed.
      return mlir::AffineMap();
    }
    results.push_back(affine_expr);
  }

  return mlir::AffineMap::get(num_dims, num_symbols, results, context);
}

llvm::MapVector<SymbolicExpr, Interval>
ConvertAffineConstraintsToSymbolicConstraints(
    const llvm::MapVector<mlir::AffineExpr, Interval>& affine_constraints,
    SymbolicExprContext* context, int num_dims) {
  llvm::MapVector<SymbolicExpr, Interval> symbolic_constraints;
  for (const auto& [affine_expr, interval] : affine_constraints) {
    SymbolicExpr expr = AffineToSymbolicExpr(affine_expr, context, num_dims);
    symbolic_constraints[expr] = interval;
  }
  return symbolic_constraints;
}

}  // namespace gpu
}  // namespace xla
