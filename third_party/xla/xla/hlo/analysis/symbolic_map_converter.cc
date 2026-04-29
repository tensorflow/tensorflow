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

#include "xla/hlo/analysis/symbolic_map_converter.h"

#include <cstdint>

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"

namespace xla {

// Helper function to convert xla::SymbolicExpr to mlir::AffineExpr.
mlir::AffineExpr SymbolicExprToAffineExpr(SymbolicExpr symbolic_expr,
                                          int num_dims) {
  if (!symbolic_expr) {
    return mlir::AffineExpr();
  }
  mlir::MLIRContext* context = symbolic_expr.GetContext();
  mlir::AffineExpr lhs, rhs;
  if (symbolic_expr.GetLHS() && symbolic_expr.GetRHS()) {
    lhs = SymbolicExprToAffineExpr(symbolic_expr.GetLHS(), num_dims);
    rhs = SymbolicExprToAffineExpr(symbolic_expr.GetRHS(), num_dims);
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

llvm::SmallVector<SymbolicExpr> AffineExprsToSymbolicExprs(
    llvm::ArrayRef<mlir::AffineExpr> affine_exprs, int num_dims) {
  llvm::SmallVector<SymbolicExpr> symbolic_exprs;
  symbolic_exprs.reserve(affine_exprs.size());
  for (mlir::AffineExpr expr : affine_exprs) {
    symbolic_exprs.push_back(AffineExprToSymbolicExpr(expr, num_dims));
  }
  return symbolic_exprs;
}

SymbolicExpr AffineExprToSymbolicExpr(mlir::AffineExpr affine_expr,
                                      int num_dims) {
  if (!affine_expr) {
    return SymbolicExpr();
  }
  mlir::MLIRContext* context = affine_expr.getContext();
  switch (affine_expr.getKind()) {
    case mlir::AffineExprKind::Constant:
      return CreateSymbolicConstant(
          mlir::cast<mlir::AffineConstantExpr>(affine_expr).getValue(),
          context);
    case mlir::AffineExprKind::DimId:
      return CreateSymbolicVariable(
          mlir::cast<mlir::AffineDimExpr>(affine_expr).getPosition(), context);
    case mlir::AffineExprKind::SymbolId:
      return CreateSymbolicVariable(
          mlir::cast<mlir::AffineSymbolExpr>(affine_expr).getPosition() +
              num_dims,
          context);
    case mlir::AffineExprKind::Add: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(affine_expr);
      return AffineExprToSymbolicExpr(bin_op.getLHS(), num_dims) +
             AffineExprToSymbolicExpr(bin_op.getRHS(), num_dims);
    }
    case mlir::AffineExprKind::Mul: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(affine_expr);
      return AffineExprToSymbolicExpr(bin_op.getLHS(), num_dims) *
             AffineExprToSymbolicExpr(bin_op.getRHS(), num_dims);
    }
    case mlir::AffineExprKind::FloorDiv: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(affine_expr);
      return AffineExprToSymbolicExpr(bin_op.getLHS(), num_dims)
          .floorDiv(AffineExprToSymbolicExpr(bin_op.getRHS(), num_dims));
    }
    case mlir::AffineExprKind::CeilDiv: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(affine_expr);
      return AffineExprToSymbolicExpr(bin_op.getLHS(), num_dims)
          .ceilDiv(AffineExprToSymbolicExpr(bin_op.getRHS(), num_dims));
    }
    case mlir::AffineExprKind::Mod: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(affine_expr);
      return AffineExprToSymbolicExpr(bin_op.getLHS(), num_dims) %
             AffineExprToSymbolicExpr(bin_op.getRHS(), num_dims);
    }
  }
}

SymbolicMap AffineMapToSymbolicMap(const mlir::AffineMap& affine_map) {
  if (!affine_map) {
    return SymbolicMap();
  }
  llvm::SmallVector<SymbolicExpr> results;
  results.reserve(affine_map.getNumResults());
  int num_dims = affine_map.getNumDims();
  for (mlir::AffineExpr expr : affine_map.getResults()) {
    results.push_back(AffineExprToSymbolicExpr(expr, num_dims));
  }
  return SymbolicMap::Get(affine_map.getContext(), num_dims,
                          affine_map.getNumSymbols(), results);
}

mlir::AffineMap SymbolicMapToAffineMap(SymbolicMap symbolic_map) {
  if (!symbolic_map) {
    return mlir::AffineMap();
  }
  int num_dims = symbolic_map.GetNumDims();
  int num_symbols = symbolic_map.GetNumSymbols();
  llvm::SmallVector<mlir::AffineExpr> results;
  results.reserve(symbolic_map.GetNumResults());
  for (SymbolicExpr expr : symbolic_map.GetResults()) {
    mlir::AffineExpr affine_expr =
        SymbolicExprToAffineExpr(expr, symbolic_map.GetNumDims());
    if (!affine_expr) {
      // Conversion failed.
      return mlir::AffineMap();
    }
    results.push_back(affine_expr);
  }

  return mlir::AffineMap::get(num_dims, num_symbols, results,
                              symbolic_map.GetContext());
}

llvm::MapVector<SymbolicExpr, Interval>
ConvertAffineConstraintsToSymbolicConstraints(
    const llvm::MapVector<mlir::AffineExpr, Interval>& affine_constraints,
    int num_dims) {
  llvm::MapVector<SymbolicExpr, Interval> symbolic_constraints;
  for (const auto& [affine_expr, interval] : affine_constraints) {
    SymbolicExpr expr = AffineExprToSymbolicExpr(affine_expr, num_dims);
    symbolic_constraints[expr] = interval;
  }
  return symbolic_constraints;
}

}  // namespace xla
