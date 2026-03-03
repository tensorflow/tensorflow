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

#ifndef XLA_HLO_ANALYSIS_SYMBOLIC_MAP_CONVERTER_H_
#define XLA_HLO_ANALYSIS_SYMBOLIC_MAP_CONVERTER_H_

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/analysis/symbolic_map.h"

namespace xla {

// Converts an mlir::AffineExpr to xla::SymbolicExpr.
SymbolicExpr AffineExprToSymbolicExpr(::mlir::AffineExpr affine_expr,
                                      int num_dims);

// Converts a list of mlir::AffineExpr to xla::SymbolicExpr.
llvm::SmallVector<SymbolicExpr> AffineExprsToSymbolicExprs(
    llvm::ArrayRef<mlir::AffineExpr> affine_exprs, int num_dims);

// Converts an xla::SymbolicExpr to an mlir::AffineExpr.
mlir::AffineExpr SymbolicExprToAffineExpr(SymbolicExpr symbolic_expr,
                                          int num_dims);

// Converts an mlir::AffineMap to xla::SymbolicMap.
SymbolicMap AffineMapToSymbolicMap(const mlir::AffineMap& affine_map);

// Converts xla::SymbolicMap to an mlir::AffineMap.
// Returns a null AffineMap if the conversion is not possible.
mlir::AffineMap SymbolicMapToAffineMap(SymbolicMap symbolic_map);

// Converts AffineExpr-based constraints to SymbolicExpr-based constraints.
llvm::MapVector<SymbolicExpr, Interval>
ConvertAffineConstraintsToSymbolicConstraints(
    const llvm::MapVector<mlir::AffineExpr, Interval>& affine_constraints,
    int num_dims);

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_SYMBOLIC_MAP_CONVERTER_H_
