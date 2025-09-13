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

#ifndef XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_MAP_CONVERTER_H_
#define XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_MAP_CONVERTER_H_

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"

namespace xla {
namespace gpu {

// Converts an mlir::AffineMap to a list of xla::gpu::SymbolicExpr.
llvm::SmallVector<SymbolicExpr> AffineMapToSymbolicExprs(
    const mlir::AffineMap& affine_map, SymbolicExprContext* context);

// Converts a list of xla::gpu::SymbolicExpr to an mlir::AffineMap.
// Returns a null AffineMap if the conversion is not possible.
mlir::AffineMap SymbolicExprsToAffineMap(
    const llvm::SmallVector<SymbolicExpr>& symbolic_exprs,
    mlir::MLIRContext* context, int num_dims, int num_symbols);

// Converts AffineExpr-based constraints to SymbolicExpr-based constraints.
llvm::MapVector<SymbolicExpr, Interval>
ConvertAffineConstraintsToSymbolicConstraints(
    const llvm::MapVector<mlir::AffineExpr, Interval>& affine_constraints,
    SymbolicExprContext* context, int num_dims);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_MAP_CONVERTER_H_
