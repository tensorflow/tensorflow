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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/service/gpu/model/experimental/symbolic_expr.h"
#include "xla/service/gpu/model/experimental/symbolic_map.h"

namespace xla {
namespace gpu {

// Helper function to convert mlir::AffineExpr to xla::gpu::SymbolicExpr.
SymbolicExpr AffineToSymbolicExpr(::mlir::AffineExpr affine_expr,
                                  SymbolicExprContext* context, int num_dims);

// Converts an mlir::AffineMap to xla::gpu::SymbolicMap.
SymbolicMap AffineMapToSymbolicMap(const mlir::AffineMap& affine_map,
                                   SymbolicExprContext* context);

// Converts xla::gpu::SymbolicMap to an mlir::AffineMap.
// Returns a null AffineMap if the conversion is not possible.
mlir::AffineMap SymbolicMapToAffineMap(SymbolicMap symbolic_map,
                                       mlir::MLIRContext* context);

// Converts AffineExpr-based constraints to SymbolicExpr-based constraints.
llvm::MapVector<SymbolicExpr, Interval>
ConvertAffineConstraintsToSymbolicConstraints(
    const llvm::MapVector<mlir::AffineExpr, Interval>& affine_constraints,
    SymbolicExprContext* context, int num_dims);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_EXPERIMENTAL_SYMBOLIC_MAP_CONVERTER_H_
