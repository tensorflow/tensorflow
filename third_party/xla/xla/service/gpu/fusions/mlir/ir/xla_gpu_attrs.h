/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_FUSIONS_MLIR_IR_XLA_GPU_ATTRS_H_
#define XLA_SERVICE_GPU_FUSIONS_MLIR_IR_XLA_GPU_ATTRS_H_

#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

// Custom printer to print an array of DimVar.
void PrintDimVars(mlir::AsmPrinter& p, mlir::ArrayRef<DimVar> dim_vars);

// Custom parser to parse an array of DimVar.
mlir::FailureOr<llvm::SmallVector<DimVar>> ParseDimVars(
    mlir::AsmParser& parser);

// Custom printer to print an array of RangeVar.
void PrintRangeVars(mlir::AsmPrinter& p, mlir::ArrayRef<RangeVar> range_vars);

// Custom parser to parse an array of RangeVar.
mlir::FailureOr<llvm::SmallVector<RangeVar>> ParseRangeVars(
    mlir::AsmParser& parser);

// Custom printer to print constraints.
void PrintConstraints(
    mlir::AsmPrinter& p,
    mlir::ArrayRef<::std::pair<::mlir::AffineExpr, Interval>> range_vars);

// Custom parser to parse constraints.
mlir::FailureOr<llvm::SmallVector<::std::pair<::mlir::AffineExpr, Interval>>>
ParseConstraints(mlir::AsmParser& parser);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_MLIR_IR_XLA_GPU_ATTRS_H_
