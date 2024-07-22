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

#include "xla/service/gpu/fusions/mlir/ir/xla_gpu_attrs.h"

#include <utility>

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

void PrintDimVars(mlir::AsmPrinter& p, llvm::ArrayRef<DimVar> dim_vars) {}

mlir::FailureOr<llvm::SmallVector<DimVar>> ParseDimVars(
    mlir::AsmParser& parser) {
  return mlir::failure();
}

void PrintRangeVars(mlir::AsmPrinter& p, llvm::ArrayRef<RangeVar> range_vars) {}

mlir::FailureOr<llvm::SmallVector<RangeVar>> ParseRangeVars(
    mlir::AsmParser& parser) {
  return mlir::failure();
}

void PrintConstraints(
    mlir::AsmPrinter& p,
    mlir::ArrayRef<std::pair<mlir::AffineExpr, xla::gpu::Interval>>
        range_vars) {}

mlir::FailureOr<
    llvm::SmallVector<std::pair<mlir::AffineExpr, xla::gpu::Interval>>>
ParseConstraints(mlir::AsmParser& parser) {
  return mlir::failure();
}

}  // namespace gpu
}  // namespace xla
