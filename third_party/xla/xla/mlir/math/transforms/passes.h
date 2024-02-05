/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_MLIR_MATH_TRANSFORMS_PASSES_H_
#define XLA_MLIR_MATH_TRANSFORMS_PASSES_H_

#include <memory>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project

namespace xla {

#define GEN_PASS_DECL_MATHAPPROXIMATIONPASS
#define GEN_PASS_DECL_MATHOPTIMIZATIONPASS
#include "xla/mlir/math/transforms/passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateMathOptimizationPass(bool enable_avx2 = false);

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateMathApproximationPass(llvm::ArrayRef<std::string> oplist = {});

#define GEN_PASS_REGISTRATION
#include "xla/mlir/math/transforms/passes.h.inc"

}  // namespace xla

#endif  // XLA_MLIR_MATH_TRANSFORMS_PASSES_H_
