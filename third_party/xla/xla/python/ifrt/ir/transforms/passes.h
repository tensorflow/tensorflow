/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_
#define XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_

#include <memory>

#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project

namespace xla {
namespace ifrt {

#define GEN_PASS_DECL
#include "xla/python/ifrt/ir/transforms/passes.h.inc"  // IWYU pragma: export

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateSpmdExpandableInterfaceVerificationPass(
    SpmdExpandableInterfaceVerificationPassOptions options = {});

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateSpmdExpansionPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtDuplicatedCalleeEliminationPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateIfrtVerifyShardingSpecifiedPass();

// Generated definitions. This should be placed after all Pass creations.
#define GEN_PASS_REGISTRATION
#include "xla/python/ifrt/ir/transforms/passes.h.inc"  // IWYU pragma: export

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_IR_TRANSFORMS_PASSES_H_
