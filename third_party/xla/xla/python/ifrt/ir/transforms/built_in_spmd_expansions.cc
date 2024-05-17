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

#include "xla/python/ifrt/ir/transforms/built_in_spmd_expansions.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/python/ifrt/ir/transforms/spmd_expanders/noop_ifrt_spmd_expander.h"
#include "xla/python/ifrt/ir/transforms/spmd_expanders/terminator_ifrt_spmd_expander.h"

namespace xla {
namespace ifrt {
namespace {

void AttachFuncDialectOpsSpmdExpansions(mlir::MLIRContext* context,
                                        mlir::func::FuncDialect* dialect) {
  mlir::func::ReturnOp::attachInterface<
      TerminatorIfrtSpmdExpander<mlir::func::ReturnOp>>(*context);
  mlir::func::CallOp::attachInterface<NoOpIfrtSpmdExpander<mlir::func::CallOp>>(
      *context);
}

}  // namespace

void AttachBuiltInSpmdExpansions(mlir::DialectRegistry& registry) {
  registry.addExtension(AttachFuncDialectOpsSpmdExpansions);
}

}  // namespace ifrt
}  // namespace xla
