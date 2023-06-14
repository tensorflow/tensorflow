/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/python/ifrt/ir/transforms/built_in_spmd_expansions.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/xla/python/ifrt/ir/transforms/spmd_expanders/unimplemented_ifrt_spmd_expander.h"

namespace xla {
namespace ifrt {
namespace {

void AttachFuncDialectOpsSpmdExpansions(mlir::MLIRContext* context,
                                        mlir::func::FuncDialect* dialect) {
  // TODO(b/261623129): Implement the SPMD expander for func::ReturnOp.
  mlir::func::ReturnOp::attachInterface<
      UnimplementedIfrtSpmdExpander<mlir::func::ReturnOp>>(*context);
}

}  // namespace

void AttachBuiltInSpmdExpansions(mlir::DialectRegistry& registry) {
  registry.addExtension(AttachFuncDialectOpsSpmdExpansions);
}

}  // namespace ifrt
}  // namespace xla
