/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/assign_op_key.h"

#include <stdint.h>

#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/constants.h"
#include "tensorflow/compiler/mlir/tfrt/transforms/mlrt/util.h"

namespace tensorflow {
namespace mlrt_compiler {
namespace {

class AssignOpKeyPass
    : public mlir::PassWrapper<AssignOpKeyPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
 public:
  AssignOpKeyPass() = default;
  AssignOpKeyPass& operator=(const AssignOpKeyPass&) = delete;

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignOpKeyPass)

 private:
  llvm::StringRef getArgument() const final { return "tf-mlrt-assign-op-key"; }
  llvm::StringRef getDescription() const final {
    return "tf-mlrt-assign-op-key";
  }

  void runOnOperation() override;
};

void AssignOpKeyPass::runOnOperation() {
  auto module = getOperation();
  mlir::OpBuilder builder(module);

  int32_t op_key = 0;
  module.walk([&builder, &op_key](mlir::Operation* op) mutable {
    if (UseFallback(op)) {
      op->setAttr(tensorflow::tfrt_compiler::kOpKeyAttrName,
                  builder.getI32IntegerAttr(op_key));
      op_key++;
    }
  });
}

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> CreateAssignOpKeyPass() {
  return std::make_unique<AssignOpKeyPass>();
}

}  // namespace mlrt_compiler
}  // namespace tensorflow
