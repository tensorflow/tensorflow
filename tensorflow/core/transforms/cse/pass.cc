/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/transforms/cse/pass.h"

#include <memory>

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"

namespace mlir {
namespace tfg {
namespace {

#define GEN_PASS_DEF_CSEPASS
#include "tensorflow/core/transforms/passes.h.inc"

class CSEPass : public impl::CSEPassBase<CSEPass> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
    dialect_ = context->getOrLoadDialect<TFGraphDialect>();
    return success();
  }
  void runOnOperation() override;

 private:
  /// The cached TFG dialect instance.
  TFGraphDialect *dialect_;
};
}  // namespace

void CSEPass::runOnOperation() {
  GraphFuncOp func = getOperation();

  // Strip and save operation names.
  DenseMap<Operation *, Attribute> op_names;
  func.walk([&](Operation *op) {
    if (Attribute name = op->removeAttr(dialect_->getNameAttrIdentifier())) {
      op_names.insert({op, name});
    }
  });

  // Run a nested CSE pass.
  OpPassManager nested_manager(func->getName());
  nested_manager.addPass(createCSEPass());
  if (failed(runPipeline(nested_manager, func))) {
    return signalPassFailure();
  }

  // Re-assign names to any remaining operations.
  func.walk([&](Operation *op) {
    if (Attribute name = op_names.lookup(op)) {
      op->setAttr(dialect_->getNameAttrIdentifier(), name);
    }
  });
}

std::unique_ptr<Pass> CreateCSEPass() { return std::make_unique<CSEPass>(); }
}  // namespace tfg
}  // namespace mlir
