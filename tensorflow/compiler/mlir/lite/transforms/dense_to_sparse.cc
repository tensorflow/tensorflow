/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This transformation pass convert dense tensor to sparse format.

#include "absl/memory/memory.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

//===----------------------------------------------------------------------===//
// The DenseToSparse Pass.
//
namespace mlir {
namespace TFL {

namespace {

struct DenseToSparse : public PassWrapper<DenseToSparse, FunctionPass> {
  void runOnFunction() override;
};

void DenseToSparse::runOnFunction() {
  FuncOp func = getFunction();
  OpBuilder builder(func);

  func.walk([&](SparseOpInterface sparse_op) {
    const auto& sparse_operands = sparse_op.GetSparseOperands();
    for (const int operand : sparse_operands) {
      auto* op = sparse_op.getOperation();
      const auto& value = op->getOperand(operand);
      builder.setInsertionPoint(op);
      if (auto* inst = value.getDefiningOp()) {
        // Replace defining op with SparseConst or SparseQConst.
        // TODO(yunluli): Implement.
      }

      // TODO(yunluli): Implement.
      bool needs_densify = false;

      if (needs_densify) {
        auto densify = builder.create<DensifyOp>(op->getLoc(), value);
        value.replaceAllUsesWith(densify);
        densify.setOperand(value);
      }
    }
  });
}

}  // namespace

// Creates an instance of the TensorFlow Lite dialect DenseToSparse pass.
std::unique_ptr<OperationPass<FuncOp>> CreateDenseToSparsePass() {
  return absl::make_unique<DenseToSparse>();
}

static PassRegistration<DenseToSparse> pass(
    "tfl-dense-to-sparse", "Convert dense tensor to sparse format.");

}  // namespace TFL
}  // namespace mlir
