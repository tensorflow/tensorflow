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

#include <vector>

#include "llvm/Support/Casting.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"

namespace mlir {
namespace TFL {
namespace {

struct GetArithmeticCountPass
    : public PassWrapper<GetArithmeticCountPass, FunctionPass> {
  void runOnFunction() override;

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-get-arithmetic-count";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Calculate arithmetic count for tfl operations.";
  }
};

void GetArithmeticCountPass::runOnFunction() {
  auto func = getFunction();
  OpBuilder builder(func);
  func->walk([&](TflArithmeticCountOpInterface arithmetic_count_op) {
    Operation* op = arithmetic_count_op.getOperation();
    int64_t arithmetic_count = arithmetic_count_op.GetArithmeticCount(op);
    auto attr =
        builder.getIntegerAttr(builder.getIntegerType(64), arithmetic_count);
    op->setAttr("_arithmetic_count", attr);
  });
}

}  // namespace

/// Creates an instance of the TensorFlow Lite dialect GetArithmeticCount
/// pass.
std::unique_ptr<OperationPass<FuncOp>> CreateGetArithmeticCountPass() {
  return std::make_unique<GetArithmeticCountPass>();
}

static PassRegistration<GetArithmeticCountPass> pass;

}  // namespace TFL
}  // namespace mlir
