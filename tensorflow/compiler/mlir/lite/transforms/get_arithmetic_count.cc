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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

struct GetArithmeticCountPass
    : public GetArithmeticCountPassBase<GetArithmeticCountPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GetArithmeticCountPass)

  void runOnOperation() override;
};

void GetArithmeticCountPass::runOnOperation() {
  auto func = getOperation();
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
std::unique_ptr<OperationPass<func::FuncOp>> CreateGetArithmeticCountPass() {
  return std::make_unique<GetArithmeticCountPass>();
}

}  // namespace TFL
}  // namespace mlir
