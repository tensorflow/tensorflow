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

// This transformation pass transforms functional control flow operations in the
// TensorFlow dialect to their region based counterparts, i.e.,
// tf.If -> tf.IfRegion

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {

namespace {

struct FunctionalControlFlowToRegions
    : public PassWrapper<FunctionalControlFlowToRegions,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// Create a call to function `fn` with arguments `args` and return the CallOp.
// The arguments are cast to the required type before the call.
CallOp CreateCall(Location loc, Operation::operand_range args, FuncOp fn,
                  OpBuilder* builder) {
  FunctionType fn_type = fn.getType();
  llvm::SmallVector<Value, 4> operands;
  int num_operands = fn_type.getNumInputs();
  operands.reserve(num_operands);
  for (const auto& ArgAndType : zip(args, fn_type.getInputs())) {
    Value arg = std::get<0>(ArgAndType);
    Type expected_type = std::get<1>(ArgAndType);
    if (arg.getType() != expected_type) {
      arg = builder->create<CastOp>(loc, expected_type, arg,
                                    /*Truncate=*/builder->getBoolAttr(false));
    }
    operands.push_back(arg);
  }
  return builder->create<CallOp>(loc, fn, operands);
}

// Transform a functional IfOp to a region based IfRegionOp.
LogicalResult ConvertIfOp(IfOp if_op) {
  auto if_region = OpBuilder(if_op).create<TF::IfRegionOp>(
      if_op.getLoc(), if_op.getResultTypes(), if_op.cond(),
      if_op.is_stateless());

  // Insert call to the given function into the 'region'.
  auto create_region_with_call = [&if_op](FlatSymbolRefAttr symbol,
                                          Region& region) {
    OpBuilder builder(region);
    builder.createBlock(&region);
    auto func = if_op.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
        symbol.getValue());
    auto call = CreateCall(if_op.getLoc(), if_op.input(), func, &builder);
    builder.create<YieldOp>(if_op.getLoc(), call.getResults());
  };

  create_region_with_call(if_op.then_branchAttr(), if_region.then_branch());
  create_region_with_call(if_op.else_branchAttr(), if_region.else_branch());

  if_op.replaceAllUsesWith(if_region.getResults());
  if_op.erase();
  return success();
}

void FunctionalControlFlowToRegions::runOnOperation() {
  ModuleOp module = getOperation();
  auto result = module.walk([](Operation* op) {
    if (IfOp if_op = llvm::dyn_cast<IfOp>(op)) {
      if (failed(ConvertIfOp(if_op))) {
        if_op.emitOpError() << " failed to convert to region form";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return signalPassFailure();
}
}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFFunctionalControlFlowToRegions() {
  return std::make_unique<FunctionalControlFlowToRegions>();
}

static PassRegistration<FunctionalControlFlowToRegions> pass(
    "tf-functional-control-flow-to-regions",
    "Transform functional control flow Ops to Region based counterparts");

}  // namespace TF
}  // namespace mlir
