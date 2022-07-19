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
// tf.If -> tf.IfRegion and tf.While -> tf.WhileRegion

#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

#define DEBUG_TYPE "tf-functional-cf-to-region"

namespace mlir {
namespace TF {

namespace {

struct FunctionalControlFlowToRegions
    : public TF::FunctionalControlFlowToRegionsPassBase<
          FunctionalControlFlowToRegions> {
  void runOnOperation() override;
};

// Creates a call to function `func` in region `caller_region`. Use `args` as
// the call arguments, and terminate the region with a yield. The arguments are
// cast to the required type before the call. `use_region_args` control whether
// the input arguments are used as is (for IfOp) or block arguments of the same
// type as the input arguments are created and then used as call arguments (for
// While).
YieldOp CreateCall(Operation* op, func::FuncOp func, Region& caller_region,
                   ValueRange args, bool use_region_args) {
  assert(caller_region.empty() &&
         "Expected empty region for newly created ops");
  OpBuilder builder(caller_region);
  Block* entry = builder.createBlock(&caller_region);

  auto loc = op->getLoc();
  if (use_region_args) {
    auto inputs = func.getFunctionType().getInputs();
    entry->addArguments(inputs, SmallVector<Location>(inputs.size(), loc));
    args = entry->getArguments();
  }
  llvm::SmallVector<Value, 4> casted_args;
  casted_args.reserve(func.getNumArguments());
  for (const auto& ArgAndType : zip(args, func.getFunctionType().getInputs())) {
    Value arg = std::get<0>(ArgAndType);
    Type expected_type = std::get<1>(ArgAndType);
    if (arg.getType() != expected_type) {
      arg = builder.create<CastOp>(loc, expected_type, arg,
                                   /*Truncate=*/builder.getBoolAttr(false));
    }
    casted_args.push_back(arg);
  }
  auto call = builder.create<func::CallOp>(loc, func, casted_args);
  return builder.create<YieldOp>(loc, call.getResults());
}

// Converts the condition for an IfOp/WhileOp to a boolean value.
Value ConvertConditionToBoolean(Operation* op, Value cond) {
  if (auto ranked_type = cond.getType().dyn_cast<RankedTensorType>())
    if (ranked_type.getRank() == 0 &&
        ranked_type.getElementType().isSignlessInteger(1))
      return cond;

  OpBuilder builder(op);
  return builder.create<TF::ToBoolOp>(op->getLoc(), cond);
}

// Transform a functional IfOp to a region based IfRegionOp.
LogicalResult ConvertIfOp(IfOp if_op) {
  Value cond = ConvertConditionToBoolean(if_op, if_op.cond());
  OpBuilder builder(if_op);
  auto if_region = builder.create<TF::IfRegionOp>(
      if_op.getLoc(), if_op.getResultTypes(), cond, if_op.is_stateless(),
      builder.getStringAttr(if_op.then_function().getName()),
      builder.getStringAttr(if_op.else_function().getName()));
  CopyDeviceAndUnderscoredAttributes(if_op, if_region);

  CreateCall(if_op, if_op.then_function(),
             /*caller_region=*/if_region.then_branch(), if_op.input(),
             /*use_region_args=*/false);
  CreateCall(if_op, if_op.else_function(),
             /*caller_region=*/if_region.else_branch(), if_op.input(),
             /*use_region_args=*/false);
  if_op.replaceAllUsesWith(if_region.getResults());
  if_op.erase();
  return success();
}

LogicalResult ConvertWhileOp(WhileOp while_op) {
  auto while_region = OpBuilder(while_op).create<TF::WhileRegionOp>(
      while_op.getLoc(), while_op.getResultTypes(), while_op.input(),
      while_op.parallel_iterations(), while_op.is_stateless(),
      while_op.shape_invariant());
  CopyDeviceAndUnderscoredAttributes(while_op, while_region);

  YieldOp cond_yield =
      CreateCall(while_op, while_op.cond_function(),
                 /*caller_region=*/while_region.cond(), while_op.input(),
                 /*use_region_args=*/true);
  Value i1_cond =
      ConvertConditionToBoolean(cond_yield, cond_yield.getOperand(0));
  cond_yield.setOperand(0, i1_cond);

  CreateCall(while_op, while_op.body_function(),
             /*caller_region=*/while_region.body(), while_op.input(),
             /*use_region_args=*/true);
  while_op.replaceAllUsesWith(while_region.getResults());
  while_op.erase();
  return success();
}

void FunctionalControlFlowToRegions::runOnOperation() {
  ModuleOp module = getOperation();
  auto result = module.walk([](Operation* op) {
    if (IfOp if_op = llvm::dyn_cast<IfOp>(op)) {
      if (failed(ConvertIfOp(if_op))) {
        op->emitOpError() << "failed to convert to region form";
        return WalkResult::interrupt();
      }
    } else if (auto while_op = llvm::dyn_cast<WhileOp>(op)) {
      if (failed(ConvertWhileOp(while_op))) {
        op->emitOpError() << "failed to convert to region form";
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

}  // namespace TF
}  // namespace mlir
