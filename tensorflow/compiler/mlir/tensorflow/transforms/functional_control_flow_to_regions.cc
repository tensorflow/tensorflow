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

#include <cassert>
#include <memory>
#include <vector>

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
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"

#define DEBUG_TYPE "tf-functional-cf-to-region"

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_FUNCTIONALCONTROLFLOWTOREGIONSPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct FunctionalControlFlowToRegions
    : public impl::FunctionalControlFlowToRegionsPassBase<
          FunctionalControlFlowToRegions> {
  FunctionalControlFlowToRegions() = default;
  explicit FunctionalControlFlowToRegions(bool allow_passthrough_args)
      : FunctionalControlFlowToRegionsPassBase(
            FunctionalControlFlowToRegionsPassOptions{allow_passthrough_args}) {
  }
  void runOnOperation() override;
};

// Creates a call to function `func` in region `caller_region`. Use `args` as
// the call arguments, and terminate the region with a yield. The arguments are
// cast to the required type before the call. `use_region_args` control whether
// the input arguments are used as is (for IfOp) or block arguments of the same
// type as the input arguments are created and then used as call arguments (for
// While).
YieldOp CreateCall(Operation* op, func::FuncOp func, Region& caller_region,
                   ValueRange args, bool use_region_args,
                   bool forward_block_args) {
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

  auto results = call.getResults();
  auto block_args = entry->getArguments();
  auto yield_args = std::vector<Value>();
  yield_args.insert(yield_args.end(), results.begin(), results.end());
  if (forward_block_args) {
    yield_args.insert(yield_args.end(), block_args.begin(), block_args.end());
  }
  return builder.create<YieldOp>(loc, yield_args);
}

// Converts the condition for an IfOp/WhileOp to a boolean value.
Value ConvertConditionToBoolean(Operation* op, Value cond) {
  if (auto ranked_type = mlir::dyn_cast<RankedTensorType>(cond.getType()))
    if (ranked_type.getRank() == 0 &&
        ranked_type.getElementType().isSignlessInteger(1))
      return cond;

  OpBuilder builder(op);
  Value to_bool = builder.create<TF::ToBoolOp>(op->getLoc(), cond);
  CopyDeviceAndUnderscoredAttributes(op, to_bool.getDefiningOp());
  return to_bool;
}

// Transform a functional IfOp to a region based IfRegionOp.
LogicalResult ConvertIfOp(IfOp if_op) {
  Value cond = ConvertConditionToBoolean(if_op, if_op.getCond());
  OpBuilder builder(if_op);
  auto if_region = builder.create<TF::IfRegionOp>(
      if_op.getLoc(), if_op.getResultTypes(), cond, if_op.getIsStateless(),
      builder.getStringAttr(if_op.then_function().getName()),
      builder.getStringAttr(if_op.else_function().getName()));
  CopyDeviceAndUnderscoredAttributes(if_op, if_region);

  CreateCall(if_op, if_op.then_function(),
             /*caller_region=*/if_region.getThenBranch(), if_op.getInput(),
             /*use_region_args=*/false,
             /*forward_block_args=*/false);
  CreateCall(if_op, if_op.else_function(),
             /*caller_region=*/if_region.getElseBranch(), if_op.getInput(),
             /*use_region_args=*/false,
             /*forward_block_args=*/false);
  if_op.replaceAllUsesWith(if_region.getResults());
  if_op.erase();
  return success();
}

LogicalResult ConvertCaseOp(CaseOp case_op) {
  OpBuilder builder(case_op);
  auto case_region = builder.create<TF::CaseRegionOp>(
      case_op.getLoc(), case_op.getResultTypes(), case_op.getBranchIndex(),
      case_op.getIsStateless(), case_op.getBranches().size());
  CopyDeviceAndUnderscoredAttributes(case_op, case_region);

  for (const auto& item : llvm::enumerate(case_region.getBranches())) {
    CreateCall(case_op, case_op.branch_function(item.index()),
               /*caller_region=*/item.value(), case_op.getInput(),
               /*use_region_args=*/false,
               /*forward_block_args=*/false);
  }
  case_op.replaceAllUsesWith(case_region.getResults());
  case_op.erase();
  return success();
}

LogicalResult ConvertWhileOp(WhileOp while_op, bool allow_passthrough_args) {
  auto while_region = OpBuilder(while_op).create<TF::WhileRegionOp>(
      while_op.getLoc(), while_op.getResultTypes(), while_op.getInput(),
      while_op.getParallelIterations(), while_op.getIsStateless(),
      while_op.getShapeInvariant());
  CopyDeviceAndUnderscoredAttributes(while_op, while_region);

  YieldOp cond_yield =
      CreateCall(while_op, while_op.cond_function(),
                 /*caller_region=*/while_region.getCond(), while_op.getInput(),
                 /*use_region_args=*/true,
                 /*forward_block_args=*/allow_passthrough_args);
  Value i1_cond =
      ConvertConditionToBoolean(cond_yield, cond_yield.getOperand(0));
  cond_yield.setOperand(0, i1_cond);

  CreateCall(while_op, while_op.body_function(),
             /*caller_region=*/while_region.getBody(), while_op.getInput(),
             /*use_region_args=*/true,
             /*forward_block_args=*/false);
  while_op.replaceAllUsesWith(while_region.getResults());
  while_op.erase();
  return success();
}

LogicalResult ConvertGeneratorDatasetOp(GeneratorDatasetOp generator_op) {
  auto generator_region =
      OpBuilder(generator_op)
          .create<TF::GeneratorDatasetRegionOp>(
              generator_op.getLoc(), generator_op->getResultTypes(),
              generator_op.getInitFuncOtherArgs(),
              generator_op.getNextFuncOtherArgs(),
              generator_op.getFinalizeFuncOtherArgs(),
              generator_op.getOutputTypes(), generator_op.getOutputShapes(),
              generator_op.getMetadata());
  CopyDeviceAndUnderscoredAttributes(generator_op, generator_region);

  func::FuncOp init_function =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          generator_op, generator_op.getInitFunc());
  func::FuncOp next_function =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          generator_op, generator_op.getNextFunc());
  func::FuncOp finalize_function =
      SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          generator_op, generator_op.getFinalizeFunc());

  if (!init_function || !next_function || !finalize_function) {
    return failure();
  }

  CreateCall(generator_op, init_function, generator_region.getInit(),
             generator_region.getInitFuncOtherArgs(),
             /*use_region_args=*/true, /*forward_block_args=*/false);
  CreateCall(generator_op, next_function, generator_region.getNext(),
             generator_region.getNextFuncOtherArgs(),
             /*use_region_args=*/true, /*forward_block_args=*/false);
  CreateCall(generator_op, finalize_function, generator_region.getFinalize(),
             generator_region.getFinalizeFuncOtherArgs(),
             /*use_region_args=*/true, /*forward_block_args=*/false);

  generator_op->replaceAllUsesWith(generator_region->getResults());
  generator_op->erase();

  return success();
}

void FunctionalControlFlowToRegions::runOnOperation() {
  ModuleOp module = getOperation();
  auto result = module.walk([&](Operation* op) {
    if (IfOp if_op = llvm::dyn_cast<IfOp>(op)) {
      if (failed(ConvertIfOp(if_op))) {
        op->emitOpError() << "failed to convert to region form";
        return WalkResult::interrupt();
      }
    } else if (CaseOp case_op = llvm::dyn_cast<CaseOp>(op)) {
      if (failed(ConvertCaseOp(case_op))) {
        op->emitOpError() << "failed to convert to region form";
        return WalkResult::interrupt();
      }
    } else if (auto while_op = llvm::dyn_cast<WhileOp>(op)) {
      if (failed(ConvertWhileOp(while_op, allow_passthrough_args_))) {
        op->emitOpError() << "failed to convert to region form";
        return WalkResult::interrupt();
      }
    } else if (auto generator_op = llvm::dyn_cast<GeneratorDatasetOp>(op)) {
      if (allow_passthrough_args_) {
        if (failed(ConvertGeneratorDatasetOp(generator_op))) {
          op->emitOpError() << "failed to convert to region form";
          return WalkResult::interrupt();
        }
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
std::unique_ptr<OperationPass<ModuleOp>> CreateTFFunctionalControlFlowToRegions(
    bool allow_passthrough_args) {
  return std::make_unique<FunctionalControlFlowToRegions>(
      allow_passthrough_args);
}

}  // namespace TF
}  // namespace mlir
