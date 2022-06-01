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

// This file implements logic for flattening tuples in HLO ops.

#include <cassert>
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

// Calculates the flatten types of a value.
void FlattenTupleType(Value value, llvm::SmallVectorImpl<Type> &types) {
  if (!value.getType().isa<TupleType>()) {
    types.push_back(value.getType());
    return;
  }

  // This function doesn't handle nested tuple.
  auto tupleType = value.getType().cast<TupleType>();
  types.append(tupleType.begin(), tupleType.end());
}

// Flattens value into flatten_values.
void FlattenTupleValue(OpBuilder &builder, Location loc, Value value,
                       llvm::SmallVectorImpl<Value> &flatten_values) {
  if (!value.getType().isa<TupleType>()) {
    flatten_values.push_back(value);
    return;
  }

  // This function doesn't handle nested tuple.
  int flattenIdx = 0;
  auto tupleType = value.getType().cast<TupleType>();
  for (auto childType : tupleType.getTypes()) {
    auto getTupleOp = builder.create<mhlo::GetTupleElementOp>(
        loc, childType, value, builder.getI32IntegerAttr(flattenIdx++));
    flatten_values.push_back(getTupleOp);
  }
}

// FlattenTupleValue and CreateTupleValue is a pair of functions to create and
// flatten tuples in the exact same order. CreateTupleValue returns the result
// of the root TupleOp or given value if the type is not TupleType.
Value CreateTupleValue(OpBuilder &builder, Location loc,
                       const llvm::ArrayRef<Value> &flatten_values,
                       Type tuple_type) {
  if (!tuple_type.isa<TupleType>()) {
    assert(flatten_values.size() == 1);
    return flatten_values[0];
  }

  assert(tuple_type.cast<TupleType>().getTypes().size() ==
         flatten_values.size());
  return builder.create<mhlo::TupleOp>(loc, flatten_values);
}

// Flattens the tuples in the region's arguments and returning values.
void FlattenTupleInRegion(Region &region, PatternRewriter &rewriter) {
  Location loc = region.getLoc();
  OpBuilder regionOpBuilder(region);

  // Flatten tuples in arguments. The order of arguments must match the order
  // in FlattenTupleType, FlattenTupleValue and CreateTupleValue.
  const int originalNumArgs = region.getNumArguments();
  for (int argIdx : llvm::seq<int>(0, originalNumArgs)) {
    auto argument = region.getArgument(argIdx);

    // Adds new arguments to replace the tuple argument.
    llvm::SmallVector<Type, 4> newTypes;
    llvm::SmallVector<Value, 4> newArguments;
    FlattenTupleType(argument, newTypes);
    for (auto type : newTypes) {
      newArguments.push_back(region.addArgument(type, loc));
    }

    // Replaces uses of the replacing argument.
    auto tupleValue = CreateTupleValue(regionOpBuilder, loc, newArguments,
                                       argument.getType());
    argument.replaceAllUsesWith(tupleValue);
  }
  // Removes old tuple arguments.
  for (int argIdx = originalNumArgs - 1; argIdx >= 0; --argIdx) {
    region.eraseArgument(argIdx);
  }

  // Flatten tuples in results.
  for (auto &block : region.getBlocks()) {
    Operation *terminator = block.getTerminator();
    assert(isa<mhlo::ReturnOp>(terminator));
    auto returnOp = llvm::cast<mhlo::ReturnOp>(terminator);

    // Creates a new ReturnOp with flatten values.
    OpBuilder builder(returnOp);
    llvm::SmallVector<Value, 4> results;
    for (auto operand : returnOp.getOperands()) {
      FlattenTupleValue(builder, returnOp.getLoc(), operand, results);
    }
    builder.create<mhlo::ReturnOp>(loc, results);
    rewriter.eraseOp(returnOp);
  }
}

// Applies tuple flattening patterns to given target. This helper
// function is used to flatten ops recursively.
template <typename T>
void ApplyFlatteningTuplePatterns(T target, MLIRContext *context);

struct FlattenWhileOp : public RewritePattern {
  explicit FlattenWhileOp(MLIRContext *context)
      : RewritePattern(mhlo::WhileOp::getOperationName(), 1, context,
                       {mhlo::WhileOp::getOperationName(),
                        mhlo::TupleOp::getOperationName(),
                        mhlo::GetTupleElementOp::getOperationName()}),
        context(context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto whileOp = cast<mhlo::WhileOp>(op);
    // HLO WhileOp should have two regions: cond and body.
    if (whileOp->getNumRegions() != 2) return failure();

    // Operands of mhlo::WhileOp can be a variadic list of tensors and
    // tuples. Tuples need to be flattened in order to be used in
    // TF::WhileOp. Note that in WhileOp, operand and result types are
    // always the same.
    OpBuilder builder(whileOp);
    llvm::SmallVector<Value, 4> flattenedOperands;
    llvm::SmallVector<Type, 4> flattenedOperandTypes;
    for (auto operand : whileOp->getOperands()) {
      FlattenTupleType(operand, flattenedOperandTypes);
      FlattenTupleValue(builder, whileOp.getLoc(), operand, flattenedOperands);
    }

    // The applyPatternsAndFoldGreedily can't be called on child regions, so
    // creates temporary regions to apply flattening rules recursively.
    auto module = whileOp->getParentOfType<ModuleOp>();
    BlockAndValueMapping mapping;
    Region newCond(module);
    whileOp.cond().cloneInto(&newCond, mapping);
    Region newBody(module);
    whileOp.body().cloneInto(&newBody, mapping);

    // Flattens the tuples in child regions.
    FlattenTupleInRegion(newCond, rewriter);
    FlattenTupleInRegion(newBody, rewriter);

    // There might be WhileOp in child regions, flattens tuple in them too.
    ApplyFlatteningTuplePatterns<MutableArrayRef<Region>>(newCond, context);
    ApplyFlatteningTuplePatterns<MutableArrayRef<Region>>(newBody, context);

    // Creates a new mhlo::WhileOp with no tuples.
    auto newWhile = builder.create<mhlo::WhileOp>(
        whileOp.getLoc(), flattenedOperandTypes, flattenedOperands);
    newCond.cloneInto(&newWhile.cond(), mapping);
    newBody.cloneInto(&newWhile.body(), mapping);

    // Replaces uses of the old WhileOp.
    auto newResultIter = newWhile.result_begin();
    for (auto oldResult : whileOp.getResults()) {
      llvm::SmallVector<Type, 4> flattenedTypes;
      FlattenTupleType(oldResult, flattenedTypes);
      llvm::SmallVector<Value, 4> flattenedResults;
      while (flattenedResults.size() < flattenedTypes.size()) {
        assert(newResultIter != newWhile->result_end());
        flattenedResults.push_back(*newResultIter++);
      }
      auto tupleValue = CreateTupleValue(builder, whileOp.getLoc(),
                                         flattenedResults, oldResult.getType());
      oldResult.replaceAllUsesWith(tupleValue);
    }
    rewriter.eraseOp(whileOp);
    return success();
  }

 private:
  MLIRContext *context;
};

template <typename T>
void ApplyFlatteningTuplePatterns(T target, MLIRContext *context) {
  RewritePatternSet patterns(context);
  patterns.add<FlattenWhileOp>(context);
  (void)applyPatternsAndFoldGreedily(target, std::move(patterns));
}

class FlattenTuplePass : public FlattenTuplePassBase<FlattenTuplePass> {
 public:
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ApplyFlatteningTuplePatterns(getOperation(), ctx);
  }
};
}  // end namespace

static PassRegistration<FlattenTuplePass> pass;

std::unique_ptr<OperationPass<func::FuncOp>> createFlattenTuplePass() {
  return std::make_unique<FlattenTuplePass>();
}

}  // end namespace mhlo
}  // end namespace mlir
