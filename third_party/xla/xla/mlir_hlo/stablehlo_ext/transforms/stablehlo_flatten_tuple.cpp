/* Copyright 2025 The OpenXLA Authors.

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
#include <memory>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo_ext/transforms/passes.h"  // NOLINT: Used in passes.h.inc

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_STABLEHLOFLATTENTUPLEPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

// Calculates the flatten types of a value.
void flattenTupleType(Value value, llvm::SmallVectorImpl<Type> &types) {
  if (!mlir::isa<TupleType>(value.getType())) {
    types.push_back(value.getType());
    return;
  }

  // This function doesn't handle nested tuple.
  auto tupleType = mlir::cast<TupleType>(value.getType());
  types.append(tupleType.begin(), tupleType.end());
}

// FlattenTupleValue and CreateTupleValue is a pair of functions to create and
// flatten tuples in the exact same order. CreateTupleValue returns the result
// of the root TupleOp or given value if the type is not TupleType.
Value createTupleValue(OpBuilder &builder, Location loc,
                       ValueRange flattenValues, Type tupleType) {
  if (!mlir::isa<TupleType>(tupleType)) {
    assert(flattenValues.size() == 1);
    return flattenValues[0];
  }

  assert(mlir::cast<TupleType>(tupleType).getTypes().size() ==
         flattenValues.size());
  return builder.create<stablehlo::TupleOp>(loc, flattenValues);
}

void flattenTupleValue(OpBuilder &builder, Location loc, Value value,
                       llvm::SmallVectorImpl<Value> &flattenedValues) {
  auto tupleType = mlir::dyn_cast<TupleType>(value.getType());
  if (!tupleType) {
    flattenedValues.push_back(value);
    return;
  }
  int flattenIdx = 0;
  for (auto innerType : tupleType.getTypes()) {
    auto innerValue = builder.create<stablehlo::GetTupleElementOp>(
        loc, innerType, value, builder.getI32IntegerAttr(flattenIdx++));
    flattenTupleValue(builder, loc, innerValue, flattenedValues);
  }
}

struct FlattenCustomCallOp : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    bool flattenResult = op->getNumResults() == 1 &&
                         mlir::isa<TupleType>(op->getResult(0).getType());
    bool flattenOperands = llvm::any_of(op.getInputs(), [](Value operand) {
      return mlir::isa<TupleType>(operand.getType());
    });

    if (!flattenResult && !flattenOperands) return failure();

    llvm::SmallVector<Value> flattenedOperands;
    for (auto operand : op.getInputs())
      flattenTupleValue(rewriter, op->getLoc(), operand, flattenedOperands);

    llvm::SmallVector<Type, 4> flattenedResultTypes;
    if (!flattenResult) {
      flattenedResultTypes.push_back(op->getResult(0).getType());
    } else {
      // Check for nested tuples.
      for (Type innerType :
           mlir::cast<TupleType>(op->getResult(0).getType()).getTypes())
        if (mlir::isa<TupleType>(innerType)) return failure();

      for (auto result : op->getResults())
        flattenTupleType(result, flattenedResultTypes);
    }

    auto flattenedCall = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), flattenedResultTypes, flattenedOperands, op->getAttrs());

    rewriter.replaceOp(op, flattenResult
                               ? createTupleValue(rewriter, op->getLoc(),
                                                  flattenedCall.getResults(),
                                                  op->getResult(0).getType())
                               : flattenedCall.getResult(0));
    return success();
  }
};

class StablehloFlattenTuplePass
    : public impl::StablehloFlattenTuplePassBase<StablehloFlattenTuplePass> {
 public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FlattenCustomCallOp>(context);
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification =
        mlir::GreedySimplifyRegionLevel::Disabled;
    config.fold = false;
    config.cseConstants = false;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace stablehlo_ext
}  // namespace mlir
