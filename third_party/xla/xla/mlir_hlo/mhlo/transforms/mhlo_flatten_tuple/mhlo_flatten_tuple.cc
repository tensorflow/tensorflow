/* Copyright 2021 The OpenXLA Authors.

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
#include <string>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_FLATTENTUPLEPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

// Calculates the flatten types of a value.
void flattenTupleType(Value value, llvm::SmallVectorImpl<Type> &types) {
  if (!value.getType().isa<TupleType>()) {
    types.push_back(value.getType());
    return;
  }

  // This function doesn't handle nested tuple.
  auto tupleType = value.getType().cast<TupleType>();
  types.append(tupleType.begin(), tupleType.end());
}

// FlattenTupleValue and CreateTupleValue is a pair of functions to create and
// flatten tuples in the exact same order. CreateTupleValue returns the result
// of the root TupleOp or given value if the type is not TupleType.
Value createTupleValue(OpBuilder &builder, Location loc,
                       ValueRange flattenValues, Type tupleType) {
  if (!tupleType.isa<TupleType>()) {
    assert(flattenValues.size() == 1);
    return flattenValues[0];
  }

  assert(tupleType.cast<TupleType>().getTypes().size() == flattenValues.size());
  return builder.create<mhlo::TupleOp>(loc, flattenValues);
}

void flattenTupleValue(OpBuilder &builder, Location loc, Value value,
                       llvm::SmallVectorImpl<Value> &flattenedValues) {
  auto tupleType = value.getType().dyn_cast<TupleType>();
  if (!tupleType) {
    flattenedValues.push_back(value);
    return;
  }
  int flattenIdx = 0;
  for (auto innerType : tupleType.getTypes()) {
    auto innerValue = builder.create<mhlo::GetTupleElementOp>(
        loc, innerType, value, builder.getI32IntegerAttr(flattenIdx++));
    flattenTupleValue(builder, loc, innerValue, flattenedValues);
  }
}

struct FlattenCustomCallOp : public OpRewritePattern<CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CustomCallOp op,
                                PatternRewriter &rewriter) const override {
    bool flattenResult =
        op->getNumResults() == 1 && op->getResult(0).getType().isa<TupleType>();
    bool flattenOperands = llvm::any_of(op.getInputs(), [](Value operand) {
      return operand.getType().isa<TupleType>();
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
           op->getResult(0).getType().cast<TupleType>().getTypes())
        if (innerType.isa<TupleType>()) return failure();

      for (auto result : op->getResults())
        flattenTupleType(result, flattenedResultTypes);
    }

    auto flattenedCall = rewriter.create<mhlo::CustomCallOp>(
        op->getLoc(), flattenedResultTypes, flattenedOperands, op->getAttrs());

    rewriter.replaceOp(op, flattenResult
                               ? createTupleValue(rewriter, op->getLoc(),
                                                  flattenedCall.getResults(),
                                                  op->getResult(0).getType())
                               : flattenedCall.getResult(0));
    return success();
  }
};

class FlattenTuplePass : public impl::FlattenTuplePassBase<FlattenTuplePass> {
 public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FlattenCustomCallOp>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // end namespace

static PassRegistration<FlattenTuplePass> pass;

std::unique_ptr<OperationPass<func::FuncOp>> createFlattenTuplePass() {
  return std::make_unique<FlattenTuplePass>();
}

}  // end namespace mhlo
}  // end namespace mlir
