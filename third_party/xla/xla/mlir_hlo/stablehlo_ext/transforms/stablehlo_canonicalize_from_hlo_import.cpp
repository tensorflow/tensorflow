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

// This file implements logic for some optimizations to reduce size on export.

#include <cassert>
#include <iterator>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "stablehlo_ext/transforms/passes.h"  // NOLINT: Used in passes.h.inc

#define DEBUG_TYPE "stablehlo-ext-canonicalize-from-hlo-import"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_STABLEHLOCANONICALIZEFROMHLOIMPORTPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

/////////////
// Flatten Tuples in entry computation

// Expands the mhlo.tuple used in return op. Also updates function
// signature accordingly.
LogicalResult expandTupledTensorInReturnOp(func::FuncOp func) {
  FunctionType oldFuncType = func.getFunctionType();
  // Update input signatures.
  // We will flatten the tuples for the function inputs as well.
  // So if an input is tuple, will be flattened and packed as following:
  // func_1(%arg0: tuple<input1, input2>) =>
  //
  // func_1(%arg0: <input1>, %arg1: <input2>) {
  //  %0 = mhlo.tuple(%arg0, %arg1)
  // }
  SmallVector<Type, 4> expandedInputTypes;
  SmallVector<BlockArgument, 20> funcArguments(func.getArguments().begin(),
                                               func.getArguments().end());
  for (auto argument : funcArguments) {
    auto type = argument.getType();
    auto tupleType = mlir::dyn_cast_or_null<TupleType>(type);
    if (!tupleType) {
      expandedInputTypes.push_back(type);
    } else {
      // We need to
      // 1) expand the tuple
      // 2) insert a new tuple
      // 3) rewire the new tuple
      int originalArgumentIndex = argument.getArgNumber();
      int argumentIndex = originalArgumentIndex;
      SmallVector<Value, 4> flattenedOperands;
      // insert the flattened tuples after the original tuple.
      Location loc = func.getBody().getLoc();
      for (auto flattenedType : tupleType.getTypes()) {
        expandedInputTypes.push_back(flattenedType);
        if (failed(
                func.insertArgument(++argumentIndex, flattenedType, {}, loc))) {
          return failure();
        }

        flattenedOperands.push_back(func.getArgument(argumentIndex));
      }

      // Construct a new tuple and rewire it.
      OpBuilder builder(func.getBody());
      builder.setInsertionPointToStart(&func.getBody().front());
      auto newTuple =
          builder.create<stablehlo::TupleOp>(loc, tupleType, flattenedOperands);
      func.getArgument(originalArgumentIndex).replaceAllUsesWith(newTuple);

      // Now the original argument has been rewired, we should be able to
      // safely erase it.
      if (failed(func.eraseArgument(originalArgumentIndex))) {
        return failure();
      }
    }
  }

  // Update output signatures.
  auto returnOp = cast<mlir::func::ReturnOp>(func.getBody().back().back());
  OpBuilder builder(returnOp);

  // Expand all tuples in old return operands.
  SmallVector<Value, 4> expandedReturnOperands;
  SmallVector<Type, 4> expandedResultTypes;
  for (auto value : returnOp.getOperands()) {
    if (auto tupleTy = mlir::dyn_cast<TupleType>(value.getType())) {
      llvm::copy(tupleTy.getTypes(), std::back_inserter(expandedResultTypes));
      for (auto [index, ty] : llvm::enumerate(tupleTy.getTypes())) {
        expandedReturnOperands.push_back(
            builder.createOrFold<stablehlo::GetTupleElementOp>(
                value.getLoc(), ty, value, index));
      }
    } else {
      expandedReturnOperands.push_back(value);
      expandedResultTypes.push_back(value.getType());
    }
  }

  if (returnOp.getOperands() == expandedReturnOperands) return success();

  builder.create<mlir::func::ReturnOp>(returnOp.getLoc(),
                                       expandedReturnOperands);
  returnOp.erase();
  auto newFuncType = FunctionType::get(oldFuncType.getContext(),
                                       expandedInputTypes, expandedResultTypes);
  func.setType(newFuncType);
  return success();
}

/////////////
// Flatten Tuples in Custom Calls

// Calculates the flatten types of a value.
void flattenTupleType(Type type, llvm::SmallVectorImpl<Type> &flattenedTypes) {
  auto tupleType = mlir::dyn_cast<mlir::TupleType>(type);
  if (!tupleType) {
    flattenedTypes.push_back(type);
    return;
  }

  for (auto childType : tupleType.getTypes()) {
    flattenTupleType(childType, flattenedTypes);
  }
}

// FlattenTupleValue and CreateTupleValue is a pair of functions to create and
// flatten tuples in the exact same order. CreateTupleValue returns the result
// of the root TupleOp or given value if the type is not TupleType.
Value createTupleValue(OpBuilder &builder, Location loc,
                       ValueRange &flattenValues, Type type) {
  auto tupleType = mlir::dyn_cast<mlir::TupleType>(type);
  if (!tupleType) {
    assert(!flattenValues.empty());
    auto retval = flattenValues.front();
    flattenValues = flattenValues.drop_front();
    return retval;
  }

  SmallVector<Value> flattenedSubValues;
  for (auto childType : tupleType.getTypes()) {
    flattenedSubValues.push_back(
        createTupleValue(builder, loc, flattenValues, childType));
  }

  return builder.create<mlir::stablehlo::TupleOp>(loc, flattenedSubValues)
      .getResult();
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
    // We only flatten a single result tuple, as this is what we expect from
    // HLO, where an instruction can only have a single result.
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
    if (flattenResult) {
      flattenTupleType(op->getResult(0).getType(), flattenedResultTypes);
    } else {
      flattenedResultTypes.append(op->result_type_begin(),
                                  op->result_type_end());
    }

    auto flattenedCall = rewriter.create<stablehlo::CustomCallOp>(
        op->getLoc(), flattenedResultTypes, flattenedOperands, op->getAttrs());

    if (flattenResult) {
      ValueRange flattenedResultsRef(flattenedCall.getResults());
      Value newResult =
          createTupleValue(rewriter, op->getLoc(), flattenedResultsRef,
                           op->getResult(0).getType());
      // Verify all flattened results have been consumed.
      assert(flattenedResultsRef.empty());
      rewriter.replaceOp(op, newResult);
    } else {
      rewriter.replaceOp(op, flattenedCall.getResults());
    }
    return success();
  }
};

// Simplify a model after HLO import.
struct StablehloCanonicalizeFromHloImportPass
    : public impl::StablehloCanonicalizeFromHloImportPassBase<
          StablehloCanonicalizeFromHloImportPass> {
  using StablehloCanonicalizeFromHloImportPassBase::
      StablehloCanonicalizeFromHloImportPassBase;

  void runOnOperation() override {
    // If entry function, flatten the input tuples
    func::FuncOp func = getOperation();
    if (func.getName() == entryFunctionNameOption.getValue()) {
      // Recursively expand tuples until all of them are gone.
      while (
          llvm::any_of(llvm::concat<const Type>(func.getArgumentTypes(),
                                                func.getResultTypes()),
                       [](Type type) { return mlir::isa<TupleType>(type); })) {
        if (failed(expandTupledTensorInReturnOp(func))) {
          return signalPassFailure();
        }
      }
    }

    // Flatten tuples in function body
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<FlattenCustomCallOp>(context);
    stablehlo::populateStablehloHloImportCanonicalizationPatterns(context,
                                                                  &patterns);

    // Apply patterns without folding
    GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true)
        .setRegionSimplificationLevel(mlir::GreedySimplifyRegionLevel::Disabled)
        .enableFolding(false)
        .enableConstantCSE(false);
    if (failed(applyPatternsGreedily(func, std::move(patterns), config)))
      signalPassFailure();
  }
};

}  // end namespace

}  // namespace stablehlo_ext
}  // namespace mlir
