/* Copyright 2026 The OpenXLA Authors.

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
#include <memory>
#include <utility>

#include "passes.h"  // NOLINT: Used in passes.h.inc
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_original_value_util.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/xla_data.pb.h"

#define DEBUG_TYPE "stablehlo-ext-canonicalize-from-hlo-import"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_STABLEHLOCANONICALIZEFROMHLOIMPORTPASS
#include "passes.h.inc"

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
      auto newTuple = stablehlo::TupleOp::create(builder, loc, tupleType,
                                                 flattenedOperands);
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

  mlir::func::ReturnOp::create(builder, returnOp.getLoc(),
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

  return mlir::stablehlo::TupleOp::create(builder, loc, flattenedSubValues)
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
    auto innerValue = stablehlo::GetTupleElementOp::create(
        builder, loc, innerType, value,
        builder.getI32IntegerAttr(flattenIdx++));
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

    auto flattenedCall = stablehlo::CustomCallOp::create(
        rewriter, op->getLoc(), flattenedResultTypes, flattenedOperands,
        op->getAttrs());

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

/////////////////////////////////
// WhileOp
/////////////////////////////////

// This is a copy of the StableHLO pattern with the same name, but specialized
// for the HLO import case to be able to handle original value.
//
// Turn loop invariant values into implicit capture.
// Check if there is at least one value is forwarded from one iteration to
// the next, or one of the yielded value is an implicit capture already.
// Otherwise there is nothing to do here.
//
// Pattern: while -> while (loop invariants as implicit captures)
struct WhileOpImplicitCapture : public OpRewritePattern<stablehlo::WhileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::WhileOp whileOp,
                                PatternRewriter& rewriter) const override {
    Block* cond = whileOp.SingleBlock::getBody(0);
    Block* body = whileOp.SingleBlock::getBody(1);
    auto bodyReturnOp = cast<stablehlo::ReturnOp>(body->getTerminator());
    if (!llvm::any_of(llvm::zip(whileOp->getOperands(), body->getArguments(),
                                bodyReturnOp->getOperands()),
                      [&](auto zip) {
                        return (std::get<0>(zip) == std::get<2>(zip) ||
                                std::get<1>(zip) == std::get<2>(zip));
                      })) {
      return rewriter.notifyMatchFailure(whileOp, "no loop invariant found");
    }

    SmallVector<Value> newOperands, resultsToReplace;
    SmallVector<unsigned> invariantArgIdxs;
    BitVector invariantArgIdxBitVector(cond->getNumArguments());
    for (const auto& enumeratedOperands : llvm::enumerate(llvm::zip(
             whileOp.getOperands(), cond->getArguments(), body->getArguments(),
             bodyReturnOp->getOperands(), whileOp->getResults()))) {
      const auto& operands = enumeratedOperands.value();
      Value whileOperand = std::get<0>(operands);
      BlockArgument condBlockArg = std::get<1>(operands);
      BlockArgument bodyBlockArg = std::get<2>(operands);
      Value bodyReturnOperand = std::get<3>(operands);
      Value whileResult = std::get<4>(operands);

      bool forwarded = (whileOperand == bodyReturnOperand ||
                        bodyBlockArg == bodyReturnOperand);
      if (forwarded) {
        invariantArgIdxs.push_back(enumeratedOperands.index());
        invariantArgIdxBitVector.set(enumeratedOperands.index());
        condBlockArg.replaceAllUsesWith(whileOperand);
        bodyBlockArg.replaceAllUsesWith(whileOperand);
        whileResult.replaceAllUsesWith(whileOperand);
        continue;
      }
      newOperands.push_back(whileOperand);
      resultsToReplace.push_back(whileResult);
    }
    cond->eraseArguments(invariantArgIdxBitVector);
    body->eraseArguments(invariantArgIdxBitVector);
    for (int idx : llvm::reverse(invariantArgIdxs)) {
      bodyReturnOp->eraseOperand(idx);
    }

    stablehlo::WhileOp newWhileOp = stablehlo::WhileOp::create(
        rewriter, whileOp.getLoc(), bodyReturnOp->getOperandTypes(),
        newOperands, whileOp->getAttrs());

    auto copy_original_value = [&](stablehlo::WhileOp whileOp,
                                   stablehlo::WhileOp newWhileOp) {
      std::shared_ptr<xla::OriginalValue> original_value = nullptr;
      auto original_value_attr =
          whileOp->getAttrOfType<StringAttr>("mhlo.original_value");
      if (!original_value_attr) {
        return;
      }
      auto status_or_original_value =
          xla::ParseOriginalValue(original_value_attr.getValue());
      if (!status_or_original_value.ok()) {
        return;
      }
      original_value = status_or_original_value.value();
      if (!original_value || !original_value->IsTuple()) {
        return;
      }

      auto new_shape_or = xla::ExtractXlaShape(newWhileOp);
      if (failed(new_shape_or)) {
        newWhileOp->removeAttr("mhlo.original_value");
        return;
      }
      std::shared_ptr<xla::OriginalValue> new_original_value =
          std::make_shared<xla::OriginalValue>(new_shape_or.value());
      llvm::DenseMap<unsigned int, unsigned int> old_to_new_tuple_idx;
      for (unsigned int i = 0, j = 0; i < whileOp->getNumResults(); ++i) {
        if (!invariantArgIdxBitVector[i]) {
          old_to_new_tuple_idx[i] = j++;
        }
      }
      xla::CopyOriginalValue(original_value, new_original_value,
                             old_to_new_tuple_idx);
      newWhileOp->setAttr(
          "mhlo.original_value",
          rewriter.getStringAttr(new_original_value->ToString()));
    };
    copy_original_value(whileOp, newWhileOp);

    newWhileOp.getBodyRegion(0).takeBody(whileOp.getBodyRegion(0));
    newWhileOp.getBodyRegion(1).takeBody(whileOp.getBodyRegion(1));
    for (auto results : llvm::zip(resultsToReplace, newWhileOp->getResults())) {
      std::get<0>(results).replaceAllUsesWith(std::get<1>(results));
    }
    rewriter.eraseOp(whileOp);
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
    patterns.add<WhileOpImplicitCapture>(context);
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
