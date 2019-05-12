//===- ToyCombine.cpp - Toy High Level Optimizer --------------------------===//
//
// Copyright 2019 The MLIR Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================
//
// This file implements a simple combiner for optimizing pattern in the Toy
// dialect.
//
//===----------------------------------------------------------------------===//

#include "toy/Dialect.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"

#include <numeric>

namespace toy {

namespace {

/// Fold transpose(transpose(x)) -> transpose(x)
struct SimplifyRedundantTranspose : public mlir::RewritePattern {
  /// We register this pattern to match every toy.transpose in the IR.
  /// The "benefit" is used by the framework to order the patterns and process
  /// them in order of profitability.
  SimplifyRedundantTranspose(mlir::MLIRContext *context)
      : RewritePattern(TransposeOp::getOperationName(), /* benefit = */ 1,
                       context) {}

  /// This method is attempting to match a pattern and rewrite it. The rewriter
  /// argument is the orchestrator of the sequence of rewrites. It is expected
  /// to interact with it to perform any changes to the IR from here.
  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // We can directly cast the current operation as this will only get invoked
    // on TransposeOp.
    TransposeOp transpose = llvm::cast<TransposeOp>(op);
    // look through the input to the current transpose
    mlir::Value *transposeInput = transpose.getOperand();
    mlir::Operation *transposeInputInst = transposeInput->getDefiningOp();
    // If the input is defined by another Transpose, bingo!
    TransposeOp transposeInputOp =
        mlir::dyn_cast_or_null<TransposeOp>(transposeInputInst);
    if (!transposeInputOp)
      return matchFailure();

    // Use the rewriter to perform the replacement
    rewriter.replaceOp(op, {transposeInputOp.getOperand()}, {transposeInputOp});
    return matchSuccess();
  }
};

/// Fold reshape(constant(x)) -> constant(x'), with x' being reshaped in place.
struct SimplifyReshapeConstant : public mlir::RewritePattern {
  SimplifyReshapeConstant(mlir::MLIRContext *context)
      : RewritePattern(ReshapeOp::getOperationName(), /* benefit = */ 1,
                       context) {}

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    ReshapeOp reshape = llvm::cast<ReshapeOp>(op);
    // look through the input to the current reshape
    mlir::Value *reshapeInput = reshape.getOperand();
    mlir::Operation *reshapeInputInst = reshapeInput->getDefiningOp();
    // If the input is defined by another reshape, bingo!
    ConstantOp constantOp =
        mlir::dyn_cast_or_null<ConstantOp>(reshapeInputInst);
    if (!constantOp)
      return matchFailure();

    auto reshapeType = op->getResult(0)->getType().cast<ToyArrayType>();
    if (auto valueAttr =
            constantOp.getAttrOfType<mlir::DenseElementsAttr>("value")) {
      // FIXME Check matching of element count!
      //      auto oldType = constantOp.getType();
      auto newType = rewriter.getTensorType(
          reshapeType.getShape(), valueAttr.getType().getElementType());
      auto newAttr =
          mlir::DenseElementsAttr::get(newType, valueAttr.getRawData());
      auto newConstant = rewriter.create<ConstantOp>(
          constantOp.getLoc(), reshapeType.getShape(), newAttr);
      rewriter.replaceOp(op, {newConstant});
    } else if (auto valueAttr =
                   constantOp.getAttrOfType<mlir::FloatAttr>("value")) {
      // Broadcast
      auto dataSize = std::accumulate(reshapeType.getShape().begin(),
                                      reshapeType.getShape().end(), 1,
                                      std::multiplies<int>());
      std::vector<mlir::Attribute> data(dataSize, valueAttr);
      auto tensorTy = rewriter.getTensorType(reshapeType.getShape(),
                                             reshapeType.getElementType());
      auto newAttr = mlir::DenseElementsAttr::get(tensorTy, data);
      auto newConstant = rewriter.create<ConstantOp>(
          constantOp.getLoc(), reshapeType.getShape(), newAttr);
      rewriter.replaceOp(op, {newConstant});
    } else {
      llvm_unreachable("Unsupported Constant format");
    }
    return matchSuccess();
  }
};

/// Fold reshape(reshape(x)) -> reshape(x)
struct SimplifyReshapeReshape : public mlir::RewritePattern {
  SimplifyReshapeReshape(mlir::MLIRContext *context)
      : RewritePattern(ReshapeOp::getOperationName(), /* benefit = */ 1,
                       context) {}

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    ReshapeOp reshape = llvm::cast<ReshapeOp>(op);
    // look through the input to the current reshape
    mlir::Value *reshapeInput = reshape.getOperand();
    mlir::Operation *reshapeInputInst = reshapeInput->getDefiningOp();
    // If the input is defined by another reshape, bingo!
    ReshapeOp reshapeInputOp =
        mlir::dyn_cast_or_null<ReshapeOp>(reshapeInputInst);
    if (!reshapeInputOp)
      return matchFailure();

    // Use the rewriter to perform the replacement
    rewriter.replaceOp(op, {reshapeInputOp});
    return matchSuccess();
  }
};

/// Fold reshape(x)) -> x, when input type matches output type
struct SimplifyNullReshape : public mlir::RewritePattern {
  SimplifyNullReshape(mlir::MLIRContext *context)
      : RewritePattern(ReshapeOp::getOperationName(), /* benefit = */ 1,
                       context) {}

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    ReshapeOp reshape = llvm::cast<ReshapeOp>(op);
    if (reshape.getOperand()->getType() != reshape.getResult()->getType())
      return matchFailure();
    rewriter.replaceOp(reshape, {reshape.getOperand()});
    return matchSuccess();
  }
};

} // end anonymous namespace.

// Register our patterns for rewrite by the Canonicalization framework.
void TransposeOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &results, mlir::MLIRContext *context) {
  results.push_back(llvm::make_unique<SimplifyRedundantTranspose>(context));
}

// Register our patterns for rewrite by the Canonicalization framework.
void ReshapeOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &results, mlir::MLIRContext *context) {
  results.push_back(llvm::make_unique<SimplifyReshapeConstant>(context));
  results.push_back(llvm::make_unique<SimplifyReshapeReshape>(context));
  results.push_back(llvm::make_unique<SimplifyNullReshape>(context));
}

namespace {

/// Fold type.cast(x) -> x, when input type matches output type
struct SimplifyIdentityTypeCast : public mlir::RewritePattern {
  SimplifyIdentityTypeCast(mlir::MLIRContext *context)
      : RewritePattern(TypeCastOp::getOperationName(), /* benefit = */ 1,
                       context) {}

  mlir::PatternMatchResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    TypeCastOp typeCast = llvm::cast<TypeCastOp>(op);
    auto resTy = typeCast.getResult()->getType();
    auto *candidateOp = op;
    while (candidateOp && candidateOp->isa<TypeCastOp>()) {
      if (resTy == candidateOp->getOperand(0)->getType()) {
        rewriter.replaceOp(typeCast, {candidateOp->getOperand(0)});
        return matchSuccess();
      }
      candidateOp = candidateOp->getOperand(0)->getDefiningOp();
    }
    return matchFailure();
  }
};

} // end anonymous namespace.

void TypeCastOp::getCanonicalizationPatterns(
    mlir::OwningRewritePatternList &results, mlir::MLIRContext *context) {
  results.push_back(llvm::make_unique<SimplifyIdentityTypeCast>(context));
}

} // namespace toy
