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

#include <utility>

#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

struct InferReturnTypesPattern : public RewritePattern {
  explicit InferReturnTypesPattern(MLIRContext *context)
      : RewritePattern("mhlo_test.get_return_types", 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1) return failure();
    auto *definingOp = op->getOperand(0).getDefiningOp();
    auto definingOpInt =
        llvm::dyn_cast_or_null<InferTypeOpInterface>(definingOp);
    if (!definingOpInt) return failure();
    SmallVector<Type, 4> types;
    if (failed(definingOpInt.inferReturnTypes(
            op->getContext(), op->getLoc(), definingOp->getOperands(),
            definingOp->getAttrDictionary(), definingOp->getRegions(),
            types))) {
      return failure();
    }

    // Replace the op with another pass-through op with attributes added.
    OperationState state(op->getLoc(), "mhlo_test.return_types",
                         op->getOperands(), op->getResultTypes(),
                         op->getAttrs());
    auto *newOp = rewriter.create(state);
    for (const auto &it : llvm::enumerate(types)) {
      newOp->setAttr((StringRef("types") + Twine(it.index())).str(),
                     TypeAttr::get(it.value()));
    }
    rewriter.replaceOp(op, {newOp->getResults()});
    return success();
  }
};

struct InferReturnTypeComponentsPattern : public RewritePattern {
  InferReturnTypeComponentsPattern(MLIRContext *context)
      : RewritePattern("mhlo_test.get_return_type_components", 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1) return failure();
    auto *definingOp = op->getOperand(0).getDefiningOp();
    auto definingOpInt =
        llvm::dyn_cast_or_null<InferShapedTypeOpInterface>(definingOp);
    if (!definingOpInt) return failure();
    SmallVector<ShapedTypeComponents, 4> components;
    if (failed(definingOpInt.inferReturnTypeComponents(
            op->getContext(), op->getLoc(), definingOp->getOperands(),
            definingOp->getAttrDictionary(), definingOp->getRegions(),
            components))) {
      return failure();
    }

    // Replace the op with another pass-through op with attributes added.
    OperationState state(op->getLoc(), "mhlo_test.return_type_components",
                         op->getOperands(), op->getResultTypes(),
                         op->getAttrs());
    auto *newOp = rewriter.create(state);
    for (const auto &it : llvm::enumerate(components)) {
      if (it.value().hasRank()) {
        newOp->setAttr((StringRef("dims") + Twine(it.index())).str(),
                       rewriter.getI64ArrayAttr(it.value().getDims()));
      }
      if (it.value().getElementType()) {
        newOp->setAttr((Twine("element_type") + Twine(it.index())).str(),
                       TypeAttr::get(it.value().getElementType()));
      }
    }
    rewriter.replaceOp(op, {newOp->getResults()});
    return success();
  }
};

struct ReifyReturnTypeShapesPattern : public RewritePattern {
  ReifyReturnTypeShapesPattern(MLIRContext *context)
      : RewritePattern("mhlo_test.reify_return_type_shapes", 1, context) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1) return failure();
    auto definingOp =
        op->getOperand(0).getDefiningOp<InferShapedTypeOpInterface>();
    if (!definingOp) return failure();
    SmallVector<Value, 4> returnShapes;
    if (failed(definingOp.reifyReturnTypeShapes(
            rewriter, definingOp->getOperands(), returnShapes))) {
      return failure();
    }
    rewriter.replaceOp(op, returnShapes);
    return success();
  }
};

struct TestInferShapedTypeMethodsPass
    : public TestInferShapedTypeMethodsPassBase<
          TestInferShapedTypeMethodsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<shape::ShapeDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<InferReturnTypesPattern>(&getContext());
    patterns.add<ReifyReturnTypeShapesPattern>(&getContext());
    patterns.add<InferReturnTypeComponentsPattern>(&getContext());
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createTestInferShapedTypeMethodsPass() {
  return std::make_unique<TestInferShapedTypeMethodsPass>();
}

}  // namespace mhlo
}  // namespace mlir
