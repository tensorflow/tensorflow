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

#include <algorithm>

#include "mlir-hlo/Analysis/shape_component_analysis.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {

namespace {

// Returns true if `reshape` only adds `1` dimensions.
bool isExpandShape(ShapeComponentAnalysis &shapeComponentAnalysis,
                   mhlo::DynamicReshapeOp reshape) {
  auto output_shape =
      shapeComponentAnalysis.dimensionsForShapeTensor(reshape.output_shape());
  auto operand_shape =
      shapeComponentAnalysis.dimensionsForShape(reshape.operand());
  if (!output_shape || !operand_shape ||
      output_shape->size() <= operand_shape->size())
    return false;
  int inputDim = 0;
  // Check if the reshape only inserts 1 dimensions and everything else is the
  // same.
  for (const auto &dim : *output_shape) {
    if (dim.isConstant(1)) continue;
    if (inputDim >= operand_shape->size() || dim != (*operand_shape)[inputDim])
      return false;
    ++inputDim;
  }
  return inputDim == operand_shape->size();
}

// Rewrite dynamic reshapes that only insert one dimensions into
// linalg.tensor_expand_shape.
struct ReshapeToExpandShape final
    : public OpRewritePattern<mhlo::DynamicReshapeOp> {
  ReshapeToExpandShape(MLIRContext *ctx,
                       ShapeComponentAnalysis &shapeComponentAnalysis)
      : OpRewritePattern(ctx), shapeComponentAnalysis(shapeComponentAnalysis) {}
  LogicalResult matchAndRewrite(mhlo::DynamicReshapeOp op,
                                PatternRewriter &rewriter) const override {
    if (!isExpandShape(shapeComponentAnalysis, op)) return failure();
    auto output_shape =
        shapeComponentAnalysis.dimensionsForShapeTensor(op.output_shape());
    SmallVector<ReassociationExprs> reassociations(output_shape->size());
    auto it = reassociations.begin();
    int64_t runningIndex = 0;
    for (const auto &dim : *output_shape) {
      it->push_back(rewriter.getAffineDimExpr(runningIndex++));
      if (!dim.isConstant(1)) ++it;
    }
    // If the last dimension was a 1 expand it from the penultimate dim.
    if (output_shape->back().isConstant(1)) std::prev(it)->append(*it);
    reassociations.erase(it, reassociations.end());

    rewriter.replaceOpWithNewOp<linalg::TensorExpandShapeOp>(
        op, op.getResult().getType(), op.operand(), reassociations);
    shapeComponentAnalysis.reset();
    return success();
  }
  ShapeComponentAnalysis &shapeComponentAnalysis;
};

// Remove compute_reshape_shape if we can prove that the dynamic shape does not
// contain a `-1` dimension.
struct RemoveComputeReshapeShape final
    : public OpRewritePattern<mhlo::ComputeReshapeShapeOp> {
  RemoveComputeReshapeShape(MLIRContext *ctx,
                            ShapeComponentAnalysis &shapeComponentAnalysis)
      : OpRewritePattern(ctx), shapeComponentAnalysis(shapeComponentAnalysis) {}
  LogicalResult matchAndRewrite(mhlo::ComputeReshapeShapeOp op,
                                PatternRewriter &rewriter) const override {
    auto dynamic_shape =
        shapeComponentAnalysis.dimensionsForShapeTensor(op.dynamic_shape());
    if (!dynamic_shape) return failure();

    if (llvm::any_of(*dynamic_shape, [](const auto &dim) {
          return !dim.isKnownNotNegativeOne();
        })) {
      return failure();
    }
    rewriter.replaceOp(op, op.dynamic_shape());
    shapeComponentAnalysis.reset();
    return success();
  }
  ShapeComponentAnalysis &shapeComponentAnalysis;
};

struct RemoveRedundantCstrReshapable final
    : public OpRewritePattern<mhlo::CstrReshapableOp> {
  RemoveRedundantCstrReshapable(MLIRContext *ctx,
                                ShapeComponentAnalysis &shapeComponentAnalysis)
      : OpRewritePattern(ctx), shapeComponentAnalysis(shapeComponentAnalysis) {}
  LogicalResult matchAndRewrite(mhlo::CstrReshapableOp op,
                                PatternRewriter &rewriter) const override {
    // Get shape analysis info for the number of elements.
    auto numElementsDims =
        shapeComponentAnalysis.dimensionsForShapeTensor(op.num_elements());
    if (!numElementsDims) return failure();
    assert(numElementsDims->size() == 1 && "expect one value for a scalar");
    auto numElementsDim = numElementsDims->front();

    // Get shape analysis info for the dynamic shape.
    auto dynShapeDims =
        shapeComponentAnalysis.dimensionsForShapeTensor(op.dynamic_shape());
    if (!dynShapeDims) return failure();

    // If any of the dynamic shape's dimensions may be -1, we cannot do
    // anything.
    // TODO(frgossen): We can still eliminate the constraint if there is exactly
    // one -1 and the remaining factors are a subset of the factors for the
    // number of elements.
    if (llvm::any_of(*dynShapeDims, [](const auto &d) {
          return !d.isKnownNotNegativeOne();
        })) {
      return failure();
    }

    // We can only handle simple products with constants and symbols.
    SmallVector<AffineSymbolExpr> symbolicFactors;
    int64_t concreteProductNumElems = 1;
    if (!IsSimpleProduct(numElementsDim.expr, &concreteProductNumElems,
                         &symbolicFactors)) {
      return failure();
    }

    // Find all factors based on the dynamic shape.
    //   - Accumulate the conrete product to later compare it against its
    //     equivalent based on the number of elements.
    //   - Remove symbolic factors from the list and fail if we find an unknown
    //     factor or a factor remains in the list, i.e. if the sets of symbolic
    //     factors differ.
    int64_t concreteProductDynShape = 1;
    for (auto d : *dynShapeDims) {
      if (auto constExpr = d.expr.dyn_cast<AffineConstantExpr>()) {
        concreteProductDynShape *= constExpr.getValue();
        continue;
      }
      if (auto symExpr = d.expr.dyn_cast<AffineSymbolExpr>()) {
        auto symDynShape = d.symbols[symExpr.getPosition()];
        bool isFactorInBothProducts = false;
        for (int i = 0; i < symbolicFactors.size(); ++i) {
          auto symNumElements =
              numElementsDim.symbols[symbolicFactors[i].getPosition()];
          if (symDynShape == symNumElements) {
            symbolicFactors[i] = symbolicFactors.back();
            symbolicFactors.pop_back();
            isFactorInBothProducts = true;
            break;
          }
        }
        if (!isFactorInBothProducts) return failure();
        continue;
      }
      return failure();
    }

    // If symbolic factors remain, we cannot prove the products to be equal.
    if (!symbolicFactors.empty()) return failure();

    // The products can only differ in the concrete factors at this point, which
    // we can evaluate statically.
    bool productsEq = concreteProductNumElems == concreteProductDynShape;
    rewriter.replaceOpWithNewOp<shape::ConstWitnessOp>(op, productsEq);
    return success();
  }
  bool IsSimpleProduct(
      AffineExpr expr, int64_t *concreteProduct,
      SmallVectorImpl<AffineSymbolExpr> *symbolicFactors) const {
    auto binExpr = expr.dyn_cast<AffineBinaryOpExpr>();
    if (binExpr && binExpr.getKind() == AffineExprKind::Mul) {
      return IsSimpleProduct(binExpr.getLHS(), concreteProduct,
                             symbolicFactors) &&
             IsSimpleProduct(binExpr.getRHS(), concreteProduct,
                             symbolicFactors);
    }
    if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
      symbolicFactors->push_back(symExpr);
      return true;
    }
    if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
      *concreteProduct *= constExpr.getValue();
      return true;
    }
    return false;
  }
  ShapeComponentAnalysis &shapeComponentAnalysis;
};

class ReshapeSimplifierPass final
    : public ReshapeSimplifierBase<ReshapeSimplifierPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnFunction() override;

 private:
};
}  // end namespace

void ReshapeSimplifierPass::runOnFunction() {
  MLIRContext *ctx = &getContext();
  mlir::RewritePatternSet patterns(ctx);
  ShapeComponentAnalysis shapeComponentAnalysis;

  // clang-format off
  patterns.insert<ReshapeToExpandShape,
                  RemoveComputeReshapeShape,
                  RemoveRedundantCstrReshapable>(ctx, shapeComponentAnalysis);
  // clang-format on

  if (failed(mlir::applyPatternsAndFoldGreedily(getFunction(),
                                                std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<FunctionPass> createReshapeSimplifierPass() {
  return std::make_unique<ReshapeSimplifierPass>();
}

}  // end namespace mlir
